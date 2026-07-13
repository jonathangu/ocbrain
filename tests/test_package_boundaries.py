from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from ocbrain import cli
from ocbrain_ops.store import DEFAULT_OPS_DB
from ocbrain_training.store import DEFAULT_TRAINING_DB

ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT / "src"

RETIRED_CORE_MODULES = {
    "autolabel.py",
    "autopilot.py",
    "dream.py",
    "embed.py",
    "excerpt.py",
    "feedback.py",
    "judge.py",
    "loops.py",
    "maintenance.py",
    "promote.py",
    "publicsafety.py",
    "retrieval_eval.py",
    "review.py",
    "safeguards.py",
    "schema.py",
    "stallcheck.py",
    "teacher.py",
}


def _project(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text())["project"]


def test_distribution_metadata_and_console_ownership() -> None:
    core = _project(ROOT / "pyproject.toml")
    training = _project(ROOT / "packages/training/pyproject.toml")
    ops = _project(ROOT / "packages/ops/pyproject.toml")

    assert (core["name"], training["name"], ops["name"]) == (
        "ocbrain",
        "ocbrain-training",
        "ocbrain-ops",
    )
    assert core["version"] == training["version"] == ops["version"] == "1.0.0"
    assert core["scripts"] == {
        "ocbrain": "ocbrain.cli:main",
        "ocbrain-closeout": "ocbrain.cli:main",
    }
    assert training["scripts"] == {"ocbrain-training": "ocbrain_training.cli:main"}
    assert ops["scripts"] == {
        "ocbrain-ops": "ocbrain_ops.cli:main",
        "ocbrain-watchdog": "ocbrain_ops.stallcheck:main",
        "brain-loop-ingest": "ocbrain_ops.cli:loop_ingest_main",
    }
    assert training["dependencies"] == ["ocbrain>=1,<2"]
    assert ops["dependencies"] == ["ocbrain>=1,<2"]


def test_core_source_and_parser_exclude_companion_implementations() -> None:
    core_package = ROOT / "src/ocbrain"
    assert not (core_package / "dataset").exists()
    assert RETIRED_CORE_MODULES.isdisjoint(path.name for path in core_package.glob("*.py"))

    subparsers = next(action for action in cli.build_parser()._actions if action.dest == "command")
    core_commands = set(subparsers.choices)
    assert core_commands.isdisjoint(cli.COMPANION_COMMANDS)
    assert {"init", "status", "sync", "preview", "mcp"} <= core_commands


def test_importing_core_cli_and_mcp_does_not_import_companions() -> None:
    probe = """
import json
import sys
import ocbrain.cli
import ocbrain.mcp
print(json.dumps(sorted(
    name for name in sys.modules
    if name.startswith(('ocbrain_training', 'ocbrain_ops'))
)))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CORE_SRC)
    result = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(CORE_SRC)!r});{probe}"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert json.loads(result.stdout) == []


def test_companion_stores_are_distinct_from_the_core_default() -> None:
    assert DEFAULT_TRAINING_DB.name == "training.sqlite"
    assert DEFAULT_OPS_DB.name == "ops.sqlite"
    assert DEFAULT_TRAINING_DB != DEFAULT_OPS_DB
    assert DEFAULT_TRAINING_DB != cli.DEFAULT_DB_PATH
    assert DEFAULT_OPS_DB != cli.DEFAULT_DB_PATH

