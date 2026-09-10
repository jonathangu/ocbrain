"""OCBrain ships as one distribution.

The v1.1 split into `ocbrain-ops` and `ocbrain-training` companions is gone:
every table those packages wrote stayed empty in production, so the split was
guarding a boundary that carried no traffic. What survives is a single
`ocbrain` wheel whose console script is the only entry point, and these tests
pin that shape so a companion cannot quietly reappear.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from ocbrain import cli

ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT / "src"

# Names that must not come back into `src/ocbrain`. Every one of them was an
# ops or training module deleted in the v2 great deletion.
DELETED_COMPANION_MODULES = {
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
    "review.py",
    "safeguards.py",
    "schema.py",
    "stallcheck.py",
    "teacher.py",
}


def test_distribution_metadata_and_console_ownership() -> None:
    core = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

    assert core["name"] == "ocbrain"
    assert core["scripts"] == {
        "ocbrain": "ocbrain.cli:main",
        "ocbrain-closeout": "ocbrain.cli:main",
    }
    assert not (ROOT / "packages").exists()


def test_core_source_excludes_the_deleted_companion_modules() -> None:
    core_package = ROOT / "src/ocbrain"
    assert not (core_package / "dataset").exists()
    assert DELETED_COMPANION_MODULES.isdisjoint(path.name for path in core_package.glob("*.py"))


def test_every_subcommand_is_served_by_this_package() -> None:
    """No lazy entry-point dispatch survives: the parser is the whole surface."""

    subparsers = next(action for action in cli.build_parser()._actions if action.dest == "command")
    core_commands = set(subparsers.choices)
    assert {"init", "status", "sync", "preview", "mcp", "public-safety-check"} <= core_commands
    assert not hasattr(cli, "COMPANION_COMMANDS")
    assert not hasattr(cli, "dispatch_companion_command")


def test_importing_core_cli_and_mcp_does_not_import_a_companion() -> None:
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


def test_public_safety_hook_calls_the_checked_out_core_cli() -> None:
    hook = (ROOT / "ops/hooks/pre-push").read_text()
    assert "${ROOT}/src" in hook
    assert "-m ocbrain.cli public-safety-check" in hook
    assert "ocbrain_ops" not in hook
