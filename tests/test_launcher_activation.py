from __future__ import annotations

import os
import subprocess
from pathlib import Path

LAUNCHER = Path(__file__).parents[1] / "scripts" / "ocbrain-mcp"


def _launcher_args(root: Path) -> list[str]:
    env = {
        **os.environ,
        "OCBRAIN_ROOT": str(root),
        "OCBRAIN_PYTHON": "/bin/echo",
    }
    result = subprocess.run(  # noqa: S603 - fixed local test launcher
        [str(LAUNCHER)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip().split()


def test_launcher_defaults_to_repo_database(tmp_path: Path) -> None:
    args = _launcher_args(tmp_path)
    db_index = args.index("--db") + 1
    assert args[db_index] == str(tmp_path / "data" / "ocbrain.sqlite")
    assert args[-1] == "mcp"


def test_launcher_reads_explicit_ignored_activation_pointer(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    core = data / "v1" / "ocbrain-core-v1.sqlite"
    (data / "active-core.path").write_text(f"{core}\n")

    args = _launcher_args(tmp_path)

    db_index = args.index("--db") + 1
    assert args[db_index] == str(core)


def test_launcher_rejects_relative_activation_pointer(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "active-core.path").write_text("relative.sqlite\n")
    env = {
        **os.environ,
        "OCBRAIN_ROOT": str(tmp_path),
        "OCBRAIN_PYTHON": "/bin/echo",
    }

    result = subprocess.run(  # noqa: S603 - fixed local test launcher
        [str(LAUNCHER)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 2
    assert "must contain one absolute database path" in result.stderr
