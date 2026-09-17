from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

LAUNCHER = Path(__file__).parents[1] / "scripts" / "ocbrain-mcp"


def _launcher_args(root: Path, *, extra_env: dict[str, str] | None = None) -> list[str]:
    env = {
        **os.environ,
        "OCBRAIN_ROOT": str(root),
        "OCBRAIN_PYTHON": "/bin/echo",
    }
    env.pop("OCBRAIN_DB", None)
    env.pop("OCBRAIN_ACTIVE_DB_FILE", None)
    if extra_env:
        env.update(extra_env)
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
    guard_index = args.index("--active-db-file") + 1
    assert guard_index > db_index
    assert args[guard_index] == str(data / "active-core.path")


def test_launcher_explicit_database_override_is_not_pointer_guarded(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    pointer_db = data / "v1" / "pointer.sqlite"
    explicit_db = data / "doctor-smoke.sqlite"
    (data / "active-core.path").write_text(f"{pointer_db}\n")

    args = _launcher_args(
        tmp_path,
        extra_env={"OCBRAIN_DB": str(explicit_db)},
    )

    db_index = args.index("--db") + 1
    assert args[db_index] == str(explicit_db)
    assert "--active-db-file" not in args


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


def test_launcher_does_not_impose_idle_timeout_but_preserves_explicit_opt_in(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "print-idle-timeout"
    probe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os\n"
        "print(json.dumps({'timeout': os.environ.get('OCBRAIN_MCP_IDLE_TIMEOUT_SECONDS')}))\n"
    )
    probe.chmod(0o700)

    base_env = {
        **os.environ,
        "OCBRAIN_ROOT": str(tmp_path),
        "OCBRAIN_PYTHON": str(probe),
    }
    base_env.pop("OCBRAIN_DB", None)
    base_env.pop("OCBRAIN_ACTIVE_DB_FILE", None)
    base_env.pop("OCBRAIN_MCP_IDLE_TIMEOUT_SECONDS", None)

    default = subprocess.run(  # noqa: S603 - generated local test probe
        [str(LAUNCHER)],
        check=True,
        capture_output=True,
        text=True,
        env=base_env,
    )
    assert json.loads(default.stdout)["timeout"] is None

    explicit = subprocess.run(  # noqa: S603 - generated local test probe
        [str(LAUNCHER)],
        check=True,
        capture_output=True,
        text=True,
        env={**base_env, "OCBRAIN_MCP_IDLE_TIMEOUT_SECONDS": "900"},
    )
    assert json.loads(explicit.stdout)["timeout"] == "900"


# --------------------------------------------------------------------------- #
# Symlink resolution
# --------------------------------------------------------------------------- #

# ~/.local/bin/ocbrain-mcp is a symlink into a repo checkout. The launcher must
# root itself where the link POINTS, not where the link LIVES; before the chase
# below existed, ROOT resolved to ~/.local and the import failed with
# ModuleNotFoundError. REPO_ROOT is the checkout this test file itself runs in.
REPO_ROOT = LAUNCHER.parents[1].resolve()


def _launcher_via(target: Path, *, extra_env: dict[str, str] | None = None) -> list[str]:
    env = {**os.environ, "OCBRAIN_PYTHON": "/bin/echo"}
    for name in ("OCBRAIN_ROOT", "OCBRAIN_DB", "OCBRAIN_ACTIVE_DB_FILE"):
        env.pop(name, None)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(  # noqa: S603 - fixed local test launcher
        [str(target)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip().split()


def test_launcher_invoked_through_a_symlink_roots_at_the_target_repo(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    link = bin_dir / "ocbrain-mcp"
    link.symlink_to(LAUNCHER)

    args = _launcher_via(link)

    assert args[args.index("--db") + 1] == str(REPO_ROOT / "data" / "ocbrain.sqlite")


def test_launcher_walks_a_chain_of_relative_symlinks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(LAUNCHER, repo / "scripts" / "ocbrain-mcp")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (repo / "ocbrain-mcp-link").symlink_to("scripts/ocbrain-mcp")
    (bin_dir / "ocbrain-mcp").symlink_to(
        os.path.relpath(repo / "ocbrain-mcp-link", bin_dir)
    )

    args = _launcher_via(bin_dir / "ocbrain-mcp")

    assert args[args.index("--db") + 1] == str(repo / "data" / "ocbrain.sqlite")


def test_an_explicit_ocbrain_root_wins_even_through_a_symlink(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    link = bin_dir / "ocbrain-mcp"
    link.symlink_to(LAUNCHER)

    args = _launcher_via(link, extra_env={"OCBRAIN_ROOT": str(tmp_path / "elsewhere")})

    assert args[args.index("--db") + 1] == str(tmp_path / "elsewhere" / "data" / "ocbrain.sqlite")
