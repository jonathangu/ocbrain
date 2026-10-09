from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "brain-sync.sh"


def _plan(home: Path, **overrides: str) -> tuple[list[str], list[str]]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("OCBRAIN_")}
    env.update(
        {
            "HOME": str(home),
            "OCBRAIN_DB": str(home / "core.sqlite"),
            "OCBRAIN_PYTHON": "/nonexistent-python",
            "OCBRAIN_ROOT": str(ROOT),
            **overrides,
        }
    )
    completed = subprocess.run(  # noqa: S603 - fixed repository script
        ["/bin/bash", str(SCRIPT), "--print-plan"],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
    )
    entries = [part.decode() for part in completed.stdout.split(b"\0") if part]
    history = [e.removeprefix("history:") for e in entries if e.startswith("history:")]
    memory = [e.removeprefix("memory:") for e in entries if e.startswith("memory:")]
    return history, memory


def test_import_history_covers_every_existing_claude_seat(tmp_path: Path) -> None:
    for name in (".claude", ".claude-a", ".claude-b"):
        (tmp_path / name / "projects").mkdir(parents=True)
    (tmp_path / ".claude-nodir").mkdir()

    history, _ = _plan(tmp_path)

    claude = [h for h in history if "/.claude" in h]
    assert sorted(claude) == sorted(
        str(tmp_path / name / "projects") for name in (".claude", ".claude-a", ".claude-b")
    )
    assert not any("*" in h for h in history)


def test_no_claude_seats_leaves_no_unmatched_glob(tmp_path: Path) -> None:
    history, _ = _plan(tmp_path)

    assert not any(".claude" in h or "*" in h for h in history)


def test_extra_memory_paths_append_only_when_existing(tmp_path: Path) -> None:
    present = tmp_path / "extra.md"
    present.write_text("x")
    folder = tmp_path / "extra-dir"
    folder.mkdir()
    missing = tmp_path / "missing"

    _, memory = _plan(
        tmp_path,
        OCBRAIN_SYNC_EXTRA_MEMORY_PATHS=f"{present}:{missing}::{folder}",
    )

    assert memory[-2:] == [str(present), str(folder)]
    assert str(missing) not in memory


def test_default_memory_paths_unchanged_without_extras(tmp_path: Path) -> None:
    _, memory = _plan(tmp_path)

    assert memory == [
        str(tmp_path / ".claude" / "CLAUDE.md"),
        str(tmp_path / ".codex" / "AGENTS.md"),
        str(tmp_path / ".hermes" / "SOUL.md"),
        str(tmp_path / ".hermes" / "memories"),
    ]
