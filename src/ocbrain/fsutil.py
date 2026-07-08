"""Filesystem utilities shared across ocbrain lanes (spec §2.1).

Homes ``file_fingerprint`` and ``history_runtime`` (moved out of ``cli.py`` so
dataset/review/autopilot code never has to import the CLI module — ``cli.py``
re-exports them for back-compat), plus single-instance locking and SQLite
snapshot helpers used by the autopilot pipeline (spec §4.1 stages 0-1).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path


def file_fingerprint(path: Path) -> str:
    """Stable fingerprint over path + size + mtime_ns (append-only source files)."""
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(path).encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(str(stat.st_size).encode())
    digest.update(b"\0")
    digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def history_runtime(path: Path) -> str:
    """Infer the runtime that produced a transcript from its path components."""
    parts = set(path.parts)
    if ".codex" in parts:
        return "codex"
    if ".claude" in parts:
        return "claude"
    if ".openclaw" in parts:
        return "openclaw"
    return "unknown"


@contextlib.contextmanager
def file_lock(path: Path | str) -> Iterator[bool]:
    """Single-instance advisory lock (autopilot stage 0).

    Yields ``True`` if the exclusive, non-blocking lock was acquired and
    ``False`` if another holder already owns it (caller should exit quietly).
    The lock file is created if absent and released on context exit.
    """
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+")  # noqa: SIM115 - closed in finally
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def snapshot_sqlite(src: Path | str, dest: Path | str) -> Path:
    """Checkpoint the WAL and copy a SQLite DB to ``dest`` (autopilot stage 1).

    Attempts ``PRAGMA wal_checkpoint(TRUNCATE)`` so the copy is a single
    self-contained file; if a reader blocks the checkpoint, falls back to
    copying db + ``-wal`` + ``-shm`` sidecars together so no committed data is
    lost. Returns the destination path.
    """
    src_path = Path(src)
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    checkpointed = False
    conn = sqlite3.connect(src_path)
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        # row == (busy, log, checkpointed); busy==0 means the checkpoint ran clean.
        checkpointed = bool(row) and row[0] == 0
    except sqlite3.OperationalError:
        checkpointed = False
    finally:
        conn.close()

    shutil.copyfile(src_path, dest_path)
    if not checkpointed:
        for suffix in ("-wal", "-shm"):
            sidecar = src_path.with_name(src_path.name + suffix)
            if sidecar.exists():
                shutil.copyfile(sidecar, dest_path.with_name(dest_path.name + suffix))
    return dest_path
