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
    """Copy a SQLite DB to ``dest`` via the online backup API (autopilot stage 1).

    Uses SQLite's backup API from a dedicated read connection rather than
    ``PRAGMA wal_checkpoint(TRUNCATE)`` + ``shutil.copyfile``. The backup API
    reads a transactionally consistent image even while another connection (the
    autopilot's own ``conn``) is live, so it can never produce a torn copy and,
    critically, never truncates the source WAL out from under that live
    connection — the failure mode that could leave a shared handle poisoned.
    ``dest`` is written as a clean, self-contained database. Returns ``dest``.
    """
    src_path = Path(src)
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # A backup targets the main file; clear any leftover dest + sidecars so the
    # copy is not blended with the remnants of an interrupted prior snapshot.
    for suffix in ("", "-wal", "-shm"):
        stale = dest_path.with_name(dest_path.name + suffix)
        if stale.exists():
            stale.unlink()

    src_conn = sqlite3.connect(src_path)
    dst_conn = sqlite3.connect(dest_path)
    try:
        with dst_conn:
            src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()
    return dest_path
