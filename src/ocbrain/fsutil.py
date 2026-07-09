"""Filesystem utilities shared across ocbrain lanes (spec §2.1).

Homes ``file_fingerprint`` and ``history_runtime`` (moved out of ``cli.py`` so
dataset/review/autopilot code never has to import the CLI module — ``cli.py``
re-exports them for back-compat), single-instance locking and SQLite snapshot
helpers used by the autopilot pipeline (spec §4.1 stages 0-1), plus the
run-shared :class:`ParseCache` transcript memo (v0.3 incremental mining).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import pickle
import sqlite3
from collections import OrderedDict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar

from ocbrain.db import DB_BUSY_TIMEOUT_MS

_T = TypeVar("_T")


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
    # Backup source is the live brain DB; wait on a concurrent writer/checkpoint
    # lock rather than fail the snapshot with "database is locked".
    src_conn.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")
    dst_conn = sqlite3.connect(dest_path)
    try:
        with dst_conn:
            src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()
    return dest_path


# --- run-shared transcript parse memo (v0.3 incremental mining) ---------------

_CACHE_DIR_SUFFIX = ".cache"


def db_side_dir(conn: sqlite3.Connection, name: str) -> Path | None:
    """Return a DB-anchored side directory ``<dbfile>.cache/<name>``.

    Durable side artifacts (the parse memo, scratch) live *beside* the SQLite
    file they belong to, so in tests (tmp DBs) they land in the tmp tree and
    NEVER in the live ``data/`` tree. Returns ``None`` for an in-memory / temp
    database (no on-disk anchor), in which case callers keep a memory-only memo.
    """
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    # row = (seq, name, file); tolerate both sqlite3.Row and plain-tuple factories.
    file = row["file"] if isinstance(row, sqlite3.Row) else row[2]
    if not file:
        return None  # ':memory:' or an anonymous temp DB has no file anchor
    base = Path(file)
    return base.with_name(base.name + _CACHE_DIR_SUFFIX) / name


def parse_cache_key(fingerprint: str, params: object) -> str:
    """Compose a memo key from a file ``fingerprint`` plus a parse-params digest.

    Two miners that parse the same file with the SAME options share one entry;
    a miner that parses with different options (e.g. founder-id stamping that
    changes turn collapsing) gets a distinct key, so a shared entry is only ever
    reused when it is byte-identical to what the second miner would produce.
    """
    digest = hashlib.sha256()
    digest.update(fingerprint.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(repr(params).encode("utf-8", errors="replace"))
    return digest.hexdigest()


class _Miss:
    __slots__ = ()


_MISS = _Miss()


class ParseCache:
    """Content-fingerprint-keyed memo for parsed transcripts (v0.3 §mine).

    Within one mining run the SFT / DPO / persona miners each want the SAME
    parsed ``Session`` for a new-or-changed transcript; without a memo the file
    is read and normalized once per miner (up to 3x). This memo collapses that
    to one parse per (file, parse-params) via :meth:`get`. It is backed by a
    bounded in-memory LRU and, when a ``side_dir`` is given, an on-disk pickle
    store so the memo also survives across separate miner *processes*.

    ``parses`` (loader/cache-miss count) and ``hits`` are observable so tests can
    assert that a second, unchanged pass parses zero files.
    """

    def __init__(
        self,
        side_dir: Path | None = None,
        *,
        max_entries: int = 512,
        max_disk_entries: int = 4096,
    ) -> None:
        self._mem: OrderedDict[str, object] = OrderedDict()
        self._side_dir = side_dir
        self._max_entries = max(1, max_entries)
        self._max_disk_entries = max(0, max_disk_entries)
        self.parses = 0  # loader invocations (true cache misses)
        self.hits = 0  # served from memory or disk without re-parsing
        if side_dir is not None:
            try:
                side_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self._side_dir = None

    def get(self, key: str, loader: Callable[[], _T]) -> _T:
        """Return the memoized value for ``key`` or compute it via ``loader``."""
        if key in self._mem:
            self._mem.move_to_end(key)
            self.hits += 1
            return self._mem[key]  # type: ignore[return-value]
        disk = self._load_disk(key)
        if disk is not _MISS:
            self.hits += 1
            self._remember(key, disk)
            return disk  # type: ignore[return-value]
        obj = loader()
        self.parses += 1
        self._remember(key, obj)
        self._store_disk(key, obj)
        return obj

    def _remember(self, key: str, obj: object) -> None:
        self._mem[key] = obj
        self._mem.move_to_end(key)
        while len(self._mem) > self._max_entries:
            self._mem.popitem(last=False)  # evict least-recently-used

    def _disk_path(self, key: str) -> Path | None:
        if self._side_dir is None:
            return None
        return self._side_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.pkl"

    def _load_disk(self, key: str) -> object:
        path = self._disk_path(key)
        if path is None or not path.exists():
            return _MISS
        try:
            with path.open("rb") as handle:
                return pickle.load(handle)
        except (OSError, pickle.UnpicklingError, EOFError, ValueError):
            return _MISS

    def _store_disk(self, key: str, obj: object) -> None:
        path = self._disk_path(key)
        if path is None or self._max_disk_entries == 0:
            return
        try:
            tmp = path.with_suffix(".pkl.tmp")
            with tmp.open("wb") as handle:
                pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
        except (OSError, pickle.PicklingError, TypeError):
            return
        self._evict_disk()

    def _evict_disk(self) -> None:
        if self._side_dir is None:
            return
        try:
            files = sorted(
                self._side_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime
            )
        except OSError:
            return
        for stale in files[: max(0, len(files) - self._max_disk_entries)]:
            with contextlib.suppress(OSError):
                stale.unlink()
