"""Regression: current-schema startup is read-only, while a real migration uses
the core busy_timeout instead of crashing immediately on a concurrent writer.

The heavy autopilot fire crashed exit-1 here: the core connect path set no
busy_timeout while concurrent writers (light cycles, stallcheck, MCP feedback)
routinely hold the write lock.
"""

import sqlite3
import threading
import time

import pytest

from ocbrain.db import DB_BUSY_TIMEOUT_MS, connect, init_db


def test_factory_sets_busy_timeout(tmp_path):
    conn = connect(tmp_path / "brain.sqlite")
    try:
        (value,) = conn.execute("PRAGMA busy_timeout").fetchone()
        assert value == DB_BUSY_TIMEOUT_MS
    finally:
        conn.close()


def test_zero_timeout_fails_fast_under_write_lock(tmp_path):
    """Negative control: without a real wait the migrate write fails immediately,
    proving the lock contention is real and the busy_timeout is load-bearing."""
    db_path = tmp_path / "brain.sqlite"
    init_db(connect(db_path))

    holder = connect(db_path)
    holder.execute("BEGIN IMMEDIATE")  # take the RESERVED write lock
    try:
        fast = connect(db_path)
        fast.execute("PRAGMA busy_timeout=0")  # do not wait at all
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            fast.execute("BEGIN IMMEDIATE")
        fast.close()
    finally:
        holder.rollback()
        holder.close()


def test_migrate_path_waits_out_a_held_lock(tmp_path):
    """A writer holds the lock briefly; a second connection's migrate path
    (init_db) must wait and succeed within a short busy_timeout."""
    db_path = tmp_path / "brain.sqlite"
    seeded = connect(db_path)
    init_db(seeded)
    seeded.execute("DROP VIEW memory")
    seeded.execute(
        "CREATE VIEW memory AS SELECT * FROM knowledge "
        "WHERE status = 'current' AND inject = 1"
    )
    seeded.commit()
    seeded.close()

    result: dict[str, object] = {}
    barrier = threading.Barrier(2)

    def migrate() -> None:
        conn = connect(db_path)
        conn.execute("PRAGMA busy_timeout=3000")  # short, for test speed
        barrier.wait()
        try:
            init_db(conn)  # DROP VIEW / rebuild memory view: needs the write lock
            result["ok"] = True
        except Exception as exc:  # pragma: no cover - failure path
            result["error"] = repr(exc)
        finally:
            conn.close()

    holder = connect(db_path)
    holder.execute("BEGIN IMMEDIATE")  # hold the write lock
    worker = threading.Thread(target=migrate)
    worker.start()
    barrier.wait()  # ensure the worker is about to contend for the lock
    time.sleep(0.3)  # hold well under the 3000ms busy_timeout
    holder.rollback()  # release
    holder.close()

    worker.join(timeout=10)
    assert not worker.is_alive()
    assert result.get("ok") is True, result.get("error")


def test_current_schema_reinit_does_not_need_writer_lock(tmp_path):
    db_path = tmp_path / "brain.sqlite"
    initialized = connect(db_path)
    init_db(initialized)
    initialized.close()

    holder = connect(db_path)
    holder.execute("BEGIN IMMEDIATE")
    try:
        reader = connect(db_path)
        reader.execute("PRAGMA busy_timeout=0")
        init_db(reader)
        reader.close()
    finally:
        holder.rollback()
        holder.close()
