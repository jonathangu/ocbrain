from __future__ import annotations

import sqlite3

import pytest

from ocbrain.db import connect
from ocbrain.fsutil import checkpoint_sqlite_wal
from ocbrain.write_batch import DatasetWriteBatch


def test_dataset_write_batch_commits_at_operation_bound(tmp_path):
    path = tmp_path / "batch.sqlite"
    conn = connect(path)
    conn.execute("CREATE TABLE probe (id INTEGER PRIMARY KEY, value TEXT)")
    conn.commit()
    batch = DatasetWriteBatch(conn, max_operations=2, max_seconds=60)

    batch.ensure()
    conn.execute("INSERT INTO probe (value) VALUES ('one')")
    batch.operation()
    assert conn.in_transaction is True

    batch.ensure()
    conn.execute("INSERT INTO probe (value) VALUES ('two')")
    batch.operation()
    assert conn.in_transaction is False

    metrics = batch.metrics()
    assert metrics["operations"] == 2
    assert metrics["batches_committed"] == 1
    assert metrics["writer_lock_seconds"] >= 0
    assert metrics["max_writer_lock_seconds"] >= 0


def test_dataset_write_batch_keeps_unexpired_multirow_transaction(tmp_path):
    path = tmp_path / "batch.sqlite"
    conn = connect(path)
    conn.execute("CREATE TABLE probe (id INTEGER PRIMARY KEY, value TEXT)")
    conn.commit()
    batch = DatasetWriteBatch(conn, max_operations=3, max_seconds=60)

    for value in ("one", "two"):
        batch.flush_if_expired()
        batch.ensure()
        conn.execute("INSERT INTO probe (value) VALUES (?)", (value,))
        batch.operation()

    assert conn.in_transaction is True
    assert batch.metrics()["batches_committed"] == 0
    batch.flush()
    assert batch.metrics()["operations"] == 2
    assert batch.metrics()["batches_committed"] == 1


def test_checkpoint_requires_writer_exit_and_truncates_wal(tmp_path):
    path = tmp_path / "checkpoint.sqlite"
    conn = connect(path)
    conn.execute("CREATE TABLE probe (value TEXT)")
    conn.commit()
    conn.executemany("INSERT INTO probe VALUES (?)", [("x" * 200,)] * 500)

    with pytest.raises(RuntimeError, match="committed writer"):
        checkpoint_sqlite_wal(conn, path)

    conn.commit()
    result = checkpoint_sqlite_wal(conn, path)
    assert result["status"] == "ok"
    assert result["busy"] == 0
    assert result["wal_bytes_after"] == 0


def test_checkpoint_reports_busy_reader_without_claiming_success(tmp_path):
    path = tmp_path / "busy.sqlite"
    writer = connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE probe (value TEXT)")
    writer.commit()
    reader = sqlite3.connect(path)
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM probe").fetchall()
    writer.execute("INSERT INTO probe VALUES ('new')")
    writer.commit()

    result = checkpoint_sqlite_wal(writer, path)
    assert result["status"] == "busy"
    assert result["busy"] == 1

    reader.rollback()
    reader.close()
    recovered = checkpoint_sqlite_wal(writer, path)
    assert recovered["status"] == "ok"
