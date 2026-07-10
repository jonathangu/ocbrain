from __future__ import annotations

import sqlite3
from pathlib import Path

from ocbrain.db import connect, init_db


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def test_pre_grading_dataset_schema_migrates_additively(tmp_path: Path):
    path = tmp_path / "old.sqlite"
    raw = sqlite3.connect(path)
    raw.executescript(
        """
        CREATE TABLE dataset_examples (
          id TEXT PRIMARY KEY, dataset TEXT, content_hash TEXT, dedup_key TEXT,
          quality_label TEXT, occurred_at TEXT
        );
        CREATE INDEX idx_dsx_dataset_label ON dataset_examples(dataset, quality_label);
        CREATE INDEX idx_dsx_dedup ON dataset_examples(dataset, dedup_key);
        CREATE INDEX idx_dsx_occurred ON dataset_examples(dataset, occurred_at);
        CREATE TABLE dataset_exports (
          id TEXT PRIMARY KEY, ts TEXT, dataset TEXT, path TEXT, min_scope TEXT,
          min_label TEXT, format TEXT, example_count INTEGER, excluded_count INTEGER,
          bytes INTEGER, payload_hash TEXT, manifest_json TEXT, egress_audit_id TEXT
        );
        """
    )
    raw.commit()
    raw.close()

    conn = connect(path)
    init_db(conn)
    assert {
        "grade_score",
        "grade_json",
        "grade_model",
        "grade_prompt_version",
        "graded_at",
    } <= _columns(conn, "dataset_examples")
    assert "min_grade" in _columns(conn, "dataset_exports")
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='dataset_grade_runs'"
    ).fetchone()
    # Re-init stays idempotent.
    before = _columns(conn, "dataset_examples")
    init_db(conn)
    assert _columns(conn, "dataset_examples") == before
