from __future__ import annotations

import sqlite3
from pathlib import Path

from ocbrain.db import connect, counts, init_db, list_current_knowledge, upsert_knowledge

_V2_KNOWLEDGE_COLS = {
    "origin",
    "quarantine_reason",
    "quality_label",
    "quality_confidence",
    "quality_updated_at",
    "promote_score",
}
_V2_EVIDENCE_COLS = {"injection_scan_status", "injection_scan_hits"}
_V2_TABLES = {
    "signal_events",
    "harvest_watermarks",
    "judge_runs",
    "autopilot_runs",
    "dataset_examples",
    "dataset_sources",
    "dataset_exports",
}

# A pre-v0.2 schema: the knowledge/evidence tables lack the additive v2 columns
# and the memory view has no quarantine predicate. init_db must migrate this in
# place without a rebuild and without losing rows.
_OLD_SCHEMA = """
CREATE TABLE evidence (
  id TEXT PRIMARY KEY,
  source_type TEXT NOT NULL,
  source_runtime TEXT,
  source_uri TEXT,
  content_hash TEXT NOT NULL,
  claim TEXT NOT NULL,
  artifact_uri TEXT,
  artifact_hash TEXT,
  verifier_status TEXT DEFAULT 'unknown',
  loop_tags TEXT,
  project TEXT,
  privacy_scope TEXT DEFAULT 'workspace',
  occurred_at TEXT,
  ingested_at TEXT NOT NULL,
  UNIQUE(source_uri, content_hash)
);
CREATE TABLE knowledge (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  subject TEXT,
  predicate TEXT,
  value_numeric REAL,
  value_text TEXT,
  value_bool INTEGER,
  unit TEXT,
  target_value REAL,
  slug TEXT,
  title TEXT,
  body_uri TEXT,
  doc_kind TEXT,
  status TEXT DEFAULT 'candidate',
  superseded_by TEXT,
  invalidation_reason TEXT,
  gate TEXT NOT NULL,
  prescriptive INTEGER DEFAULT 0,
  inject INTEGER DEFAULT 0,
  risk TEXT DEFAULT 'low',
  confidence REAL,
  content_hash TEXT,
  loop_tags TEXT,
  project TEXT,
  privacy_scope TEXT DEFAULT 'workspace',
  approved_by TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE VIEW memory AS
 SELECT id, type, subject, predicate, title, body_uri, project, privacy_scope
 FROM knowledge
 WHERE status='current' AND inject=1;
"""


def _seed_old_db(path: Path, *, rows: int = 3) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(_OLD_SCHEMA)
    for i in range(rows):
        conn.execute(
            """
            INSERT INTO knowledge (id, type, subject, predicate, value_text, status,
              gate, inject, privacy_scope, created_at, updated_at)
            VALUES (?, 'value', ?, 'p', 'v', 'current', 'auto', 0, 'workspace',
              '2026-06-01T00:00:00+00:00', '2026-06-01T00:00:00+00:00')
            """,
            (f"know_old_{i}", f"subject_{i}"),
        )
        conn.execute(
            """
            INSERT INTO evidence (id, source_type, content_hash, claim, ingested_at)
            VALUES (?, 'closeout', ?, 'legacy claim', '2026-06-01T00:00:00+00:00')
            """,
            (f"evd_old_{i}", f"hash_{i}"),
        )
    conn.commit()
    conn.close()


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
    }


def test_fresh_init_has_v2_columns(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    assert _V2_KNOWLEDGE_COLS <= _columns(conn, "knowledge")
    assert _V2_EVIDENCE_COLS <= _columns(conn, "evidence")


def test_fresh_init_has_v2_tables(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    assert _V2_TABLES <= _tables(conn)


def test_migrate_existing_adds_columns_and_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    _seed_old_db(db_path)
    conn = connect(db_path)
    assert not (_V2_KNOWLEDGE_COLS <= _columns(conn, "knowledge"))  # precondition

    init_db(conn)

    assert _V2_KNOWLEDGE_COLS <= _columns(conn, "knowledge")
    assert _V2_EVIDENCE_COLS <= _columns(conn, "evidence")
    assert _V2_TABLES <= _tables(conn)


def test_migration_preserves_seeded_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    _seed_old_db(db_path, rows=5)
    conn = connect(db_path)

    init_db(conn)

    summary = counts(conn)
    assert summary["knowledge"] == 5
    assert summary["evidence"] == 5


def test_fresh_and_migrated_schema_parity(tmp_path: Path) -> None:
    fresh = connect(tmp_path / "fresh.sqlite")
    init_db(fresh)

    migrated_path = tmp_path / "migrated.sqlite"
    _seed_old_db(migrated_path)
    migrated = connect(migrated_path)
    init_db(migrated)

    assert _columns(fresh, "knowledge") == _columns(migrated, "knowledge")
    assert _columns(fresh, "evidence") == _columns(migrated, "evidence")


def test_ensure_column_idempotent(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    before = _columns(conn, "knowledge")
    init_db(conn)  # second run must not duplicate or error
    init_db(conn)  # third for good measure
    assert _columns(conn, "knowledge") == before


def test_reinit_is_a_noop_on_counts(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime",
        predicate="fact",
        value_text="v",
        status="current",
    )
    conn.commit()
    before = counts(conn)
    init_db(conn)
    assert counts(conn) == before


def test_memory_view_excludes_quarantined(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime",
        predicate="fact",
        value_bool=True,
        status="current",
        inject=True,
    )
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0] == 1

    conn.execute(
        "UPDATE knowledge SET quarantine_reason = 'injection_suspected' WHERE id = ?",
        (knowledge_id,),
    )
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0] == 0


def test_list_current_knowledge_excludes_quarantined(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime",
        predicate="fact",
        value_text="v",
        status="current",
    )
    conn.execute(
        "UPDATE knowledge SET quarantine_reason = 'secret_leak' WHERE id = ?",
        (knowledge_id,),
    )
    conn.commit()
    ids = [row["id"] for row in list_current_knowledge(conn)]
    assert knowledge_id not in ids


def test_legacy_drop_block_still_burns_down_old_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE events (id TEXT)")
    conn.execute("CREATE TABLE candidate_decisions (id TEXT)")
    conn.commit()
    conn.close()

    conn = connect(db_path)
    init_db(conn)
    names = _tables(conn)
    assert "events" not in names
    assert "candidate_decisions" not in names
    assert "memory" in names
