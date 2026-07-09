"""v0.3 additive schema: knowledge embedding columns + embed_runs/projection_cursor tables.

Mirrors the v0.2 migration tests: fresh init has everything, a pre-v0.3 DB migrates
in place without a rebuild and without losing rows, and re-init is idempotent. The
pre-v0.3 seed reuses the canonical schema minus the v0.3 additions.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ocbrain.db import connect, counts, init_db, upsert_knowledge

_V3_KNOWLEDGE_COLS = {"embedding", "embedded_at"}
_V3_TABLES = {"embed_runs", "projection_cursor"}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
    }


# A minimal pre-v0.3 knowledge table: has the v0.2 columns but lacks embedding /
# embedded_at, and the DB has neither v0.3 table. init_db must add them in place.
_PRE_V3_SCHEMA = """
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
  origin TEXT,
  quarantine_reason TEXT,
  quality_label TEXT,
  quality_confidence REAL,
  quality_updated_at TEXT,
  promote_score REAL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""


def _seed_pre_v3(path: Path, *, rows: int = 3) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(_PRE_V3_SCHEMA)
    for i in range(rows):
        conn.execute(
            """
            INSERT INTO knowledge (id, type, subject, predicate, value_text, status,
              gate, inject, privacy_scope, created_at, updated_at)
            VALUES (?, 'value', ?, 'p', 'v', 'current', 'auto', 0, 'workspace',
              '2026-06-01T00:00:00+00:00', '2026-06-01T00:00:00+00:00')
            """,
            (f"know_pre_{i}", f"subject_{i}"),
        )
    conn.commit()
    conn.close()


def test_fresh_init_has_v3_columns_and_tables(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    assert _V3_KNOWLEDGE_COLS <= _columns(conn, "knowledge")
    assert _V3_TABLES <= _tables(conn)


def test_embedding_column_is_blob_and_embedded_at_text(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    decls = {
        row["name"]: row["type"]
        for row in conn.execute("PRAGMA table_info(knowledge)")
        if row["name"] in _V3_KNOWLEDGE_COLS
    }
    assert decls["embedding"] == "BLOB"
    assert decls["embedded_at"] == "TEXT"


def test_migrate_pre_v3_adds_columns_and_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    _seed_pre_v3(db_path)
    conn = connect(db_path)
    assert not (_V3_KNOWLEDGE_COLS <= _columns(conn, "knowledge"))  # precondition

    init_db(conn)

    assert _V3_KNOWLEDGE_COLS <= _columns(conn, "knowledge")
    assert _V3_TABLES <= _tables(conn)


def test_migration_preserves_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    _seed_pre_v3(db_path, rows=4)
    conn = connect(db_path)
    init_db(conn)
    assert counts(conn)["knowledge"] == 4


def test_reinit_idempotent_on_v3(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    before_cols = _columns(conn, "knowledge")
    init_db(conn)
    init_db(conn)
    assert _columns(conn, "knowledge") == before_cols


def test_embedding_roundtrips_bytes(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    kid = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime",
        predicate="fact",
        value_text="v",
        status="current",
    )
    blob = b"\x00\x01\x02\xff"
    conn.execute(
        "UPDATE knowledge SET embedding = ?, embedded_at = ? WHERE id = ?",
        (blob, "2026-07-09T00:00:00+00:00", kid),
    )
    conn.commit()
    row = conn.execute(
        "SELECT embedding, embedded_at FROM knowledge WHERE id = ?", (kid,)
    ).fetchone()
    assert bytes(row["embedding"]) == blob
    assert row["embedded_at"] == "2026-07-09T00:00:00+00:00"


def test_projection_cursor_single_row_constraint(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    conn.execute(
        "INSERT INTO projection_cursor (id, last_event_rowid, updated_at) "
        "VALUES (1, 42, '2026-07-09T00:00:00+00:00')"
    )
    conn.commit()
    row = conn.execute("SELECT last_event_rowid FROM projection_cursor WHERE id = 1").fetchone()
    assert row["last_event_rowid"] == 42
    # id must be pinned to 1 by the CHECK constraint.
    try:
        conn.execute("INSERT INTO projection_cursor (id, last_event_rowid) VALUES (2, 0)")
        raised = False
    except sqlite3.IntegrityError:
        raised = True
    assert raised


def test_embed_runs_insert(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    conn.execute(
        "INSERT INTO embed_runs (id, ts, items, cost_usd, status, error) "
        "VALUES ('er1', '2026-07-09T00:00:00+00:00', 128, 0.0025, 'ok', NULL)"
    )
    conn.commit()
    row = conn.execute("SELECT items, cost_usd, status FROM embed_runs WHERE id='er1'").fetchone()
    assert row["items"] == 128
    assert abs(row["cost_usd"] - 0.0025) < 1e-9
    assert row["status"] == "ok"
