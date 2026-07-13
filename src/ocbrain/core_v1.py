"""Event-authoritative OCBrain v1 core.

``brain_events`` is the only durable semantic authority in this schema.  The
evidence, belief, provenance, search, and retrieval-item tables are projections
that can be rebuilt from the event chain.  Receipt/audit tables remain separate
append-only ledgers because they describe delivery, not durable beliefs.

This module deliberately imports no training, hosted-model, autopilot, loop, or
watchdog code.  It is safe for the default MCP runtime to import directly.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ocbrain.ids import stable_id
from ocbrain.scope import ScopeContext, ScopeTag, scope_match

CORE_V1_APPLICATION_ID = 0x4F434231  # ASCII-ish "OCB1"
CORE_V1_USER_VERSION = 10_000
CORE_V1_SCHEMA_VERSION = "ocbrain.core.v1"
CORE_V1_EVENT_SCHEMA = "ocbrain.event.v1"

LEGACY_IMPORT_KINDS = {
    "legacy_evidence_imported",
    "legacy_knowledge_imported",
    "legacy_signal_imported",
    "retrieval_snapshot_imported",
}

CORE_V1_TABLES: tuple[str, ...] = (
    "schema_meta",
    "brain_events",
    "evidence_objects",
    "current_beliefs",
    "belief_evidence",
    "object_aliases",
    "projection_cursor",
    "retrieval_uses",
    "retrieval_items",
    "egress_audits",
    "context_source_handles",
    "context_source_handle_issues",
    "task_closeouts",
    "task_closeout_retrievals",
    "search_documents",
    "search_index",
)

# SQLite creates these implementation tables for the one FTS5 virtual table.
CORE_V1_FTS_TABLES: frozenset[str] = frozenset(
    {
        "search_index_data",
        "search_index_idx",
        "search_index_docsize",
        "search_index_config",
    }
)

CORE_V1_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS brain_events (
  event_seq INTEGER PRIMARY KEY,
  id TEXT NOT NULL UNIQUE,
  ts TEXT NOT NULL,
  kind TEXT NOT NULL,
  writer TEXT NOT NULL,
  session_id TEXT,
  body_json TEXT NOT NULL,
  body_hash TEXT NOT NULL,
  prev_hash TEXT,
  event_hash TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_brain_events_kind_ts ON brain_events(kind, ts);
CREATE INDEX IF NOT EXISTS idx_brain_events_subject
  ON brain_events(json_extract(body_json, '$.subject.id'));
CREATE INDEX IF NOT EXISTS idx_brain_events_legacy_target
  ON brain_events(json_extract(body_json, '$.target_id'));
CREATE INDEX IF NOT EXISTS idx_brain_events_tombstone_target
  ON brain_events(json_extract(body_json, '$.target'))
  WHERE kind='tombstone_recorded';

CREATE TRIGGER IF NOT EXISTS brain_events_no_update
BEFORE UPDATE ON brain_events BEGIN
  SELECT RAISE(ABORT, 'brain_events is append-only');
END;
CREATE TRIGGER IF NOT EXISTS brain_events_no_delete
BEFORE DELETE ON brain_events BEGIN
  SELECT RAISE(ABORT, 'brain_events is append-only');
END;

CREATE TABLE IF NOT EXISTS evidence_objects (
  evidence_id TEXT PRIMARY KEY,
  body TEXT NOT NULL,
  kind TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  source_content_hash TEXT,
  source_type TEXT,
  source_runtime TEXT,
  source_uri TEXT,
  artifact_uri TEXT,
  artifact_hash TEXT,
  verifier_status TEXT,
  occurred_at TEXT,
  recorded_at TEXT NOT NULL,
  scope_type TEXT NOT NULL,
  scope_id TEXT NOT NULL,
  visibility TEXT NOT NULL,
  egress_policy TEXT NOT NULL,
  scope_provenance TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  recorded_event_id TEXT NOT NULL REFERENCES brain_events(id)
);
CREATE INDEX IF NOT EXISTS idx_evidence_objects_scope
  ON evidence_objects(scope_type, scope_id);

CREATE TABLE IF NOT EXISTS current_beliefs (
  belief_id TEXT PRIMARY KEY,
  body TEXT NOT NULL,
  belief_type TEXT,
  attributes_json TEXT NOT NULL DEFAULT '{}',
  scope_type TEXT NOT NULL,
  scope_id TEXT NOT NULL,
  visibility TEXT NOT NULL,
  egress_policy TEXT NOT NULL,
  scope_provenance TEXT NOT NULL DEFAULT 'explicit',
  confidence REAL,
  confidence_band TEXT,
  evidence_ids TEXT NOT NULL,
  status TEXT NOT NULL,
  serve INTEGER NOT NULL DEFAULT 0 CHECK (serve IN (0, 1)),
  pinned INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0, 1)),
  approved_event_id TEXT REFERENCES brain_events(id),
  last_event_id TEXT NOT NULL REFERENCES brain_events(id),
  last_compiled_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_current_beliefs_scope
  ON current_beliefs(scope_type, scope_id, status, serve);

CREATE TABLE IF NOT EXISTS belief_evidence (
  belief_id TEXT NOT NULL REFERENCES current_beliefs(belief_id),
  evidence_id TEXT NOT NULL,
  relation TEXT NOT NULL,
  created_at TEXT NOT NULL,
  source_event_id TEXT NOT NULL REFERENCES brain_events(id),
  PRIMARY KEY (belief_id, evidence_id, relation)
);
CREATE INDEX IF NOT EXISTS idx_belief_evidence_evidence
  ON belief_evidence(evidence_id, belief_id);

CREATE TABLE IF NOT EXISTS object_aliases (
  alias_id TEXT PRIMARY KEY,
  canonical_id TEXT NOT NULL,
  object_kind TEXT NOT NULL,
  source TEXT NOT NULL,
  source_event_id TEXT NOT NULL REFERENCES brain_events(id)
);
CREATE INDEX IF NOT EXISTS idx_object_aliases_canonical
  ON object_aliases(canonical_id);

CREATE TABLE IF NOT EXISTS projection_cursor (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  last_event_rowid INTEGER NOT NULL,
  last_event_hash TEXT,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retrieval_uses (
  id TEXT PRIMARY KEY,
  knowledge_id TEXT,
  served_to_runtime TEXT,
  task_ref TEXT,
  affected_decision INTEGER,
  corrected INTEGER,
  outcome TEXT NOT NULL DEFAULT 'unknown',
  note TEXT,
  query_text TEXT,
  served_ids_json TEXT,
  context_json TEXT,
  packet_schema TEXT,
  session_id TEXT,
  feedback_source TEXT,
  feedback_at TEXT,
  served_at TEXT NOT NULL,
  source_event_id TEXT REFERENCES brain_events(id)
);
CREATE INDEX IF NOT EXISTS idx_retrieval_uses_outcome_served
  ON retrieval_uses(outcome, served_at);

CREATE TABLE IF NOT EXISTS retrieval_items (
  retrieval_use_id TEXT NOT NULL REFERENCES retrieval_uses(id),
  object_id TEXT NOT NULL,
  object_kind TEXT NOT NULL,
  rank INTEGER NOT NULL,
  score REAL,
  PRIMARY KEY (retrieval_use_id, rank)
);
CREATE INDEX IF NOT EXISTS idx_retrieval_items_object
  ON retrieval_items(object_id, retrieval_use_id);

CREATE TABLE IF NOT EXISTS egress_audits (
  id TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  target TEXT NOT NULL,
  context_json TEXT NOT NULL,
  query TEXT,
  included_json TEXT NOT NULL,
  rejected_json TEXT NOT NULL,
  payload_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS context_source_handles (
  id TEXT PRIMARY KEY,
  issued_at TEXT NOT NULL,
  retrieval_use_id TEXT REFERENCES retrieval_uses(id),
  object_id TEXT NOT NULL,
  source_kind TEXT NOT NULL,
  uri TEXT,
  content_hash TEXT NOT NULL,
  scope_json TEXT NOT NULL,
  locator_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_context_source_handles_retrieval
  ON context_source_handles(retrieval_use_id);

CREATE TABLE IF NOT EXISTS context_source_handle_issues (
  source_id TEXT NOT NULL REFERENCES context_source_handles(id),
  retrieval_use_id TEXT NOT NULL REFERENCES retrieval_uses(id),
  issued_at TEXT NOT NULL,
  PRIMARY KEY (source_id, retrieval_use_id)
);
CREATE TRIGGER IF NOT EXISTS context_source_handle_issues_no_update
BEFORE UPDATE ON context_source_handle_issues BEGIN
  SELECT RAISE(ABORT, 'context_source_handle_issues is append-only');
END;
CREATE TRIGGER IF NOT EXISTS context_source_handle_issues_no_delete
BEFORE DELETE ON context_source_handle_issues BEGIN
  SELECT RAISE(ABORT, 'context_source_handle_issues is append-only');
END;

CREATE TABLE IF NOT EXISTS task_closeouts (
  id TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL,
  closed_at TEXT NOT NULL,
  task_ref TEXT NOT NULL,
  status TEXT NOT NULL,
  summary TEXT NOT NULL,
  decision_impact TEXT NOT NULL,
  decision_note TEXT,
  awaiting TEXT,
  runtime TEXT,
  session_id TEXT,
  context_json TEXT NOT NULL,
  artifact_refs_json TEXT NOT NULL,
  verifier_refs_json TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  receipt_json TEXT NOT NULL,
  content_hash TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS task_closeout_retrievals (
  closeout_id TEXT NOT NULL REFERENCES task_closeouts(id),
  retrieval_use_id TEXT NOT NULL REFERENCES retrieval_uses(id),
  PRIMARY KEY (closeout_id, retrieval_use_id)
);
CREATE INDEX IF NOT EXISTS idx_task_closeout_retrievals_retrieval
  ON task_closeout_retrievals(retrieval_use_id);
CREATE TRIGGER IF NOT EXISTS task_closeouts_no_update
BEFORE UPDATE ON task_closeouts BEGIN
  SELECT RAISE(ABORT, 'task_closeouts is append-only');
END;
CREATE TRIGGER IF NOT EXISTS task_closeouts_no_delete
BEFORE DELETE ON task_closeouts BEGIN
  SELECT RAISE(ABORT, 'task_closeouts is append-only');
END;
CREATE TRIGGER IF NOT EXISTS task_closeout_retrievals_no_update
BEFORE UPDATE ON task_closeout_retrievals BEGIN
  SELECT RAISE(ABORT, 'task_closeout_retrievals is append-only');
END;
CREATE TRIGGER IF NOT EXISTS task_closeout_retrievals_no_delete
BEFORE DELETE ON task_closeout_retrievals BEGIN
  SELECT RAISE(ABORT, 'task_closeout_retrievals is append-only');
END;

CREATE TABLE IF NOT EXISTS search_documents (
  doc_id TEXT NOT NULL UNIQUE,
  kind,
  title,
  body,
  path
);

CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
  kind,
  title,
  body,
  content='search_documents',
  content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS search_documents_ai AFTER INSERT ON search_documents BEGIN
  INSERT INTO search_index(rowid, kind, title, body)
  VALUES (new.rowid, new.kind, new.title, new.body);
END;
CREATE TRIGGER IF NOT EXISTS search_documents_ad AFTER DELETE ON search_documents BEGIN
  INSERT INTO search_index(search_index, rowid, kind, title, body)
  VALUES ('delete', old.rowid, old.kind, old.title, old.body);
END;
CREATE TRIGGER IF NOT EXISTS search_documents_au AFTER UPDATE ON search_documents BEGIN
  INSERT INTO search_index(search_index, rowid, kind, title, body)
  VALUES ('delete', old.rowid, old.kind, old.title, old.body);
  INSERT INTO search_index(rowid, kind, title, body)
  VALUES (new.rowid, new.kind, new.title, new.body);
END;
"""

_SEARCH_TRIGGER_NAMES = (
    "search_documents_ai",
    "search_documents_ad",
    "search_documents_au",
)

_SEARCH_TRIGGER_SCHEMA = """
CREATE TRIGGER IF NOT EXISTS search_documents_ai AFTER INSERT ON search_documents BEGIN
  INSERT INTO search_index(rowid, kind, title, body)
  VALUES (new.rowid, new.kind, new.title, new.body);
END;
CREATE TRIGGER IF NOT EXISTS search_documents_ad AFTER DELETE ON search_documents BEGIN
  INSERT INTO search_index(search_index, rowid, kind, title, body)
  VALUES ('delete', old.rowid, old.kind, old.title, old.body);
END;
CREATE TRIGGER IF NOT EXISTS search_documents_au AFTER UPDATE ON search_documents BEGIN
  INSERT INTO search_index(search_index, rowid, kind, title, body)
  VALUES ('delete', old.rowid, old.kind, old.title, old.body);
  INSERT INTO search_index(rowid, kind, title, body)
  VALUES (new.rowid, new.kind, new.title, new.body);
END;
"""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def is_core_v1(conn: sqlite3.Connection) -> bool:
    """Return true only for an explicitly initialized v1 core."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_meta'"
    ).fetchone()
    if row is None:
        return False
    version = conn.execute(
        "SELECT value FROM schema_meta WHERE key='core_schema'"
    ).fetchone()
    return version is not None and str(version[0]) == CORE_V1_SCHEMA_VERSION


def init_core_v1(conn: sqlite3.Connection) -> None:
    """Initialize a fresh v1 core; refuse to layer it over legacy tables."""
    if is_core_v1(conn):
        assert_core_v1_inventory(conn)
        conn.executescript(CORE_V1_SCHEMA)
        return
    existing = [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    if existing:
        raise ValueError(
            "refusing to initialize v1 over an existing schema: " + ", ".join(existing[:8])
        )
    conn.executescript(CORE_V1_SCHEMA)
    conn.executemany(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
        (
            ("core_schema", CORE_V1_SCHEMA_VERSION),
            ("semantic_authority", "brain_events"),
            ("automatic_activation", "false"),
        ),
    )
    conn.execute(f"PRAGMA application_id={CORE_V1_APPLICATION_ID}")
    conn.execute(f"PRAGMA user_version={CORE_V1_USER_VERSION}")
    conn.commit()


def core_v1_table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def assert_core_v1_inventory(conn: sqlite3.Connection) -> None:
    """Reject accidental legacy/companion tables in a purported v1 core."""
    if not is_core_v1(conn):
        raise ValueError("database is not an OCBrain v1 core")
    actual = core_v1_table_names(conn)
    expected = set(CORE_V1_TABLES) | set(CORE_V1_FTS_TABLES)
    unexpected = sorted(actual - expected)
    missing = sorted(set(CORE_V1_TABLES) - actual)
    if unexpected or missing:
        raise RuntimeError(
            f"v1 schema inventory mismatch: unexpected={unexpected}; missing={missing}"
        )


def set_core_v1_search_triggers(conn: sqlite3.Connection, *, enabled: bool) -> None:
    """Suspend FTS maintenance for a bulk fold, or restore runtime triggers."""
    for name in _SEARCH_TRIGGER_NAMES:
        conn.execute(f'DROP TRIGGER IF EXISTS "{name}"')  # noqa: S608 - fixed allow-list
    if enabled:
        conn.executescript(_SEARCH_TRIGGER_SCHEMA)


def rebuild_core_v1_search(conn: sqlite3.Connection) -> None:
    """Bulk-rebuild and verify the external-content FTS index."""
    conn.execute("INSERT INTO search_index(search_index) VALUES ('rebuild')")
    conn.execute("INSERT INTO search_index(search_index) VALUES ('integrity-check')")


def append_core_event(
    conn: sqlite3.Connection,
    kind: str,
    body: dict[str, Any],
    *,
    writer: str = "ocbrain",
    session_id: str | None = None,
    ts: str | None = None,
    project: bool = False,
) -> str:
    """Append one hash-chained event using the legacy-compatible hash recipe."""
    if not is_core_v1(conn):
        raise ValueError("append_core_event requires an OCBrain v1 core")
    # Reading the head before obtaining SQLite's writer reservation lets two
    # otherwise successful connections append different children of the same
    # ``prev_hash``. Acquire the reservation first, without committing any
    # caller-owned transaction. ``BEGIN IMMEDIATE`` observes busy_timeout; the
    # no-match UPDATE upgrades an existing deferred transaction without
    # changing metadata or ``changes()``.
    began_autocommit_transaction = not conn.in_transaction and conn.isolation_level is None
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    else:
        conn.execute(
            "UPDATE schema_meta SET value=value WHERE key='__event_writer_reservation__'"
        )
    try:
        timestamp = ts or now_iso()
        body_json = canonical_json(body)
        body_hash = sha256_text(body_json)
        prior = conn.execute(
            "SELECT event_hash FROM brain_events ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        prev_hash = str(prior[0]) if prior else None
        event_hash = sha256_text(
            canonical_json(
                {
                    "ts": timestamp,
                    "kind": kind,
                    "writer": writer,
                    "session_id": session_id,
                    "body_hash": body_hash,
                    "prev_hash": prev_hash,
                }
            )
        )
        event_id = stable_id("evt", kind, event_hash)
        conn.execute(
            """
            INSERT INTO brain_events(
              id, ts, kind, writer, session_id, body_json, body_hash, prev_hash, event_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                timestamp,
                kind,
                writer,
                session_id,
                body_json,
                body_hash,
                prev_hash,
                event_hash,
            ),
        )
        if project:
            project_core_v1(conn)
        if began_autocommit_transaction:
            conn.commit()
        return event_id
    except Exception:
        if began_autocommit_transaction and conn.in_transaction:
            conn.rollback()
        raise


def verify_event_chain(
    conn: sqlite3.Connection, *, through_rowid: int | None = None
) -> dict[str, Any]:
    """Verify body hashes, links, and event hashes without mutating state."""
    sql = "SELECT rowid AS rid, * FROM brain_events"
    params: tuple[Any, ...] = ()
    if through_rowid is not None:
        sql += " WHERE rowid <= ?"
        params = (through_rowid,)
    sql += " ORDER BY rowid"
    previous: str | None = None
    count = 0
    last_rowid = 0
    for row in conn.execute(sql, params):
        count += 1
        last_rowid = int(row["rid"])
        if row["prev_hash"] != previous:
            return _chain_failure(count, last_rowid, "prev_hash_mismatch")
        if row["body_hash"] != sha256_text(str(row["body_json"])):
            return _chain_failure(count, last_rowid, "body_hash_mismatch")
        expected = sha256_text(
            canonical_json(
                {
                    "ts": row["ts"],
                    "kind": row["kind"],
                    "writer": row["writer"],
                    "session_id": row["session_id"],
                    "body_hash": row["body_hash"],
                    "prev_hash": row["prev_hash"],
                }
            )
        )
        if row["event_hash"] != expected:
            return _chain_failure(count, last_rowid, "event_hash_mismatch")
        previous = str(row["event_hash"])
    return {
        "verified": True,
        "events": count,
        "last_rowid": last_rowid,
        "last_event_hash": previous,
    }


def _chain_failure(position: int, rowid: int, reason: str) -> dict[str, Any]:
    return {
        "verified": False,
        "events": position,
        "last_rowid": rowid,
        "reason": reason,
    }


def project_core_v1(conn: sqlite3.Connection, *, full: bool = False) -> dict[str, Any]:
    """Fold new events into every semantic projection in one transaction."""
    if not is_core_v1(conn):
        raise ValueError("project_core_v1 requires an OCBrain v1 core")
    cursor_row = conn.execute(
        "SELECT last_event_rowid, last_event_hash FROM projection_cursor WHERE id=1"
    ).fetchone()
    if full or cursor_row is None:
        _clear_projections(conn)
        cursor = 0
        expected_previous = None
    else:
        cursor = int(cursor_row["last_event_rowid"])
        expected_previous = cursor_row["last_event_hash"]
        anchor = conn.execute(
            "SELECT event_hash FROM brain_events WHERE rowid=?", (cursor,)
        ).fetchone()
        if cursor and (
            anchor is None or str(anchor["event_hash"]) != str(expected_previous)
        ):
            raise RuntimeError("projection cursor anchor does not match the event chain")
    events = conn.execute(
        "SELECT rowid AS rid, * FROM brain_events WHERE rowid > ? ORDER BY rowid",
        (cursor,),
    )
    applied = 0
    last_rowid = cursor
    last_hash = expected_previous
    constraints = _constraint_cache(conn)
    for event in events:
        if event["prev_hash"] != last_hash:
            raise RuntimeError(f"event-chain boundary mismatch at rowid {event['rid']}")
        _verify_one_event(event)
        _apply_event(conn, event, constraints=constraints)
        applied += 1
        last_rowid = int(event["rid"])
        last_hash = str(event["event_hash"])
    cursor_updated_at = (
        str(
            conn.execute("SELECT ts FROM brain_events WHERE rowid=?", (last_rowid,)).fetchone()[0]
        )
        if last_rowid
        else "1970-01-01T00:00:00+00:00"
    )
    conn.execute(
        """
        INSERT INTO projection_cursor(id, last_event_rowid, last_event_hash, updated_at)
        VALUES (1, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          last_event_rowid=excluded.last_event_rowid,
          last_event_hash=excluded.last_event_hash,
          updated_at=excluded.updated_at
        """,
        (last_rowid, last_hash, cursor_updated_at),
    )
    return {
        "applied_events": applied,
        "last_event_rowid": last_rowid,
        "last_event_hash": last_hash,
        "full_rebuild": bool(full or cursor_row is None),
    }


def _clear_projections(conn: sqlite3.Connection) -> None:
    # Child tables first; receipt/audit ledgers are intentionally not projections.
    # Runtime retrievals and their closeout/source-handle references are an
    # append-only audit ledger. Only imported snapshot items are regenerated.
    conn.execute(
        "DELETE FROM retrieval_items WHERE retrieval_use_id IN "
        "(SELECT id FROM retrieval_uses WHERE source_event_id IS NOT NULL)"
    )
    conn.execute("DELETE FROM belief_evidence")
    conn.execute("DELETE FROM object_aliases")
    conn.execute("DELETE FROM evidence_objects")
    conn.execute("DELETE FROM current_beliefs")
    conn.execute("DELETE FROM search_documents")
    conn.execute("DELETE FROM projection_cursor")


def _verify_one_event(event: sqlite3.Row) -> None:
    if event["body_hash"] != sha256_text(str(event["body_json"])):
        raise RuntimeError(f"body hash mismatch at event {event['id']}")
    expected = sha256_text(
        canonical_json(
            {
                "ts": event["ts"],
                "kind": event["kind"],
                "writer": event["writer"],
                "session_id": event["session_id"],
                "body_hash": event["body_hash"],
                "prev_hash": event["prev_hash"],
            }
        )
    )
    if event["event_hash"] != expected:
        raise RuntimeError(f"event hash mismatch at event {event['id']}")


def _constraint_cache(conn: sqlite3.Connection) -> dict[str, list[sqlite3.Row]]:
    constraints: dict[str, list[sqlite3.Row]] = {}
    for event in conn.execute(
        "SELECT * FROM brain_events WHERE kind IN "
        "('correction_recorded','tombstone_recorded') ORDER BY rowid"
    ):
        body = json.loads(event["body_json"])
        target = (
            body.get("target_id")
            if event["kind"] == "correction_recorded"
            else body.get("target")
        )
        if target:
            constraints.setdefault(str(target), []).append(event)
    return constraints


def _apply_event(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    *,
    constraints: dict[str, list[sqlite3.Row]],
) -> None:
    kind = str(event["kind"])
    body = json.loads(event["body_json"])
    if kind == "evidence_recorded":
        _project_recorded_evidence(conn, event, body)
    elif kind == "compilation_decided":
        _project_compilation_decision(conn, event, body)
    elif kind == "correction_recorded":
        _project_correction(conn, event, body)
    elif kind == "tombstone_recorded":
        _project_tombstone(conn, event, body)
    elif kind == "scope_promoted":
        _project_scope_promotion(conn, event, body)
    elif kind == "legacy_evidence_imported":
        _project_legacy_evidence(conn, event, body)
    elif kind == "legacy_knowledge_imported":
        _project_legacy_knowledge(conn, event, body, constraints=constraints)
    elif kind == "retrieval_snapshot_imported":
        _project_retrieval_snapshot(conn, event, body)


def _project_recorded_evidence(
    conn: sqlite3.Connection, event: sqlite3.Row, body: dict[str, Any]
) -> None:
    text = str(body.get("body") or "")
    scope = _scope_dict(body.get("scope"))
    evidence_id = str(body.get("evidence_id") or stable_id("evd", text))
    _upsert_evidence_object(
        conn,
        evidence_id=evidence_id,
        body=text,
        kind=str(body.get("kind") or "observation"),
        content_hash=sha256_text(text),
        source_content_hash=None,
        source_type="event",
        source_runtime=event["writer"],
        source_uri=body.get("artifact_ref"),
        artifact_uri=body.get("artifact_ref"),
        artifact_hash=None,
        verifier_status="unknown",
        occurred_at=event["ts"],
        recorded_at=event["ts"],
        scope=scope,
        metadata={"event_body": body},
        event_id=event["id"],
    )


def _project_compilation_decision(
    conn: sqlite3.Connection, event: sqlite3.Row, body: dict[str, Any]
) -> None:
    if body.get("decision") not in {"approve", "edit"}:
        return
    proposal_id = body.get("proposal_event_id")
    proposal = conn.execute(
        "SELECT body_json FROM brain_events WHERE id=? AND kind='compilation_proposed'",
        (proposal_id,),
    ).fetchone()
    if proposal is None:
        return
    proposed = json.loads(proposal["body_json"])
    belief_id = str(proposed["belief_id"])
    belief_body = str(body.get("edited_body") or proposed.get("body") or "")
    scope = _scope_dict(proposed.get("scope"))
    evidence_ids = [str(item) for item in proposed.get("evidence_ids") or []]
    _write_belief(
        conn,
        belief_id=belief_id,
        body=belief_body,
        belief_type=None,
        attributes={},
        scope=scope,
        confidence=proposed.get("confidence"),
        evidence_ids=evidence_ids,
        status="current",
        serve=True,
        pinned=False,
        approved_event_id=event["id"],
        last_event_id=event["id"],
        compiled_at=event["ts"],
    )
    for evidence_id in evidence_ids:
        _link_belief_evidence(
            conn,
            belief_id,
            evidence_id,
            "supports",
            event["ts"],
            event["id"],
        )


def _project_correction(
    conn: sqlite3.Connection, event: sqlite3.Row, body: dict[str, Any]
) -> None:
    if body.get("target_layer") not in {"knowledge", "belief"}:
        return
    belief_id = resolve_object_id(conn, str(body.get("target_id") or ""))
    row = conn.execute(
        "SELECT * FROM current_beliefs WHERE belief_id=?", (belief_id,)
    ).fetchone()
    if row is None:
        return
    updated = dict(row)
    op = body.get("op")
    if op in {"edit", "reframe"} and body.get("body"):
        updated["body"] = str(body["body"])
    elif op == "pin":
        updated["pinned"] = 1
    elif op == "demote":
        updated["confidence"] = min(float(updated.get("confidence") or 0.5), 0.4)
    elif op in {"mark_wrong", "retract"}:
        updated["status"] = "retracted"
        updated["serve"] = 0
    updated["last_event_id"] = event["id"]
    _replace_belief_row(conn, updated)


def _project_tombstone(
    conn: sqlite3.Connection, event: sqlite3.Row, body: dict[str, Any]
) -> None:
    belief_id = resolve_object_id(conn, str(body.get("target") or ""))
    row = conn.execute(
        "SELECT * FROM current_beliefs WHERE belief_id=?", (belief_id,)
    ).fetchone()
    if row is None:
        return
    updated = dict(row)
    updated["status"] = "tombstoned"
    updated["serve"] = 0
    if body.get("mode") == "shred":
        updated["body"] = "[shredded by tombstone]"
        updated["evidence_ids"] = "[]"
        conn.execute("DELETE FROM belief_evidence WHERE belief_id=?", (belief_id,))
    updated["last_event_id"] = event["id"]
    _replace_belief_row(conn, updated)


def _project_scope_promotion(
    conn: sqlite3.Connection, event: sqlite3.Row, body: dict[str, Any]
) -> None:
    if not body.get("approved_by"):
        return
    belief_id = resolve_object_id(conn, str(body.get("belief_id") or ""))
    scope = _scope_dict(body.get("scope"))
    conn.execute(
        """
        UPDATE current_beliefs SET
          scope_type=?, scope_id=?, visibility=?, egress_policy=?,
          scope_provenance=?, last_event_id=?
        WHERE belief_id=?
        """,
        (*_scope_values(scope), event["id"], belief_id),
    )


def _project_legacy_evidence(
    conn: sqlite3.Connection, event: sqlite3.Row, body: dict[str, Any]
) -> None:
    row = body["row"]
    scope = _scope_dict(body.get("scope"))
    canonical_id = str(body["canonical_evidence_id"])
    _upsert_evidence_object(
        conn,
        evidence_id=canonical_id,
        body=str(row.get("claim") or ""),
        kind=str(row.get("source_type") or "legacy"),
        content_hash=sha256_text(str(row.get("claim") or "")),
        source_content_hash=row.get("content_hash"),
        source_type=row.get("source_type"),
        source_runtime=row.get("source_runtime"),
        source_uri=row.get("source_uri"),
        artifact_uri=row.get("artifact_uri"),
        artifact_hash=row.get("artifact_hash"),
        verifier_status=row.get("verifier_status"),
        occurred_at=row.get("occurred_at"),
        recorded_at=str(row.get("ingested_at") or event["ts"]),
        scope=scope,
        metadata={
            "legacy_row": row,
            "legacy_row_sha256": body.get("legacy_row_sha256"),
        },
        event_id=event["id"],
    )
    source_id = str(body["legacy_evidence_id"])
    if source_id != canonical_id:
        _write_alias(conn, source_id, canonical_id, "evidence", event["id"])


def _project_legacy_knowledge(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    body: dict[str, Any],
    *,
    constraints: dict[str, list[sqlite3.Row]],
) -> None:
    row = body["row"]
    legacy_id = str(body["legacy_knowledge_id"])
    canonical_id = str(body["canonical_belief_id"])
    _write_alias(conn, legacy_id, canonical_id, "belief", event["id"])
    existing = conn.execute(
        "SELECT * FROM current_beliefs WHERE belief_id=?", (canonical_id,)
    ).fetchone()
    imported_status = _legacy_status(str(row.get("status") or "candidate"))
    evidence_links = body.get("evidence_links") or []
    original_evidence = [str(link["evidence_id"]) for link in evidence_links]
    if existing is None:
        scope = _scope_dict(body.get("scope"))
        final_status = imported_status
        belief_body = str(body.get("body") or "")
        confidence = row.get("confidence")
        pinned = bool(row.get("origin") == "human" and row.get("inject"))
        approved_event_id = event["id"]
        compiled_at = str(row.get("updated_at") or row.get("created_at") or event["ts"])
        existing_evidence: list[str] = []
    else:
        scope = {
            "scope_type": existing["scope_type"],
            "scope_id": existing["scope_id"],
            "visibility": existing["visibility"],
            "egress_policy": existing["egress_policy"],
            "provenance": existing["scope_provenance"],
        }
        final_status = _restrictive_status(str(existing["status"]), imported_status)
        belief_body = str(existing["body"])
        confidence = (
            existing["confidence"]
            if existing["confidence"] is not None
            else row.get("confidence")
        )
        pinned = bool(existing["pinned"])
        approved_event_id = existing["approved_event_id"]
        compiled_at = str(existing["last_compiled_at"])
        existing_evidence = _json_list(existing["evidence_ids"])
    evidence_ids = list(dict.fromkeys([*existing_evidence, *original_evidence]))
    attributes = {
        key: value
        for key, value in row.items()
        if key not in {"embedding"}
    }
    if body.get("embedding_sha256"):
        attributes["embedding_sha256"] = body["embedding_sha256"]
    _write_belief(
        conn,
        belief_id=canonical_id,
        body=belief_body,
        belief_type=row.get("type"),
        attributes=attributes,
        scope=scope,
        confidence=confidence,
        evidence_ids=evidence_ids,
        status=final_status,
        serve=_legacy_serve(row, final_status),
        pinned=pinned,
        approved_event_id=approved_event_id,
        last_event_id=event["id"],
        compiled_at=compiled_at,
    )
    for link in evidence_links:
        _link_belief_evidence(
            conn,
            canonical_id,
            str(link["evidence_id"]),
            str(link.get("relation") or "supports"),
            str(link.get("created_at") or event["ts"]),
            event["id"],
        )
    # Legacy corrections precede the import event and target ``know_*``. An
    # unmapped belief did not exist when those events were first folded, so
    # replay only durable constraints after the alias and snapshot now exist.
    _replay_prior_constraints(
        conn,
        legacy_id,
        canonical_id,
        before_event_seq=int(event["event_seq"]),
        constraints=constraints,
    )


def _replay_prior_constraints(
    conn: sqlite3.Connection,
    legacy_id: str,
    canonical_id: str,
    *,
    before_event_seq: int,
    constraints: dict[str, list[sqlite3.Row]],
) -> None:
    prior_events = {
        int(prior["event_seq"]): prior
        for target in (legacy_id, canonical_id)
        for prior in constraints.get(target, [])
        if int(prior["event_seq"]) < before_event_seq
    }
    for prior in (prior_events[key] for key in sorted(prior_events)):
        prior_body = json.loads(prior["body_json"])
        if prior["kind"] == "correction_recorded":
            _project_correction(conn, prior, prior_body)
        else:
            _project_tombstone(conn, prior, prior_body)


def _project_retrieval_snapshot(
    conn: sqlite3.Connection, event: sqlite3.Row, body: dict[str, Any]
) -> None:
    row = body["row"]
    knowledge_id = row.get("knowledge_id")
    canonical_knowledge = resolve_object_id(conn, str(knowledge_id)) if knowledge_id else None
    served_ids = _json_list(row.get("served_ids_json"))
    normalized_ids = [resolve_object_id(conn, item) for item in served_ids]
    if canonical_knowledge and canonical_knowledge not in normalized_ids:
        normalized_ids.insert(0, canonical_knowledge)
    conn.execute(
        """
        INSERT INTO retrieval_uses(
          id, knowledge_id, served_to_runtime, task_ref, affected_decision,
          corrected, outcome, note, query_text, served_ids_json, context_json,
          packet_schema, session_id, feedback_source, feedback_at, served_at,
          source_event_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          knowledge_id=excluded.knowledge_id,
          served_to_runtime=excluded.served_to_runtime,
          task_ref=excluded.task_ref,
          affected_decision=excluded.affected_decision,
          corrected=excluded.corrected,
          outcome=excluded.outcome,
          note=excluded.note,
          query_text=excluded.query_text,
          served_ids_json=excluded.served_ids_json,
          context_json=COALESCE(excluded.context_json, retrieval_uses.context_json),
          packet_schema=COALESCE(excluded.packet_schema, retrieval_uses.packet_schema),
          session_id=excluded.session_id,
          feedback_source=excluded.feedback_source,
          feedback_at=excluded.feedback_at,
          source_event_id=excluded.source_event_id
        """,
        (
            row["id"],
            canonical_knowledge,
            row.get("served_to_runtime"),
            row.get("task_ref"),
            row.get("affected_decision"),
            row.get("corrected"),
            row.get("outcome") or "unknown",
            row.get("note"),
            row.get("query_text"),
            canonical_json(normalized_ids),
            row.get("context_json"),
            row.get("packet_schema"),
            row.get("session_id"),
            row.get("feedback_source"),
            row.get("feedback_at"),
            row.get("served_at") or event["ts"],
            event["id"],
        ),
    )
    conn.execute("DELETE FROM retrieval_items WHERE retrieval_use_id=?", (row["id"],))
    for rank, object_id in enumerate(normalized_ids):
        conn.execute(
            "INSERT INTO retrieval_items VALUES (?, ?, ?, ?, ?)",
            (row["id"], object_id, _object_kind(object_id), rank, None),
        )


def _scope_dict(value: Any) -> dict[str, str]:
    scope = ScopeTag.from_dict(value if isinstance(value, dict) else None)
    return scope.to_dict()


def _scope_values(scope: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(scope["scope_type"]),
        str(scope["scope_id"]),
        str(scope["visibility"]),
        str(scope["egress_policy"]),
        str(scope.get("provenance") or "explicit"),
    )


def _upsert_evidence_object(
    conn: sqlite3.Connection,
    *,
    evidence_id: str,
    body: str,
    kind: str,
    content_hash: str,
    source_content_hash: str | None,
    source_type: str | None,
    source_runtime: str | None,
    source_uri: str | None,
    artifact_uri: str | None,
    artifact_hash: str | None,
    verifier_status: str | None,
    occurred_at: str | None,
    recorded_at: str,
    scope: dict[str, Any],
    metadata: dict[str, Any],
    event_id: str,
) -> None:
    existing = conn.execute(
        "SELECT content_hash FROM evidence_objects WHERE evidence_id=?", (evidence_id,)
    ).fetchone()
    if existing is not None and str(existing["content_hash"]) != content_hash:
        raise RuntimeError(f"evidence id collision escaped migration mapping: {evidence_id}")
    # Projection aliases are conveniences, never semantic authority. A later
    # event may deliberately author the formerly-legacy id as a canonical
    # object; in that case the event-owned object must become directly
    # addressable on both incremental and full replay.
    conn.execute("DELETE FROM object_aliases WHERE alias_id=?", (evidence_id,))
    conn.execute(
        """
        INSERT INTO evidence_objects(
          evidence_id, body, kind, content_hash, source_content_hash,
          source_type, source_runtime,
          source_uri, artifact_uri, artifact_hash, verifier_status, occurred_at,
          recorded_at, scope_type, scope_id, visibility, egress_policy,
          scope_provenance, metadata_json, recorded_event_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(evidence_id) DO UPDATE SET
          source_type=COALESCE(excluded.source_type, evidence_objects.source_type),
          source_runtime=COALESCE(excluded.source_runtime, evidence_objects.source_runtime),
          source_uri=COALESCE(excluded.source_uri, evidence_objects.source_uri),
          artifact_uri=COALESCE(excluded.artifact_uri, evidence_objects.artifact_uri),
          artifact_hash=COALESCE(excluded.artifact_hash, evidence_objects.artifact_hash),
          verifier_status=COALESCE(excluded.verifier_status, evidence_objects.verifier_status),
          source_content_hash=COALESCE(
            excluded.source_content_hash, evidence_objects.source_content_hash
          ),
          metadata_json=excluded.metadata_json
        """,
        (
            evidence_id,
            body,
            kind,
            content_hash,
            source_content_hash,
            source_type,
            source_runtime,
            source_uri,
            artifact_uri,
            artifact_hash,
            verifier_status,
            occurred_at,
            recorded_at,
            *_scope_values(scope),
            canonical_json(metadata),
            event_id,
        ),
    )


def _write_belief(
    conn: sqlite3.Connection,
    *,
    belief_id: str,
    body: str,
    belief_type: str | None,
    attributes: dict[str, Any],
    scope: dict[str, Any],
    confidence: float | None,
    evidence_ids: list[str],
    status: str,
    serve: bool,
    pinned: bool,
    approved_event_id: str | None,
    last_event_id: str,
    compiled_at: str,
) -> None:
    conn.execute("DELETE FROM object_aliases WHERE alias_id=?", (belief_id,))
    conn.execute(
        """
        INSERT INTO current_beliefs(
          belief_id, body, belief_type, attributes_json, scope_type, scope_id,
          visibility, egress_policy, scope_provenance, confidence,
          confidence_band, evidence_ids, status, serve, pinned,
          approved_event_id, last_event_id, last_compiled_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(belief_id) DO UPDATE SET
          body=excluded.body,
          belief_type=excluded.belief_type,
          attributes_json=excluded.attributes_json,
          scope_type=excluded.scope_type,
          scope_id=excluded.scope_id,
          visibility=excluded.visibility,
          egress_policy=excluded.egress_policy,
          scope_provenance=excluded.scope_provenance,
          confidence=excluded.confidence,
          confidence_band=excluded.confidence_band,
          evidence_ids=excluded.evidence_ids,
          status=excluded.status,
          serve=excluded.serve,
          pinned=excluded.pinned,
          approved_event_id=excluded.approved_event_id,
          last_event_id=excluded.last_event_id,
          last_compiled_at=excluded.last_compiled_at
        """,
        (
            belief_id,
            body,
            belief_type,
            canonical_json(attributes),
            *_scope_values(scope),
            confidence,
            _confidence_band(confidence),
            canonical_json(evidence_ids),
            status,
            int(bool(serve)),
            int(bool(pinned)),
            approved_event_id,
            last_event_id,
            compiled_at,
        ),
    )
    if serve and status == "current":
        _replace_search_row(
            conn,
            belief_id,
            f"belief:{belief_type or 'compiled'}",
            str(attributes.get("title") or attributes.get("subject") or belief_id),
            body,
            str(attributes.get("body_uri") or belief_id),
        )
    else:
        conn.execute("DELETE FROM search_documents WHERE doc_id=?", (belief_id,))


def _replace_belief_row(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    _write_belief(
        conn,
        belief_id=str(row["belief_id"]),
        body=str(row["body"]),
        belief_type=row.get("belief_type"),
        attributes=json.loads(row.get("attributes_json") or "{}"),
        scope={
            "scope_type": row["scope_type"],
            "scope_id": row["scope_id"],
            "visibility": row["visibility"],
            "egress_policy": row["egress_policy"],
            "provenance": row.get("scope_provenance") or "explicit",
        },
        confidence=row.get("confidence"),
        evidence_ids=_json_list(row.get("evidence_ids")),
        status=str(row["status"]),
        serve=bool(row.get("serve")),
        pinned=bool(row.get("pinned")),
        approved_event_id=row.get("approved_event_id"),
        last_event_id=str(row["last_event_id"]),
        compiled_at=str(row["last_compiled_at"]),
    )


def _link_belief_evidence(
    conn: sqlite3.Connection,
    belief_id: str,
    evidence_id: str,
    relation: str,
    created_at: str,
    event_id: str,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO belief_evidence(
          belief_id, evidence_id, relation, created_at, source_event_id
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (belief_id, evidence_id, relation, created_at, event_id),
    )


def _write_alias(
    conn: sqlite3.Connection,
    alias_id: str,
    canonical_id: str,
    object_kind: str,
    event_id: str,
) -> None:
    # A legacy snapshot may reuse an id already authored by the event prefix
    # for different content. The migration event records the alternate
    # canonical id, but that mapping must not make the older relational row
    # shadow the event-authoritative object at its own canonical id.
    canonical_object = conn.execute(
        "SELECT 1 FROM evidence_objects WHERE evidence_id=? "
        "UNION ALL SELECT 1 FROM current_beliefs WHERE belief_id=? LIMIT 1",
        (alias_id, alias_id),
    ).fetchone()
    if alias_id == canonical_id or canonical_object is not None:
        return
    conn.execute(
        """
        INSERT INTO object_aliases(alias_id, canonical_id, object_kind, source, source_event_id)
        VALUES (?, ?, ?, 'legacy_v0', ?)
        ON CONFLICT(alias_id) DO UPDATE SET
          canonical_id=excluded.canonical_id,
          object_kind=excluded.object_kind,
          source_event_id=excluded.source_event_id
        """,
        (alias_id, canonical_id, object_kind, event_id),
    )


def _replace_search_row(
    conn: sqlite3.Connection,
    doc_id: str,
    kind: str,
    title: str,
    body: str,
    path: str,
) -> None:
    conn.execute(
        """
        INSERT INTO search_documents(doc_id, kind, title, body, path)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
          kind=excluded.kind,
          title=excluded.title,
          body=excluded.body,
          path=excluded.path
        """,
        (doc_id, kind, title, body, path),
    )


def resolve_object_id(conn: sqlite3.Connection, object_id: str) -> str:
    direct = conn.execute(
        "SELECT 1 FROM evidence_objects WHERE evidence_id=? "
        "UNION ALL SELECT 1 FROM current_beliefs WHERE belief_id=? LIMIT 1",
        (object_id, object_id),
    ).fetchone()
    if direct is not None:
        return object_id
    row = conn.execute(
        "SELECT canonical_id FROM object_aliases WHERE alias_id=?", (object_id,)
    ).fetchone()
    if row is not None:
        return str(row[0])
    if object_id.startswith("know_"):
        canonical = f"legacy:{object_id}"
        exists = conn.execute(
            "SELECT 1 FROM current_beliefs WHERE belief_id=?", (canonical,)
        ).fetchone()
        if exists is not None:
            return canonical
    return object_id


def get_core_v1_belief(conn: sqlite3.Connection, object_id: str) -> dict[str, Any] | None:
    canonical_id = resolve_object_id(conn, object_id)
    row = conn.execute(
        "SELECT * FROM current_beliefs WHERE belief_id=?", (canonical_id,)
    ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["requested_id"] = object_id
    result["canonical_id"] = canonical_id
    result["scope"] = {
        "scope_type": row["scope_type"],
        "scope_id": row["scope_id"],
        "visibility": row["visibility"],
        "egress_policy": row["egress_policy"],
        "provenance": row["scope_provenance"],
    }
    result["evidence_ids"] = _json_list(row["evidence_ids"])
    result["attributes"] = json.loads(row["attributes_json"] or "{}")
    return result


def get_core_v1_evidence(conn: sqlite3.Connection, evidence_id: str) -> dict[str, Any] | None:
    canonical_id = resolve_object_id(conn, evidence_id)
    row = conn.execute(
        "SELECT * FROM evidence_objects WHERE evidence_id=?", (canonical_id,)
    ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["requested_id"] = evidence_id
    result["canonical_id"] = canonical_id
    result["scope"] = {
        "scope_type": row["scope_type"],
        "scope_id": row["scope_id"],
        "visibility": row["visibility"],
        "egress_policy": row["egress_policy"],
        "provenance": row["scope_provenance"],
    }
    result["metadata"] = json.loads(row["metadata_json"] or "{}")
    return result


def search_core_v1(
    conn: sqlite3.Connection,
    query: str,
    *,
    context: ScopeContext | None = None,
    limit: int = 12,
    cross_scope: bool = False,
) -> dict[str, Any]:
    """Return the stable retrieval subset needed by ``ocbrain.context.v1``."""
    context = context or ScopeContext()
    fts = _normalize_fts_query(query)
    if not fts:
        return {"items": [], "excluded": [], "excluded_count": 0}
    compatible = sorted(context.compatible_scope_ids())
    placeholders = ",".join("?" for _ in compatible)
    scope_sql = f"cb.scope_id IN ({placeholders})"
    scope_params: list[Any] = list(compatible)
    if cross_scope:
        scope_sql = (
            f"({scope_sql} OR (cb.scope_type != 'legacy_unscoped' "
            "AND cb.visibility NOT IN ('confidential','secret')))"
        )
    rows = conn.execute(
        f"""
        SELECT cb.*, bm25(search_index) AS lexical_rank
        FROM search_index
        JOIN search_documents sd ON sd.rowid=search_index.rowid
        JOIN current_beliefs cb ON cb.belief_id=sd.doc_id
        WHERE search_index MATCH ? AND cb.serve=1 AND cb.status='current'
          AND {scope_sql}
        ORDER BY lexical_rank, cb.pinned DESC, cb.last_compiled_at DESC, cb.belief_id
        LIMIT ?
        """,  # noqa: S608 - placeholder count is derived from local ScopeContext fields
        (fts, *scope_params, max(limit * 8, 40)),
    )
    items: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    excluded_count = 0
    for rank, row in enumerate(rows):
        scope = ScopeTag(
            str(row["scope_type"]),
            str(row["scope_id"]),
            visibility=str(row["visibility"]),
            egress_policy=str(row["egress_policy"]),
            provenance=str(row["scope_provenance"]),
        )
        weight = scope_match(scope, context, cross_scope=cross_scope)
        if weight == 0:
            excluded_count += 1
            if len(excluded) < limit:
                excluded.append(
                    {
                        "belief_id": row["belief_id"],
                        "scope": scope.to_dict(),
                        "reason": "scope_mismatch",
                    }
                )
            continue
        confidence = float(row["confidence"] if row["confidence"] is not None else 0.5)
        score = weight * confidence / (rank + 1)
        items.append(
            {
                "belief_id": row["belief_id"],
                "body": row["body"],
                "scope": scope.to_dict(),
                "score": round(score, 6),
                "relevance": round(1.0 / (rank + 1), 6),
                "scope_weight": weight,
                "confidence": confidence,
                "confidence_band": row["confidence_band"],
                "evidence_ids": _json_list(row["evidence_ids"]),
                "source": "core_v1",
            }
        )
        if len(items) >= limit:
            break
    return {
        "items": items,
        "excluded": excluded,
        "excluded_count": excluded_count,
    }


def record_core_v1_evidence(
    conn: sqlite3.Connection,
    *,
    body: str,
    kind: str,
    scope: ScopeTag,
    writer: str,
    session_id: str | None = None,
    artifact_ref: str | None = None,
) -> tuple[str, str]:
    evidence_id = stable_id("evd", body, kind, artifact_ref or "", scope.scope_id)
    event_id = append_core_event(
        conn,
        "evidence_recorded",
        {
            "schema_version": CORE_V1_EVENT_SCHEMA,
            "subject": {"kind": "evidence", "id": evidence_id},
            "evidence_id": evidence_id,
            "kind": kind,
            "body": body,
            "artifact_ref": artifact_ref,
            "scope": scope.to_dict(),
        },
        writer=writer,
        session_id=session_id,
        project=True,
    )
    return evidence_id, event_id


def record_core_v1_retrieval(
    conn: sqlite3.Connection,
    *,
    query: str,
    context: dict[str, Any],
    items: Iterable[dict[str, Any]],
    runtime: str | None,
    task_ref: str | None,
    session_id: str | None,
    packet_schema: str = "ocbrain.context.v1",
) -> str:
    rows = list(items)
    served_at = now_iso()
    retrieval_id = stable_id(
        "ret",
        served_at,
        query,
        canonical_json(context),
        canonical_json(
            [item.get("belief_id") or item.get("object_id") or item.get("id") for item in rows]
        ),
    )
    conn.execute(
        """
        INSERT INTO retrieval_uses(
          id, served_to_runtime, task_ref, outcome, query_text, served_ids_json,
          context_json, packet_schema, session_id, served_at
        ) VALUES (?, ?, ?, 'served', ?, ?, ?, ?, ?, ?)
        """,
        (
            retrieval_id,
            runtime,
            task_ref,
            query,
            canonical_json(
                [
                    item.get("belief_id") or item.get("object_id") or item.get("id")
                    for item in rows
                ]
            ),
            canonical_json(context),
            packet_schema,
            session_id,
            served_at,
        ),
    )
    for rank, item in enumerate(rows):
        object_id = resolve_object_id(
            conn,
            str(item.get("belief_id") or item.get("object_id") or item.get("id") or ""),
        )
        object_kind = str(item.get("object_kind") or _object_kind(object_id))
        conn.execute(
            "INSERT INTO retrieval_items VALUES (?, ?, ?, ?, ?)",
            (retrieval_id, object_id, object_kind, rank, item.get("score")),
        )
    return retrieval_id


def _normalize_fts_query(query: str) -> str:
    terms = re.findall(r"[\w-]{2,}", query.lower())
    return " OR ".join(f'"{term}"' for term in terms[:8])


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _legacy_status(status: str) -> str:
    allowed = {"candidate", "current", "superseded", "stale", "archived"}
    return status if status in allowed else "candidate"


_STATUS_RESTRICTIVENESS = {
    "current": 0,
    "candidate": 1,
    "superseded": 2,
    "stale": 3,
    "archived": 4,
    "retracted": 5,
    "tombstoned": 6,
}


def _restrictive_status(existing: str, imported: str) -> str:
    return max((existing, imported), key=lambda item: _STATUS_RESTRICTIVENESS.get(item, 1))


def _legacy_serve(row: dict[str, Any], status: str) -> bool:
    return bool(
        status == "current"
        and not row.get("quarantine_reason")
        and row.get("quality_label") not in {"bad", "excluded"}
    )


def _confidence_band(confidence: float | None) -> str:
    if confidence is None:
        return "unknown"
    if float(confidence) >= 0.75:
        return "strong"
    if float(confidence) >= 0.45:
        return "moderate"
    return "weak"


def _object_kind(object_id: str) -> str:
    if object_id.startswith(("evd_", "legacy:evd_")):
        return "evidence"
    return "belief"


def conservative_legacy_scope(row: dict[str, Any]) -> ScopeTag:
    """Map only explicit legacy context; never infer client/finance scope from text."""
    project = str(row.get("project") or "").strip()
    visibility = "confidential" if row.get("privacy_scope") == "private" else "internal"
    if project and project.lower() not in {"workspace", "default", "none"}:
        return ScopeTag(
            "project",
            f"project:{project}",
            visibility=visibility,
            egress_policy="local_only",
            provenance="legacy_explicit_project",
        )
    body_uri = str(row.get("body_uri") or row.get("artifact_uri") or "").strip()
    if body_uri.startswith("/"):
        path = Path(body_uri).expanduser()
        # A file path is explicit locality, but without a declared request repo it
        # is not authorization to infer a globally visible scope.
        return ScopeTag(
            "legacy_unscoped",
            f"legacy:path:{sha256_text(str(path))[:16]}",
            visibility=visibility,
            egress_policy="local_only",
            provenance="legacy_explicit_path",
        )
    return ScopeTag(
        "legacy_unscoped",
        "legacy:unscoped",
        visibility=visibility,
        egress_policy="local_only",
        provenance="quarantined",
    )


__all__ = [
    "CORE_V1_APPLICATION_ID",
    "CORE_V1_EVENT_SCHEMA",
    "CORE_V1_FTS_TABLES",
    "CORE_V1_SCHEMA_VERSION",
    "CORE_V1_TABLES",
    "CORE_V1_USER_VERSION",
    "LEGACY_IMPORT_KINDS",
    "append_core_event",
    "assert_core_v1_inventory",
    "canonical_json",
    "conservative_legacy_scope",
    "core_v1_table_names",
    "get_core_v1_belief",
    "get_core_v1_evidence",
    "init_core_v1",
    "is_core_v1",
    "project_core_v1",
    "record_core_v1_evidence",
    "record_core_v1_retrieval",
    "rebuild_core_v1_search",
    "resolve_object_id",
    "search_core_v1",
    "set_core_v1_search_triggers",
    "sha256_text",
    "verify_event_chain",
]
