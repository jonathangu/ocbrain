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
import math
import re
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ocbrain.closeout import normalize_task_ref
from ocbrain.history_window import is_body_ref
from ocbrain.hybrid import semantic_neighbors
from ocbrain.ids import stable_id
from ocbrain.provenance import EMPTY_PROVENANCE, Provenance
from ocbrain.scope import (
    LOCAL_MODEL_TARGET,
    ScopeContext,
    ScopeTag,
    scope_affinity,
    scope_match,
)

CORE_V1_APPLICATION_ID = 0x4F434231  # ASCII-ish "OCB1"
CORE_V1_USER_VERSION = 10_000
CORE_V1_SCHEMA_VERSION = "ocbrain.core.v1"

# The one belief_type retrieval treats specially. Full procedure serving is not
# shipped; this constant exists so degraded mode can refuse a procedure the day
# one is minted, rather than the day someone remembers to add the guard.
PROCEDURE_BELIEF_TYPE = "procedure"
# A goal is task state stored on the belief machinery, not a knowledge claim.
# It rides `current_beliefs` because `CORE_V1_TABLES` is a closed allow-list and
# `belief_type` is free text -- but it must never be *retrieved* like knowledge.
# See `ocbrain.briefing` and `_servable_knowledge_sql` below.
GOAL_BELIEF_TYPE = "goal"
CORE_V1_EVENT_SCHEMA = "ocbrain.event.v1"

# Retrieval receipt outcomes, split by who is entitled to write them.
#
# RELEVANCE_OUTCOMES are judgements about *served items* and are the only values
# a caller may file through ``brain.feedback``. SERVED_OUTCOME and
# NO_COVERAGE_OUTCOME are written by the server when it records the receipt: it
# counts the items it just served in the same statement that writes the row, so
# the zero-item case is observed, not reported. A caller-supplied "nothing came
# back" flag would be a second, unverifiable claim about a number the server
# already holds -- and the population it would describe is exactly the one that
# goes unreported when reporting is voluntary.
RELEVANCE_OUTCOMES: tuple[str, ...] = ("helpful", "used", "irrelevant", "ignored", "harmful")
SERVED_OUTCOME = "served"
NO_COVERAGE_OUTCOME = "no_coverage"
# Stamped on a receipt the maintenance command rewrote, so an operator can tell
# a server-derived no_coverage from a reclassified one.
NO_COVERAGE_RECLASSIFY_SOURCE = "maintenance:no_coverage_reclassify"
# What each verdict is worth to ranking. Unchanged from the CASE expression this
# replaced (see ``retrieval_history_by_lineage``); only the language moved.
_FEEDBACK_SIGNAL: dict[str, float] = {
    "helpful": 2.0,
    "used": 1.0,
    "irrelevant": -1.5,
    "ignored": -0.5,
    "harmful": -4.0,
}

HYBRID_RRF_K = 60
# Qwen3's low positive tail is not evidence of topical relevance. Keep a
# moderately permissive floor for candidates that lexical retrieval can
# corroborate, but require a stronger score before a dense-only item may be
# served. These gates favor an honest empty packet over same-scope filler.
MIN_DENSE_COSINE = 0.30
MIN_DENSE_ONLY_COSINE = 0.55
# FTS5 ranks every OR-term hit, even when a long, specific query shares only
# one generic token with an unrelated belief. Suppress a one-term hit only when
# stronger multi-term candidates already cover that term; preserving new query
# term coverage avoids dropping distinctive one-term results.
MIN_LEXICAL_QUERY_TERM_MATCHES = 2
MIN_REDUNDANT_LEXICAL_STRENGTH_RATIO = 0.50
# A lexical hit is held to MIN_DENSE_COSINE too, but only when the dense arm is
# healthy enough to judge it. See ``_retrieval_tuning``.
REQUIRE_DENSE_SUPPORT = True
# Whether `ranking_prior` still multiplies by `0.85 + 0.15 * confidence`.
# Ships True, which is the behaviour every live packet was built with. Turning
# it off is a policy change, not a bug fix: see docs/THRESHOLDS.md.
CONFIDENCE_PRIOR_ENABLED = True

# The shapes a caller uses to name one exact record: a stable object id, a
# SHA-256, or a terminal artifact URI. These live here rather than in
# ``ocbrain.mcp_v1`` because ``search_core_v1`` has to recognise them too --
# ``brain.search`` short-circuited on a locator while ``brain.context`` did not,
# and that asymmetry is what let a nonexistent locator reach dense ranking.
SHA256_TEXT_RE = re.compile(r"^[0-9a-f]{64}$")
STABLE_OBJECT_ID_RE = re.compile(r"^(?:evt|evd|belief|close|ret)_[0-9a-f]{16}$")
TERMINAL_ARTIFACT_URI_RE = re.compile(
    r"^(?:[a-z][a-z0-9+.-]*://\S+|ocbrain-bundle:sha256:[0-9a-f]{64}|"
    r"closeout:close_[0-9a-f]{16})$",
    re.IGNORECASE,
)


def looks_like_exact_locator(query: str) -> bool:
    """True when the query names one exact record rather than a topic.

    Shape only: this says nothing about whether the record exists. That is the
    point -- a well-formed locator that resolves to nothing must return nothing,
    and a caller cannot be told "no such record" by a ranker that always has a
    nearest neighbour to offer.
    """
    text = str(query).strip()
    lowered = text.lower()
    return bool(
        STABLE_OBJECT_ID_RE.fullmatch(lowered)
        or SHA256_TEXT_RE.fullmatch(lowered)
        or TERMINAL_ARTIFACT_URI_RE.fullmatch(text)
    )


_RETRIEVAL_FALLBACK = SimpleNamespace(
    hybrid_rrf_k=HYBRID_RRF_K,
    min_dense_cosine=MIN_DENSE_COSINE,
    min_dense_only_cosine=MIN_DENSE_ONLY_COSINE,
    min_lexical_query_term_matches=MIN_LEXICAL_QUERY_TERM_MATCHES,
    min_redundant_lexical_strength_ratio=MIN_REDUNDANT_LEXICAL_STRENGTH_RATIO,
    require_dense_support=REQUIRE_DENSE_SUPPORT,
    confidence_prior_enabled=CONFIDENCE_PRIOR_ENABLED,
    feedback_weight=0.125,
    feedback_clamp=0.25,
    feedback_prior_observations=3.0,
)


def _retrieval_tuning() -> Any:
    """Resolve the retrieval gates, honoring config-file and env overrides.

    Falls back to the module constants when the config file is unreadable or
    malformed: a broken config must not take retrieval down with it.
    """
    try:
        from ocbrain.config import load_config

        return load_config().retrieval
    except Exception:  # noqa: BLE001 - config problems must not break serving
        return _RETRIEVAL_FALLBACK

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
  -- Empty for a pointer row: the text lives in the file named by
  -- metadata_json's body_ref, and `body_head` holds the recorded excerpt.
  -- See ocbrain.history_window.
  body TEXT NOT NULL,
  body_head TEXT,
  kind TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  source_content_hash TEXT,
  source_type TEXT,
  source_runtime TEXT,
  source_uri TEXT,
  artifact_uri TEXT,
  artifact_hash TEXT,
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
-- Import asks "have I already recorded this file under this kind?" on every
-- harvest; without this it is a full scan of the largest table in the database.
CREATE INDEX IF NOT EXISTS idx_evidence_objects_source
  ON evidence_objects(source_uri, kind, recorded_at DESC);

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
  source_event_id TEXT REFERENCES brain_events(id),
  -- Server-observed caller identity. `session_id` above stays the legacy
  -- model-supplied string; these three are what the process saw for itself.
  -- See ocbrain.provenance for what each one is worth.
  server_connection_id TEXT,
  client_session_hint TEXT,
  client_runtime_key TEXT,
  provenance_json TEXT,
  -- The folded form of `task_ref` above, so a retrieval and the closeout that
  -- links it agree on which task they belong to. NULL on historical rows.
  task_ref_norm TEXT
);
CREATE INDEX IF NOT EXISTS idx_retrieval_uses_outcome_served
  ON retrieval_uses(outcome, served_at);
CREATE INDEX IF NOT EXISTS idx_retrieval_uses_session_hint
  ON retrieval_uses(client_session_hint, served_at);

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
  content_hash TEXT NOT NULL UNIQUE,
  -- Server-observed caller identity, mirrored out of provenance_json so the
  -- closeout-to-transcript join is a column read. `session_id` above stays the
  -- legacy model-supplied string. See ocbrain.provenance.
  server_connection_id TEXT,
  client_session_hint TEXT,
  client_runtime_key TEXT,
  -- Chain pointers. `parent_closeout_id` is the closeout this one continues,
  -- written only when it resolved; `task_ref_norm` is the folded form of
  -- `task_ref` above, which stays verbatim. Both are NULL on every row written
  -- before they existed: the fold happens at write time and history is never
  -- rewritten. See ocbrain.closeout.normalize_task_ref.
  parent_closeout_id TEXT,
  task_ref_norm TEXT
);
CREATE INDEX IF NOT EXISTS idx_task_closeouts_session_hint
  ON task_closeouts(client_session_hint, closed_at);
CREATE INDEX IF NOT EXISTS idx_task_closeouts_chain
  ON task_closeouts(task_ref_norm, closed_at);

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
    version = conn.execute("SELECT value FROM schema_meta WHERE key='core_schema'").fetchone()
    return version is not None and str(version[0]) == CORE_V1_SCHEMA_VERSION


# Columns added to a v1 core after the first release. CORE_V1_SCHEMA uses
# CREATE TABLE IF NOT EXISTS throughout, so an already-initialized core keeps
# its original columns forever unless they are added explicitly. Additive and
# nullable only: no rewrite, no backfill, and an older binary reading a
# migrated core simply ignores them.
_ADDITIVE_CORE_V1_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("evidence_objects", "body_head", "TEXT"),
    ("retrieval_uses", "server_connection_id", "TEXT"),
    ("retrieval_uses", "client_session_hint", "TEXT"),
    ("retrieval_uses", "client_runtime_key", "TEXT"),
    ("retrieval_uses", "provenance_json", "TEXT"),
    ("retrieval_uses", "task_ref_norm", "TEXT"),
    ("task_closeouts", "server_connection_id", "TEXT"),
    ("task_closeouts", "client_session_hint", "TEXT"),
    ("task_closeouts", "client_runtime_key", "TEXT"),
    ("task_closeouts", "parent_closeout_id", "TEXT"),
    ("task_closeouts", "task_ref_norm", "TEXT"),
)

_ADDITIVE_CORE_V1_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_retrieval_uses_session_hint "
    "ON retrieval_uses(client_session_hint, served_at)",
    "CREATE INDEX IF NOT EXISTS idx_task_closeouts_session_hint "
    "ON task_closeouts(client_session_hint, closed_at)",
    "CREATE INDEX IF NOT EXISTS idx_task_closeouts_chain "
    "ON task_closeouts(task_ref_norm, closed_at)",
)


def migrate_core_v1_columns(conn: sqlite3.Connection) -> list[str]:
    """Apply the additive column set to an already-initialized v1 core.

    Idempotent and cheap enough to run on every open: it is one
    ``PRAGMA table_info`` per table when there is nothing to do. Run it there
    rather than behind a separate migrate command, because the MCP server opens
    an existing core without calling :func:`init_core_v1` at all, and a write
    path that referenced a column the running server had never added would fail
    at the first ``brain.context`` after deploy.
    """
    added: list[str] = []
    tables = {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    by_table: dict[str, set[str]] = {}
    for table, column, decl in _ADDITIVE_CORE_V1_COLUMNS:
        if table not in tables:
            continue
        if table not in by_table:
            by_table[table] = {
                str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")
            }
        if column in by_table[table]:
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        by_table[table].add(column)
        added.append(f"{table}.{column}")
    if added:
        for statement in _ADDITIVE_CORE_V1_INDEXES:
            conn.execute(statement)
    return added


def init_core_v1(conn: sqlite3.Connection) -> None:
    """Initialize a fresh v1 core; refuse to layer it over legacy tables."""
    if is_core_v1(conn):
        assert_core_v1_inventory(conn)
        # Migrate columns before replaying the current schema. The schema also
        # declares indexes on additive columns; on an older core those indexes
        # cannot be prepared until the columns exist.
        migrate_core_v1_columns(conn)
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
        conn.execute("UPDATE schema_meta SET value=value WHERE key='__event_writer_reservation__'")
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
        if cursor and (anchor is None or str(anchor["event_hash"]) != str(expected_previous)):
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
        str(conn.execute("SELECT ts FROM brain_events WHERE rowid=?", (last_rowid,)).fetchone()[0])
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
            body.get("target_id") if event["kind"] == "correction_recorded" else body.get("target")
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
        _project_compilation_decision(conn, event, body, constraints=constraints)
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
    # A pointer event carries no text. Its content hash is the hash of the
    # window the pointer names, not of the empty string -- otherwise every
    # pointer row would share one hash and the collision check below would
    # stop meaning anything. Both values ride the event body, so a replay of
    # an old and a new event produces the same rows either way.
    body_ref = body.get("body_ref") if is_body_ref(body.get("body_ref")) else None
    body_head = body.get("body_head")
    _upsert_evidence_object(
        conn,
        evidence_id=evidence_id,
        body=text,
        body_head=str(body_head) if isinstance(body_head, str) else None,
        kind=str(body.get("kind") or "observation"),
        content_hash=(
            str(body_ref["window_sha256"]) if body_ref is not None else sha256_text(text)
        ),
        source_content_hash=(
            str(body_ref["window_input_sha256"]) if body_ref is not None else None
        ),
        source_type="event",
        source_runtime=event["writer"],
        source_uri=body.get("artifact_ref"),
        artifact_uri=body.get("artifact_ref"),
        artifact_hash=None,
        occurred_at=event["ts"],
        recorded_at=event["ts"],
        scope=scope,
        # Keep the event body's metadata (bundle provenance and similar) but drop
        # its `body` text: that text is already in this very row's `body` column
        # and, authoritatively, in brain_events.body_json. Storing it a third time
        # cost ~23% of one real 125MB core. `recorded_event_id` still points at
        # the authoritative event for anything that needs the original.
        metadata={"event_body": _metadata_event_body(body)},
        event_id=event["id"],
    )


def _metadata_event_body(body: dict[str, Any]) -> dict[str, Any]:
    """The event body as projected metadata, without any duplicated text.

    ``body_head`` is dropped for the same reason ``body`` is: it has its own
    column on this row. Two kilobytes per transcript, kept twice, is the whole
    saving of the pointer given back.
    """
    duplicated = {"body", "body_head"}
    slimmed = {key: value for key, value in body.items() if key not in duplicated}
    if "body" in body:
        slimmed["body_omitted"] = "see evidence_objects.body / brain_events.body_json"
    if "body_head" in body:
        slimmed["body_head_omitted"] = "see evidence_objects.body_head"
    return slimmed


def _project_compilation_decision(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    body: dict[str, Any],
    *,
    constraints: dict[str, list[sqlite3.Row]],
) -> None:
    if body.get("decision") not in {"approve", "edit"}:
        return
    proposal_id = body.get("proposal_event_id")
    proposal = conn.execute(
        "SELECT event_seq, body_json FROM brain_events WHERE id=? AND kind='compilation_proposed'",
        (proposal_id,),
    ).fetchone()
    if proposal is None:
        return
    proposed = json.loads(proposal["body_json"])
    belief_id = str(proposed["belief_id"])
    belief_body = str(body.get("edited_body") or proposed.get("body") or "")
    scope = _scope_dict(proposed.get("scope"))
    evidence_ids = [str(item) for item in proposed.get("evidence_ids") or []]
    # A pin is a standing operator decision about a belief, not a property of
    # one revision of its text. Hardcoding ``pinned=False`` here meant every
    # recompilation silently unpinned whatever an operator had pinned, and a
    # scheduled curator recompiles constantly -- which is why one real corpus
    # held exactly one pinned belief. Carry the stored value forward instead.
    # Deterministic under replay: the ``pin`` correction that set it is an
    # earlier event, so a full rebuild reaches this row with the same value.
    existing = conn.execute(
        "SELECT pinned FROM current_beliefs WHERE belief_id=?", (belief_id,)
    ).fetchone()
    _write_belief(
        conn,
        belief_id=belief_id,
        body=belief_body,
        belief_type=proposed.get("belief_type"),
        attributes=dict(proposed.get("attributes") or {}),
        scope=scope,
        confidence=proposed.get("confidence"),
        evidence_ids=evidence_ids,
        status="current",
        serve=True,
        pinned=bool(existing["pinned"]) if existing is not None else False,
        approved_event_id=event["id"],
        last_event_id=event["id"],
        compiled_at=event["ts"],
    )
    # An approved proposal replaces the current support set. The evidence
    # objects and events remain immutable, but obsolete relations must not be
    # served as sources for the new belief revision.
    conn.execute("DELETE FROM belief_evidence WHERE belief_id=?", (belief_id,))
    for evidence_id in evidence_ids:
        _link_belief_evidence(
            conn,
            belief_id,
            evidence_id,
            "supports",
            event["ts"],
            event["id"],
        )
    # A decision can arrive after its proposal has already been corrected or
    # forgotten. Reapply those intervening constraints after materializing the
    # belief so incremental projection and a full rebuild cannot resurrect it.
    # Tombstones and hard restrictive corrections remain durable even when they
    # predate the proposal.
    _replay_compilation_constraints(
        conn,
        belief_id,
        proposal_event_seq=int(proposal["event_seq"]),
        before_event_seq=int(event["event_seq"]),
        constraints=constraints,
    )


def _project_correction(conn: sqlite3.Connection, event: sqlite3.Row, body: dict[str, Any]) -> None:
    if body.get("target_layer") not in {"knowledge", "belief"}:
        return
    belief_id = resolve_object_id(conn, str(body.get("target_id") or ""))
    row = conn.execute("SELECT * FROM current_beliefs WHERE belief_id=?", (belief_id,)).fetchone()
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
    elif op == "supersede":
        # Replacement, not destruction. Every correction an agent has ever
        # issued had to spell this as "retract the wrong belief, then type the
        # right one into a body field nothing indexes and nothing serves". Here
        # the retirement and the forward pointer land in one event: the old
        # belief stops serving, its era closes at this event's timestamp, and
        # ``superseded_by`` names the successor so a reader holding the old id
        # is walked forward instead of refused.
        #
        # Removal from the FTS index is free -- ``_write_belief`` deletes the
        # search row in the same statement that writes a non-serving belief.
        successor_id = str(body.get("successor_id") or "").strip()
        if not successor_id:
            return
        attributes = json.loads(updated.get("attributes_json") or "{}")
        attributes["superseded_by"] = successor_id
        attributes["valid_until"] = str(event["ts"])
        updated["attributes_json"] = canonical_json(attributes)
        updated["status"] = "retracted"
        updated["serve"] = 0
    elif op == "annotate":
        # Metadata only: never status, serve, body, or confidence. This is the
        # writer ``attributes.contradicts`` never had, and the way a mined
        # statistic gets republished -- recompute and replace, never increment,
        # so a replay that folds the same event twice cannot drift. A key whose
        # patch value is ``null`` is deleted rather than stored as null.
        patch = body.get("attributes_patch")
        if not isinstance(patch, dict):
            return
        attributes = json.loads(updated.get("attributes_json") or "{}")
        for key, value in patch.items():
            if value is None:
                attributes.pop(str(key), None)
            else:
                attributes[str(key)] = value
        updated["attributes_json"] = canonical_json(attributes)
    elif op == "restore":
        # The inverse of a soft retraction, and the reason a soft retraction is
        # worth distinguishing from a hard one at all. Refused for anything
        # tombstoned or hard-corrected so those stay terminal under both
        # incremental folding and a full rebuild; a restore event that loses this
        # check is ignored rather than honoured.
        if _restore_blocked(conn, belief_id):
            return
        updated["status"] = "current"
        updated["serve"] = 1
    updated["last_event_id"] = event["id"]
    _replace_belief_row(conn, updated)


def _restore_blocked(conn: sqlite3.Connection, belief_id: str) -> str | None:
    """Return why a belief may not be restored, or ``None`` when it may."""
    tombstone = conn.execute(
        "SELECT 1 FROM brain_events WHERE kind='tombstone_recorded' "
        "AND json_extract(body_json, '$.target')=? LIMIT 1",
        (belief_id,),
    ).fetchone()
    if tombstone is not None:
        return "tombstoned"
    hard = conn.execute(
        "SELECT 1 FROM brain_events WHERE kind='correction_recorded' "
        "AND json_extract(body_json, '$.target_id')=? "
        "AND json_extract(body_json, '$.hard')=1 "
        "AND json_extract(body_json, '$.op') IN ('mark_wrong','retract','demote') LIMIT 1",
        (belief_id,),
    ).fetchone()
    if hard is not None:
        return "hard-corrected"
    # A restore that puts a superseded belief back beside its replacement would
    # serve both halves of a contradiction the ledger has already resolved.
    # Only a *serving* successor blocks: once the replacement is itself retired,
    # restoring the original is a legitimate way back.
    successor = conn.execute(
        "SELECT json_extract(attributes_json, '$.superseded_by') AS successor_id "
        "FROM current_beliefs WHERE belief_id=?",
        (belief_id,),
    ).fetchone()
    successor_id = str((successor["successor_id"] if successor else None) or "").strip()
    if successor_id:
        serving = conn.execute(
            "SELECT 1 FROM current_beliefs "
            "WHERE belief_id=? AND status='current' AND serve=1 LIMIT 1",
            (resolve_object_id(conn, successor_id),),
        ).fetchone()
        if serving is not None:
            return f"superseded by {successor_id}"
    return None


def _project_tombstone(conn: sqlite3.Connection, event: sqlite3.Row, body: dict[str, Any]) -> None:
    belief_id = resolve_object_id(conn, str(body.get("target") or ""))
    row = conn.execute("SELECT * FROM current_beliefs WHERE belief_id=?", (belief_id,)).fetchone()
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
            existing["confidence"] if existing["confidence"] is not None else row.get("confidence")
        )
        pinned = bool(existing["pinned"])
        approved_event_id = existing["approved_event_id"]
        compiled_at = str(existing["last_compiled_at"])
        existing_evidence = _json_list(existing["evidence_ids"])
    evidence_ids = list(dict.fromkeys([*existing_evidence, *original_evidence]))
    attributes = {key: value for key, value in row.items() if key not in {"embedding"}}
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


def _replay_compilation_constraints(
    conn: sqlite3.Connection,
    belief_id: str,
    *,
    proposal_event_seq: int,
    before_event_seq: int,
    constraints: dict[str, list[sqlite3.Row]],
) -> None:
    canonical_id = resolve_object_id(conn, belief_id)
    aliases = {
        str(row["alias_id"])
        for row in conn.execute(
            "SELECT alias_id FROM object_aliases WHERE canonical_id=?", (canonical_id,)
        )
    }
    targets = {belief_id, canonical_id, *aliases}
    prior_events = {
        int(prior["event_seq"]): prior
        for target in targets
        for prior in constraints.get(target, [])
        if int(prior["event_seq"]) < before_event_seq
    }
    for prior in (prior_events[key] for key in sorted(prior_events)):
        prior_body = json.loads(prior["body_json"])
        if prior["kind"] == "tombstone_recorded":
            _project_tombstone(conn, prior, prior_body)
            continue
        is_intervening = int(prior["event_seq"]) > proposal_event_seq
        is_hard_restrictive = bool(prior_body.get("hard")) and prior_body.get("op") in {
            "mark_wrong",
            "retract",
            "demote",
        }
        if is_intervening or is_hard_restrictive:
            _project_correction(conn, prior, prior_body)


def compilation_block_reason(
    conn: sqlite3.Connection,
    belief_id: str,
    *,
    proposal_event_seq: int,
) -> str | None:
    """Return the durable/intervening constraint that blocks a proposal decision."""
    canonical_id = resolve_object_id(conn, belief_id)
    aliases = {
        str(row["alias_id"])
        for row in conn.execute(
            "SELECT alias_id FROM object_aliases WHERE canonical_id=?", (canonical_id,)
        )
    }
    targets = {belief_id, canonical_id, *aliases}
    placeholders = ",".join("?" for _ in targets)
    params = tuple(sorted(targets))
    tombstone = conn.execute(
        "SELECT 1 FROM brain_events WHERE kind='tombstone_recorded' "
        f"AND json_extract(body_json, '$.target') IN ({placeholders}) LIMIT 1",
        params,
    ).fetchone()
    if tombstone is not None:
        return "tombstoned"
    intervening = conn.execute(
        "SELECT 1 FROM brain_events WHERE kind='correction_recorded' "
        "AND json_extract(body_json, '$.target_layer') IN ('knowledge','belief') "
        f"AND json_extract(body_json, '$.target_id') IN ({placeholders}) "
        "AND event_seq > ? LIMIT 1",
        (*params, proposal_event_seq),
    ).fetchone()
    if intervening is not None:
        return "superseded by a post-proposal correction"
    hard_correction = conn.execute(
        "SELECT 1 FROM brain_events WHERE kind='correction_recorded' "
        "AND json_extract(body_json, '$.target_layer') IN ('knowledge','belief') "
        f"AND json_extract(body_json, '$.target_id') IN ({placeholders}) "
        "AND json_extract(body_json, '$.hard')=1 "
        "AND json_extract(body_json, '$.op') IN ('mark_wrong','retract','demote') LIMIT 1",
        params,
    ).fetchone()
    if hard_correction is not None:
        return "hard-corrected"
    row = conn.execute(
        "SELECT status FROM current_beliefs WHERE belief_id=?", (canonical_id,)
    ).fetchone()
    if row is not None and str(row["status"]) in {"retracted", "tombstoned"}:
        return str(row["status"])
    return None


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
    body_head: str | None = None,
    content_hash: str,
    source_content_hash: str | None,
    source_type: str | None,
    source_runtime: str | None,
    source_uri: str | None,
    artifact_uri: str | None,
    artifact_hash: str | None,
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
          evidence_id, body, body_head, kind, content_hash, source_content_hash,
          source_type, source_runtime,
          source_uri, artifact_uri, artifact_hash, occurred_at,
          recorded_at, scope_type, scope_id, visibility, egress_policy,
          scope_provenance, metadata_json, recorded_event_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(evidence_id) DO UPDATE SET
          body_head=COALESCE(excluded.body_head, evidence_objects.body_head),
          source_type=COALESCE(excluded.source_type, evidence_objects.source_type),
          source_runtime=COALESCE(excluded.source_runtime, evidence_objects.source_runtime),
          source_uri=COALESCE(excluded.source_uri, evidence_objects.source_uri),
          artifact_uri=COALESCE(excluded.artifact_uri, evidence_objects.artifact_uri),
          artifact_hash=COALESCE(excluded.artifact_hash, evidence_objects.artifact_hash),
          source_content_hash=COALESCE(
            excluded.source_content_hash, evidence_objects.source_content_hash
          ),
          recorded_at=excluded.recorded_at,
          scope_type=excluded.scope_type,
          scope_id=excluded.scope_id,
          visibility=excluded.visibility,
          egress_policy=excluded.egress_policy,
          scope_provenance=excluded.scope_provenance,
          metadata_json=excluded.metadata_json,
          recorded_event_id=excluded.recorded_event_id
        """,
        (
            evidence_id,
            body,
            body_head,
            kind,
            content_hash,
            source_content_hash,
            source_type,
            source_runtime,
            source_uri,
            artifact_uri,
            artifact_hash,
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


def evidence_body_ref(source: Any) -> dict[str, Any] | None:
    """The body pointer on an evidence row, or ``None`` for an inline body.

    Accepts either a raw ``evidence_objects`` row or the dict
    :func:`get_core_v1_evidence` returns, because both are handed around and a
    caller should not have to know which one it holds.
    """
    if isinstance(source, dict) and "metadata" in source:
        metadata = source.get("metadata")
    else:
        try:
            raw = source["metadata_json"]
        except (KeyError, IndexError, TypeError):
            return None
        try:
            metadata = json.loads(raw or "{}")
        except (TypeError, ValueError):
            return None
    if not isinstance(metadata, dict):
        return None
    event_body = metadata.get("event_body")
    ref = event_body.get("body_ref") if isinstance(event_body, dict) else None
    return ref if is_body_ref(ref) else None


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


def _evidence_support(
    conn: sqlite3.Connection, evidence_ids: Iterable[str]
) -> tuple[int, str | None]:
    """How many evidence objects back a belief, and when the newest was recorded.

    Both are facts about the record that a reader can go and check, which is the
    property the served ``confidence`` number never had.
    """
    ids = [str(value) for value in evidence_ids if str(value)]
    if not ids:
        return 0, None
    placeholders = ",".join("?" for _ in ids)
    row = conn.execute(
        "SELECT COUNT(*), MAX(recorded_at) FROM evidence_objects "
        f"WHERE evidence_id IN ({placeholders})",  # noqa: S608 - placeholders only
        ids,
    ).fetchone()
    if row is None:
        return 0, None
    return int(row[0] or 0), (str(row[1]) if row[1] else None)


def _exact_locator_result(
    conn: sqlite3.Connection,
    query: str,
    *,
    eligible: dict[str, sqlite3.Row],
    context: ScopeContext,
    delivery_target: str,
    cross_scope: bool,
    visibility_counts: dict[str, int],
) -> dict[str, Any]:
    """Resolve a locator-shaped query by equality only. A miss returns nothing.

    The visibility gate is the same one the ranker applies: holding an id is not
    authorisation to read confidential material whose scope the caller did not
    name. A locator naming an evidence object, a closeout, or a retracted belief
    resolves to no *belief* and is therefore empty here -- ``brain.get`` and
    ``brain.search`` are the surfaces that answer those.
    """
    locator = str(query).strip()
    canonical = resolve_object_id(conn, locator)
    row = eligible.get(canonical) or eligible.get(locator)
    items: list[dict[str, Any]] = []
    scope_mix: dict[str, int] = {}
    if row is not None:
        scope = ScopeTag(
            str(row["scope_type"]),
            str(row["scope_id"]),
            visibility=str(row["visibility"]),
            egress_policy=str(row["egress_policy"]),
            provenance=str(row["scope_provenance"]),
        )
        if delivery_target == LOCAL_MODEL_TARGET:
            scope_weight = scope_affinity(scope, context)
        else:
            scope_weight = scope_match(scope, context, cross_scope=cross_scope)
        if scope_weight:
            belief_id = str(row["belief_id"])
            evidence_ids = _json_list(row["evidence_ids"])
            evidence_count, evidence_latest_at = _evidence_support(conn, evidence_ids)
            items.append(
                {
                    "belief_id": belief_id,
                    "body": row["body"],
                    "scope": scope.to_dict(),
                    "score": 1.0,
                    "relevance": 1.0,
                    "scope_weight": scope_weight,
                    "evidence_count": evidence_count,
                    "evidence_latest_at": evidence_latest_at,
                    "evidence_ids": evidence_ids,
                    "source": "core_v1_exact_locator",
                    "ranking": {
                        "lexical_rank": None,
                        "dense_rank": None,
                        "dense_similarity": None,
                        "lexical_component": 0.0,
                        "dense_component": 0.0,
                        "source_quality": 0.0,
                        "recency": 0.0,
                        "ranking_prior": 0.0,
                        "feedback_boost": 0.0,
                        "exact_boost": 1.0,
                    },
                }
            )
            scope_mix[str(scope.scope_id)] = 1
    return {
        "items": items,
        "excluded": [],
        "scope_mix": scope_mix,
        "delivery_excluded_count": visibility_counts["excluded_delivery_count"],
        "exclusion_count_basis": "current_serving_inventory",
        "ranking": {
            "mode": "exact_locator",
            "dense_fallback": None,
            "eligible_count": visibility_counts["eligible_count"],
            "lexical_candidates": 0,
            "dense_candidates": 0,
            # The two facts a caller needs to tell "no such record" from "the
            # ranker had nothing to say": the query was read as a locator, and
            # this is how many records it named.
            "exact_locator": True,
            "exact_locator_matches": len(items),
        },
    }


def search_core_v1(
    conn: sqlite3.Connection,
    query: str,
    *,
    context: ScopeContext | None = None,
    limit: int = 12,
    cross_scope: bool = False,
    delivery_target: str = "local_model",
) -> dict[str, Any]:
    """Return hybrid lexical/dense retrieval for ``ocbrain.context.v1``.

    Dense vectors are a disposable loopback-only sidecar.  If it is absent or
    local inference is unavailable, retrieval remains deterministic and
    lexical.  Lifecycle, visibility, and delivery policy are applied before
    either candidate list is ranked.

    For LOCAL delivery scope ranks rather than selects: the whole serving corpus
    is a candidate and ``scope_affinity`` decides the order. Hosted delivery
    keeps its scope IN-list, because leaving the machine is a different question
    from being relevant.

    ``cross_scope`` is accepted and ignored. It is retained so that live callers
    passing it keep working; there is no longer a narrower mode for it to widen.
    """
    context = context or ScopeContext()
    fts = _normalize_fts_query(query)
    if delivery_target == LOCAL_MODEL_TARGET:
        # No scope prefilter. Scoping the SQL dropped, on average, 3.8 relevant
        # beliefs per query that the caller was entitled to read, and the
        # ranking prior already keeps neighbouring scopes in the tail.
        scope_sql = "1"
        scope_params: list[Any] = []
    else:
        compatible = sorted(context.compatible_scope_ids())
        placeholders = ",".join("?" for _ in compatible)
        scope_sql = f"(cb.scope_type='global' OR cb.scope_id IN ({placeholders}))"
        scope_params = list(compatible)
    delivery_sql = _servable_knowledge_sql(delivery_target)
    visibility_counts = _serving_visibility_counts(
        conn,
        scope_sql=scope_sql,
        scope_params=scope_params,
        delivery_sql=delivery_sql,
    )
    eligible_rows = list(
        conn.execute(
            f"""
            SELECT cb.* FROM current_beliefs cb
            WHERE cb.serve=1 AND cb.status='current' AND {scope_sql} AND {delivery_sql}
            ORDER BY cb.belief_id
            """,  # noqa: S608 - clauses are selected from fixed local constants
            scope_params,
        )
    )
    eligible = {str(row["belief_id"]): row for row in eligible_rows}
    if looks_like_exact_locator(query):
        # An id-shaped query is a lookup, not a topic. Ranking cannot answer it:
        # a locator shares no terms with any body, so the lexical arm returns
        # nothing and the dense arm returns whatever happens to be nearest --
        # which is how the nonexistent, exactly well-formed
        # `belief_ffffffffffffffff` came back as two confident unrelated beliefs
        # at cosine 0.56 and 0.61. Resolve by equality, and let a miss be empty.
        return _exact_locator_result(
            conn,
            query,
            eligible=eligible,
            context=context,
            delivery_target=delivery_target,
            cross_scope=cross_scope,
            visibility_counts=visibility_counts,
        )
    tuning = _retrieval_tuning()
    rrf_k = int(tuning.hybrid_rrf_k)
    min_dense_cosine = float(tuning.min_dense_cosine)
    min_dense_only_cosine = float(tuning.min_dense_only_cosine)
    min_lexical_matches = int(tuning.min_lexical_query_term_matches)
    min_redundant_ratio = float(tuning.min_redundant_lexical_strength_ratio)
    require_dense_support = bool(tuning.require_dense_support)
    confidence_prior_enabled = bool(
        getattr(tuning, "confidence_prior_enabled", CONFIDENCE_PRIOR_ENABLED)
    )
    candidate_limit = max(limit * 10, 120)
    lexical_rows: list[sqlite3.Row] = []
    lexical_uncorroborated = False
    if fts:
        lexical_rows = list(
            conn.execute(
                f"""
                SELECT cb.*, bm25(search_index, 0.25, 5.0, 1.0) AS lexical_score
                FROM search_index
                JOIN search_documents sd ON sd.rowid=search_index.rowid
                JOIN current_beliefs cb ON cb.belief_id=sd.doc_id
                WHERE search_index MATCH ? AND cb.serve=1 AND cb.status='current'
                  AND {scope_sql} AND {delivery_sql}
                ORDER BY lexical_score, cb.pinned DESC, cb.last_compiled_at DESC, cb.belief_id
                LIMIT ?
                """,  # noqa: S608 - clauses are selected from fixed local constants
                (fts, *scope_params, candidate_limit),
            )
        )
        query_terms = {
            term
            for term in re.findall(r"[a-z0-9]{2,}", fts.lower())
            if term != "or"
        }
        # The multi-term bar applies even to a lone lexical row. Gating this on
        # ``len(lexical_rows) > 1`` meant a single belief sharing one generic
        # token with a long, specific query was served unconditionally — the
        # peer comparison below is only needed for the *redundancy* rule, not for
        # deciding whether a row cleared the bar at all.
        if len(query_terms) >= min_lexical_matches and lexical_rows:
            query_lower = query.lower()
            overlaps: list[set[str]] = []
            protected: list[bool] = []
            for row in lexical_rows:
                attributes = json.loads(row["attributes_json"] or "{}")
                document = " ".join(
                    (
                        str(attributes.get("title") or attributes.get("subject") or ""),
                        str(row["body"]),
                    )
                )
                document_terms = set(re.findall(r"[a-z0-9]{2,}", document.lower()))
                overlaps.append(query_terms & document_terms)
                # A query that names a belief outright is an exact lookup, not a
                # topical search. Such a row is never filtered on term overlap.
                protected.append(str(row["belief_id"]).lower() in query_lower)
            corroborated = [
                (row, overlap)
                for row, overlap, keep in zip(lexical_rows, overlaps, protected, strict=True)
                if keep or len(overlap) >= min_lexical_matches
            ]
            if corroborated:
                covered_terms = set().union(*(overlap for _row, overlap in corroborated))
                strongest_corroborated = max(
                    max(0.0, -float(row["lexical_score"] or 0.0))
                    for row, _overlap in corroborated
                )
                lexical_rows = [
                    row
                    for row, overlap, keep in zip(
                        lexical_rows, overlaps, protected, strict=True
                    )
                    if (
                        keep
                        or len(overlap) >= min_lexical_matches
                        or bool(overlap - covered_terms)
                        or strongest_corroborated <= 0.0
                        or max(0.0, -float(row["lexical_score"] or 0.0))
                        >= (strongest_corroborated * min_redundant_ratio)
                    )
                ]
            else:
                # Nothing cleared the multi-term bar. This branch used to fall
                # open and serve every row, so a long specific query sharing one
                # generic token with unrelated beliefs returned those beliefs at
                # lexical rank 1-2. Defer the decision: dropping them is only
                # right if the dense arm is healthy enough to answer instead.
                lexical_uncorroborated = True
    dense_rows, dense_fallback = semantic_neighbors(
        conn,
        query,
        candidate_ids=eligible,
        limit=candidate_limit,
    )
    dense_rows = [
        row for row in dense_rows if float(row.get("similarity") or 0.0) >= min_dense_cosine
    ]
    # A healthy dense arm is one that actually scored the corpus. When the
    # sidecar is missing, stale, or the local embedder is down, ``dense_fallback``
    # is set and every dense score is absent — holding lexical hits to a dense
    # floor then would return nothing at all, so the extra gate stands down.
    dense_arm_healthy = dense_fallback is None
    degraded_excluded_procedures = 0
    if not dense_arm_healthy:
        # A wrong belief is a wrong sentence; a wrong procedure is a wrong
        # afternoon. With the dense arm down the floors below stand down and a
        # single shared token is enough to serve a row, which is a tolerable
        # risk for a claim and not for a multi-step recipe an agent will follow.
        # Gotchas are sentence-shaped and keep serving normally.
        eligible = {
            belief_id: row
            for belief_id, row in eligible.items()
            if str(row["belief_type"] or "") != PROCEDURE_BELIEF_TYPE
        }
        kept_lexical = [
            row
            for row in lexical_rows
            if str(row["belief_type"] or "") != PROCEDURE_BELIEF_TYPE
        ]
        degraded_excluded_procedures = len(lexical_rows) - len(kept_lexical)
        lexical_rows = kept_lexical
    if lexical_uncorroborated and dense_arm_healthy:
        # No lexical row cleared the multi-term bar and dense retrieval is
        # available to answer instead. Without a working dense arm these rows are
        # the only candidates there are, and silence would be worse than a weak
        # lexical match — degraded mode stays as permissive as it was before.
        lexical_rows = []
    lexical_rank = {str(row["belief_id"]): rank for rank, row in enumerate(lexical_rows, 1)}
    dense_rank = {str(row["belief_id"]): rank for rank, row in enumerate(dense_rows, 1)}
    dense_similarity = {str(row["belief_id"]): float(row["similarity"]) for row in dense_rows}
    candidate_ids = set(lexical_rank) | set(dense_rank)
    if not candidate_ids:
        return {
            "items": [],
            "excluded": [],
            "scope_mix": {},
            "delivery_excluded_count": visibility_counts["excluded_delivery_count"],
            "exclusion_count_basis": "current_serving_inventory",
            "ranking": {
                "mode": "lexical" if dense_fallback else "hybrid",
                "dense_fallback": dense_fallback,
                "eligible_count": visibility_counts["eligible_count"],
                "lexical_candidates": 0,
                "dense_candidates": 0,
                "min_dense_cosine": min_dense_cosine,
                "min_dense_only_cosine": min_dense_only_cosine,
                "min_lexical_query_term_matches": min_lexical_matches,
                "min_redundant_lexical_strength_ratio": min_redundant_ratio,
                "require_dense_support": require_dense_support and dense_arm_healthy,
                "confidence_prior_enabled": confidence_prior_enabled,
                "degraded_excluded_procedures": degraded_excluded_procedures,
            },
        }
    feedback = _retrieval_feedback_scores(
        conn,
        candidate_ids,
        weight=float(tuning.feedback_weight),
        clamp=float(tuning.feedback_clamp),
        prior_observations=float(tuning.feedback_prior_observations),
    )
    query_normalized = _dedupe_text(query)
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for belief_id in candidate_ids:
        row = eligible.get(belief_id)
        if row is None:
            continue
        exact_boost = 0.0
        body_normalized = _dedupe_text(str(row["body"]))
        if query_normalized and query_normalized in body_normalized:
            exact_boost += 0.25
        if belief_id.lower() in query.lower():
            exact_boost += 1.0
        similarity = dense_similarity.get(belief_id, 0.0)
        if belief_id in lexical_rank:
            # Lexical hits used to bypass every dense gate, so one shared generic
            # token was enough to serve an unrelated belief — and because the
            # lexical arm scores by unweighted RRF while dense is scaled by
            # similarity, that filler outranked genuinely close dense matches.
            # Hold a lexical hit to the same floor its dense score would need,
            # unless the query names the belief outright or quotes its body.
            if (
                require_dense_support
                and dense_arm_healthy
                and exact_boost == 0.0
                and similarity < min_dense_cosine
            ):
                continue
        elif similarity < min_dense_only_cosine:
            continue
        scope = ScopeTag(
            str(row["scope_type"]),
            str(row["scope_id"]),
            visibility=str(row["visibility"]),
            egress_policy=str(row["egress_policy"]),
            provenance=str(row["scope_provenance"]),
        )
        if delivery_target == LOCAL_MODEL_TARGET:
            scope_weight = scope_affinity(scope, context)
        else:
            scope_weight = scope_match(scope, context, cross_scope=cross_scope)
        if scope_weight == 0:
            # Locally this fires for one reason only: confidential or secret
            # material whose scope the caller did not name. Being in a different
            # scope no longer scores zero, so this is a visibility gate, not a
            # relevance one. Removing it would serve confidential rows.
            continue
        attributes = json.loads(row["attributes_json"] or "{}")
        confidence = float(row["confidence"] if row["confidence"] is not None else 0.65)
        quality = _source_quality(attributes)
        recency = _recency_score(str(row["last_compiled_at"]))
        lexical_component = 0.0
        if belief_id in lexical_rank:
            lexical_component = 1.0 / (rrf_k + lexical_rank[belief_id])
        dense_component = 0.0
        if belief_id in dense_rank:
            dense_component = dense_similarity[belief_id] / (rrf_k + dense_rank[belief_id])
        rrf = lexical_component + dense_component
        feedback_boost = feedback.get(belief_id, 0.0)
        confidence_term = (0.85 + 0.15 * confidence) if confidence_prior_enabled else 1.0
        ranking_prior = (
            scope_weight
            * confidence_term
            * (0.85 + 0.15 * quality)
            * (0.99 + 0.01 * recency)
        )
        score = rrf * ranking_prior * (1.0 + feedback_boost) * (1.0 + exact_boost)
        ranked.append(
            (
                score,
                belief_id,
                {
                    "belief_id": belief_id,
                    "body": row["body"],
                    "scope": scope.to_dict(),
                    "score": round(score, 8),
                    "relevance": round(rrf, 8),
                    "scope_weight": scope_weight,
                    # Evidence support is filled in below, for the served rows
                    # only. It replaces the `confidence` / `confidence_band`
                    # pair this item used to carry; see `_evidence_support`.
                    "evidence_count": 0,
                    "evidence_latest_at": None,
                    "evidence_ids": _json_list(row["evidence_ids"]),
                    "source": "core_v1_hybrid",
                    "ranking": {
                        "lexical_rank": lexical_rank.get(belief_id),
                        "dense_rank": dense_rank.get(belief_id),
                        "dense_similarity": dense_similarity.get(belief_id),
                        "lexical_component": round(lexical_component, 8),
                        "dense_component": round(dense_component, 8),
                        "source_quality": round(quality, 4),
                        "recency": round(recency, 4),
                        "ranking_prior": round(ranking_prior, 6),
                        "feedback_boost": round(feedback_boost, 6),
                        "exact_boost": round(exact_boost, 6),
                    },
                },
            )
        )
    ranked.sort(key=lambda item: (-item[0], item[1]))
    items: list[dict[str, Any]] = []
    seen_content: set[str] = set()
    # Count only rows dropped for duplicate content. Deriving this as
    # ``len(ranked) - len(items)`` conflated dedup with the ``limit`` truncation
    # below, so any ranked-but-unserved row inflated it.
    deduplicated = 0
    for _score, _belief_id, item in ranked:
        content_key = sha256_text(_dedupe_text(str(item["body"])))
        if content_key in seen_content:
            deduplicated += 1
            continue
        seen_content.add(content_key)
        items.append(item)
        if len(items) >= limit:
            break
    # One evidence query for the rows that are actually served, not for every
    # ranked candidate: at limit=12 that is 12 ids, against a candidate list of
    # up to 120.
    for item in items:
        item["evidence_count"], item["evidence_latest_at"] = _evidence_support(
            conn, item["evidence_ids"]
        )
    # What was actually served, by scope. This replaces the old
    # ``excluded_scope_count``, which counted rows the scope filter dropped and
    # so reported 0 forever once the filter was gone. The mix is the signal that
    # would have caught the outage it was meant to describe: a packet that has
    # silently stopped containing the caller's own project shows up here.
    scope_mix: dict[str, int] = {}
    for item in items:
        scope_id = str((item.get("scope") or {}).get("scope_id") or "unknown")
        scope_mix[scope_id] = scope_mix.get(scope_id, 0) + 1
    return {
        "items": items,
        "excluded": [],
        "scope_mix": scope_mix,
        "delivery_excluded_count": visibility_counts["excluded_delivery_count"],
        "exclusion_count_basis": "current_serving_inventory",
        "ranking": {
            "mode": "lexical" if dense_fallback else "hybrid_rrf",
            "dense_fallback": dense_fallback,
            "eligible_count": visibility_counts["eligible_count"],
            "lexical_candidates": len(lexical_rank),
            "dense_candidates": len(dense_rank),
            "deduplicated_candidates": deduplicated,
            "rrf_k": rrf_k,
            "min_dense_cosine": min_dense_cosine,
            "min_dense_only_cosine": min_dense_only_cosine,
            "min_lexical_query_term_matches": min_lexical_matches,
            "min_redundant_lexical_strength_ratio": min_redundant_ratio,
            "require_dense_support": require_dense_support and dense_arm_healthy,
            "confidence_prior_enabled": confidence_prior_enabled,
            # Procedures dropped because the dense arm was unavailable. Zero in
            # healthy mode; a non-zero value says the packet is deliberately
            # thinner than the corpus could support.
            "degraded_excluded_procedures": degraded_excluded_procedures,
        },
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
    body_ref: dict[str, Any] | None = None,
    body_head: str | None = None,
    identity_body: str | None = None,
) -> tuple[str, str]:
    """Append an ``evidence_recorded`` event, inline or as a pointer.

    A pointer passes ``body=""`` with a ``body_ref`` naming the file the text
    can be rebuilt from and a ``body_head`` excerpt of it. ``identity_body`` is
    then the text the evidence id is derived from, so the id stays exactly what
    it would have been with the text stored inline: the same window is the same
    evidence, a re-windowed transcript is new evidence, and no existing id
    moves. Everything the projection needs rides the event body, so replaying
    an old inline event and a new pointer event both land the right row.
    """
    evidence_id = stable_id(
        "evd",
        body if identity_body is None else identity_body,
        kind,
        artifact_ref or "",
        scope.scope_id,
    )
    event_body: dict[str, Any] = {
        "schema_version": CORE_V1_EVENT_SCHEMA,
        "subject": {"kind": "evidence", "id": evidence_id},
        "evidence_id": evidence_id,
        "kind": kind,
        "body": body,
        "artifact_ref": artifact_ref,
        "scope": scope.to_dict(),
    }
    if body_ref is not None:
        event_body["body_ref"] = body_ref
        event_body["body_head"] = body_head or ""
    event_id = append_core_event(
        conn,
        "evidence_recorded",
        event_body,
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
    provenance: Provenance | None = None,
) -> str:
    """Append the read receipt for one served packet.

    ``runtime`` and ``session_id`` are recorded verbatim as the model sent them.
    They used to be run through a ``canonical_runtime`` collapser that guessed
    which client a free-text string meant; that guess now lives read-side in
    ``scripts/procmine`` where it belongs, because ``provenance`` carries what
    the server actually observed and there is nothing left to guess about.

    ``provenance`` is deliberately absent from the ``stable_id`` inputs and from
    ``context_json``: two identical reads must stay the same read regardless of
    which connection served them.

    A packet holding no items is recorded as ``no_coverage`` rather than
    ``served``. The distinction is derived here, from the item list this call
    was handed, because this is the only place that both knows the count and
    writes the row; downstream, "the brain had nothing" and "the brain served
    junk" are otherwise indistinguishable in the outcome column.
    """
    rows = list(items)
    served_at = now_iso()
    prov = provenance or EMPTY_PROVENANCE
    prov_payload = prov.to_dict()
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
          context_json, packet_schema, session_id, served_at,
          server_connection_id, client_session_hint, client_runtime_key,
          provenance_json, task_ref_norm
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            retrieval_id,
            runtime,
            task_ref,
            SERVED_OUTCOME if rows else NO_COVERAGE_OUTCOME,
            query,
            canonical_json(
                [item.get("belief_id") or item.get("object_id") or item.get("id") for item in rows]
            ),
            canonical_json(context),
            packet_schema,
            session_id,
            served_at,
            prov.server_connection_id,
            prov.client_session_hint,
            prov.client_runtime_key,
            canonical_json(prov_payload) if prov_payload else None,
            # Same fold the closeout writes, so a read and the receipt that
            # links it agree on which task they belong to without matching on
            # free text.
            normalize_task_ref(task_ref) if task_ref else None,
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


def retrieval_served_item_count(conn: sqlite3.Connection, retrieval_use_id: str) -> int | None:
    """How many items one recorded retrieval served, or ``None`` if it is unknown.

    Reads both halves of the receipt and takes the larger: ``served_ids_json``
    is written in the same statement as the row itself, ``retrieval_items`` is
    the normalized copy. They agree on all 2,048 rows of the corpus snapshot
    frozen at 2026-08-28T19:28:58Z (0 disagreements in either direction), and
    reading both means neither a core whose item rows were never backfilled nor
    one whose receipt column is empty can be mistaken for a zero-item read.
    """
    row = conn.execute(
        "SELECT served_ids_json FROM retrieval_uses WHERE id=?", (retrieval_use_id,)
    ).fetchone()
    if row is None:
        return None
    items = int(
        conn.execute(
            "SELECT COUNT(*) FROM retrieval_items WHERE retrieval_use_id=?",
            (retrieval_use_id,),
        ).fetchone()[0]
    )
    return max(items, len(_json_list(row["served_ids_json"])))


def reclassify_no_coverage_receipts(
    conn: sqlite3.Connection, *, apply: bool = False
) -> dict[str, Any]:
    """Move relevance verdicts filed on zero-item retrievals to ``no_coverage``.

    The server now refuses these at the door, but the rows already written under
    the instruction-only rule stay in the corpus, where an ``irrelevant`` filed
    on an empty packet reads as "the brain served junk" forever. Whether to
    rewrite live history is an operator's call, not a server's, so this is an
    explicit command that reports by default and writes only under ``apply``.

    The prior verdict is appended to the row's note rather than dropped, so a
    reclassification stays legible -- and reversible by hand -- afterwards.
    """
    rows = list(
        conn.execute(
            f"""
            SELECT ru.id AS id, ru.outcome AS outcome, ru.note AS note
            FROM retrieval_uses ru
            WHERE ru.outcome IN ({",".join("?" for _ in RELEVANCE_OUTCOMES)})
              AND NOT EXISTS (
                SELECT 1 FROM retrieval_items ri WHERE ri.retrieval_use_id = ru.id
              )
              AND COALESCE(ru.served_ids_json, '[]') IN ('[]', '', 'null')
            ORDER BY ru.served_at, ru.id
            """,  # noqa: S608 - placeholder count derives only from the outcome vocabulary
            tuple(RELEVANCE_OUTCOMES),
        )
    )
    by_outcome: dict[str, int] = {}
    for row in rows:
        outcome = str(row["outcome"])
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
    stamp = now_iso()
    if apply:
        for row in rows:
            marker = f"[reclassified from {row['outcome']}: retrieval served zero items]"
            note = str(row["note"] or "").strip()
            conn.execute(
                "UPDATE retrieval_uses SET outcome=?, note=?, feedback_source=?, "
                "feedback_at=? WHERE id=?",
                (
                    NO_COVERAGE_OUTCOME,
                    f"{note} {marker}".strip(),
                    NO_COVERAGE_RECLASSIFY_SOURCE,
                    stamp,
                    str(row["id"]),
                ),
            )
    return {
        "candidates": len(rows),
        "by_outcome": dict(sorted(by_outcome.items())),
        "applied": len(rows) if apply else 0,
        "dry_run": not apply,
        "sample": [str(row["id"]) for row in rows[:12]],
    }


def _normalize_fts_query(query: str) -> str:
    terms = re.findall(r"[\w-]{2,}", query.lower())
    stopwords = {
        "about",
        "an",
        "after",
        "again",
        "also",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "have",
        "how",
        "if",
        "in",
        "into",
        "just",
        "make",
        "of",
        "on",
        "or",
        "our",
        "should",
        "that",
        "the",
        "their",
        "them",
        "then",
        "this",
        "to",
        "very",
        "via",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
        "you",
        "your",
    }
    meaningful = list(dict.fromkeys(term for term in terms if term not in stopwords))
    return " OR ".join(f'"{term}"' for term in meaningful[:24])


def _delivery_sql(target: str) -> str:
    if target == "hosted_model":
        return (
            "cb.egress_policy='hosted_ok' AND cb.scope_type!='client' "
            "AND cb.visibility NOT IN ('confidential','secret')"
        )
    if target == "local_model":
        return "cb.egress_policy!='prohibited'"
    raise ValueError(f"unsupported delivery target: {target}")


def _servable_knowledge_sql(target: str) -> str:
    """The delivery gate, plus the belief types that are not knowledge at all.

    Goals are the only such type today. A goal is retrieved by scope and status
    and by nothing else -- deterministic selection beats similarity judgement by
    +10.8pp, widening to +21pp at long context (arXiv 2606.01435) -- so letting
    one into the hybrid candidate pool would give it a *second*, ranked door,
    and the two doors would disagree. Worse, a goal is a restatement of an
    objective in the caller's own words, which is precisely the shape that
    scores well against a query about that objective and displaces the knowledge
    the caller actually asked for.

    One predicate, one call site: ``retrieve_core_v1`` builds this once and
    passes it to the lexical arm, the eligible pool, and the inventory counts,
    so the three can never drift apart.
    """
    return f"({_delivery_sql(target)}) AND COALESCE(cb.belief_type, '') <> '{GOAL_BELIEF_TYPE}'"


def _serving_visibility_counts(
    conn: sqlite3.Connection,
    *,
    scope_sql: str,
    scope_params: list[Any],
    delivery_sql: str,
) -> dict[str, int]:
    """Partition current serving inventory without exposing excluded objects.

    These counts describe the delivery gate before query ranking. They are
    intentionally query-independent so an empty or unmatched query still reports
    whether the supplied context can see any serving inventory.

    There is no ``excluded_scope_count`` here any more. Local delivery has no
    scope prefilter to exclude anything, so the number was a constant zero
    dressed up as a measurement; ``scope_mix`` reports what was actually served
    instead.
    """
    row = conn.execute(
        f"""
        WITH classified AS (
          SELECT
            CASE WHEN {scope_sql} THEN 1 ELSE 0 END AS scope_allowed,
            CASE WHEN {delivery_sql} THEN 1 ELSE 0 END AS delivery_allowed
          FROM current_beliefs cb
          WHERE cb.serve=1 AND cb.status='current'
        )
        SELECT
          COALESCE(SUM(CASE WHEN scope_allowed=1 AND delivery_allowed=0 THEN 1 ELSE 0 END), 0)
            AS excluded_delivery_count,
          COALESCE(SUM(CASE WHEN scope_allowed=1 AND delivery_allowed=1 THEN 1 ELSE 0 END), 0)
            AS eligible_count
        FROM classified
        """,  # noqa: S608 - clauses are selected from fixed local expressions
        scope_params,
    ).fetchone()
    return {
        "excluded_delivery_count": int(row["excluded_delivery_count"]),
        "eligible_count": int(row["eligible_count"]),
    }


def _source_quality(attributes: dict[str, Any]) -> float:
    raw = attributes.get("source_quality", attributes.get("quality_score", 0.75))
    try:
        return min(max(float(raw), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.75


def _recency_score(value: str) -> float:
    try:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=UTC)
        days = max((datetime.now(UTC) - observed).total_seconds() / 86_400.0, 0.0)
    except (TypeError, ValueError):
        return 0.0
    return math.exp(-days / 365.0)


def retrieval_history_by_lineage(
    conn: sqlite3.Connection, belief_ids: set[str]
) -> dict[str, dict[str, float]]:
    """Judged retrieval history for each belief, counting its lineage's too.

    Returns ``{belief_id: {"n", "signal", "inherited_n"}}`` over every judged
    retrieval that served the belief, one of its aliases, or any belief it
    replaced -- transitively, so a claim recompiled five times carries all five
    generations' record instead of the fortnight since its latest id was minted.

    The lineage is *derived* from the era pointers the ledger already projects
    (``attributes.superseded_by`` on the predecessor), never copied forward at
    supersede time. Two consequences matter. A chain accumulates by
    construction: generation three walks back through generation two to
    generation one without anything having been summed at each hop, so a copy
    that was taken once and then went stale is not a state this can reach. And
    nothing can be counted twice: the walk yields a *set* of ids, and verdicts
    are folded per ``(belief, retrieval_use)`` pair, so a single retrieval that
    served both a belief and one of its own ancestors still contributes one
    verdict.

    Alias-recorded history is attributed to the belief for the same reason:
    retrieval rows written before an alias was collapsed are stored under the
    old id, and without ``object_aliases`` that history is silently dropped.
    """
    if not belief_ids:
        return {}
    lineage = _belief_lineage_members(conn, belief_ids)
    members = sorted({member for members in lineage.values() for member in members})
    # Verdicts are fetched by a flat id list rather than by joining the lineage
    # walk to `retrieval_items` inside one statement: the flat form uses
    # `idx_retrieval_items_object`, and the joined form does not. On the live
    # core that difference is 0.9 ms against 88 ms per ranked retrieval.
    verdicts: dict[str, list[tuple[str, str]]] = {}
    placeholders = ",".join("?" for _ in members)
    outcomes = ",".join("?" for _ in RELEVANCE_OUTCOMES)
    for row in conn.execute(
        f"""
        SELECT ri.object_id AS object_id, ru.id AS use_id, ru.outcome AS outcome
        FROM retrieval_items ri
        JOIN retrieval_uses ru ON ru.id = ri.retrieval_use_id
        WHERE ri.object_id IN ({placeholders})
          AND ru.outcome IN ({outcomes})
        """,  # noqa: S608 - placeholders derive only from ids and the outcome vocabulary
        (*members, *RELEVANCE_OUTCOMES),
    ):
        verdicts.setdefault(str(row["object_id"]), []).append(
            (str(row["use_id"]), str(row["outcome"]))
        )
    history: dict[str, dict[str, float]] = {}
    for belief_id, members_by_origin in lineage.items():
        # Folded per retrieval, so a single retrieval that served two members of
        # one lineage counts as one verdict. A verdict is inherited only when
        # every member that carried it was an ancestor, which does not depend on
        # the order the members are walked in.
        folded: dict[str, tuple[str, bool]] = {}
        for member, inherited in members_by_origin.items():
            for use_id, outcome in verdicts.get(member, ()):
                previous = folded.get(use_id)
                folded[use_id] = (
                    outcome,
                    inherited and (previous is None or previous[1]),
                )
        if not folded:
            continue
        history[belief_id] = {
            "n": len(folded),
            "signal": sum(_FEEDBACK_SIGNAL[outcome] for outcome, _ in folded.values()),
            "inherited_n": sum(1 for _, inherited in folded.values() if inherited),
        }
    return history


def _belief_lineage_members(
    conn: sqlite3.Connection, belief_ids: set[str]
) -> dict[str, dict[str, bool]]:
    """Every id each belief's history may live under: ``{belief: {id: inherited}}``.

    ``inherited`` is False for the belief's own id and its aliases, True for a
    belief it replaced. The walk reads ``attributes.superseded_by``, which the
    projector stamps on the *predecessor* of every supersession -- including the
    curator's key-collision cascade, whose successor is minted through ordinary
    compilation and carries no ``supersedes`` of its own. Walking the other
    pointer would see 64 of the 274 era closures in the 2026-08-28T19:28:58Z
    snapshot.

    The era pointers are read once per call and the walk runs in Python.
    Expressed as a recursive CTE instead, each step re-scans ``current_beliefs``
    evaluating ``json_extract`` per row, because no index covers that
    expression: 85 ms per ranked retrieval against 1.5 ms here, on a hot path
    that runs on every ``brain.context``.

    Depth is deliberately unbounded, unlike ``mcp_v1.MAX_RESOLUTION_HOPS``, which
    caps the *forward* walk over this same pointer at ten. That bound exists
    because resolution reads a belief per hop and only needs one answer, so a
    long chain there is a corpus problem. This walk needs every generation, pays
    one query for the whole pointer map, and is bounded by the edges in it; the
    deepest serving lineage in the 2026-08-28T19:28:58Z snapshot measured 12
    generations, so a ten-hop cap here would silently drop two of them. The
    seen-set is what terminates a cycle, and it is checked before descending.
    """
    predecessors: dict[str, list[str]] = {}
    for row in conn.execute(
        "SELECT belief_id, json_extract(attributes_json, '$.superseded_by') AS successor_id "
        "FROM current_beliefs WHERE attributes_json LIKE '%superseded_by%'"
    ):
        successor = str(row["successor_id"] or "").strip()
        if successor:
            predecessors.setdefault(successor, []).append(str(row["belief_id"]))
    lineage: dict[str, dict[str, bool]] = {}
    for belief_id in belief_ids:
        members = {belief_id: False}
        frontier = [belief_id]
        while frontier:
            # A DAG, not a chain: several beliefs can be retired into one
            # survivor. Membership is checked before descending, so a pointer
            # cycle terminates and no id is added twice.
            for ancestor in predecessors.get(frontier.pop(), ()):
                if ancestor not in members:
                    members[ancestor] = True
                    frontier.append(ancestor)
        lineage[belief_id] = members
    collected = sorted({member for members in lineage.values() for member in members})
    placeholders = ",".join("?" for _ in collected)
    aliases: dict[str, list[str]] = {}
    for row in conn.execute(
        f"SELECT alias_id, canonical_id FROM object_aliases WHERE canonical_id IN ({placeholders})",  # noqa: S608 - placeholder count derives only from collected ids
        tuple(collected),
    ):
        aliases.setdefault(str(row["canonical_id"]), []).append(str(row["alias_id"]))
    if aliases:
        for members in lineage.values():
            for member, inherited in list(members.items()):
                for alias in aliases.get(member, ()):
                    members.setdefault(alias, inherited)
    return lineage


def _retrieval_feedback_scores(
    conn: sqlite3.Connection,
    belief_ids: set[str],
    *,
    weight: float = 0.125,
    clamp: float = 0.25,
    prior_observations: float = 3.0,
) -> dict[str, float]:
    """Score each belief by how its retrievals were judged.

    Applied multiplicatively as ``1 + boost`` at ranking time. The boost is
    damped by ``n / (n + prior_observations)`` so a single verdict cannot swing a
    belief's position; a belief needs a consistent record before it moves far.

    History comes from :func:`retrieval_history_by_lineage`, so a recompiled
    belief keeps the record its predecessors earned instead of restarting at
    zero observations every curator pass.
    """
    result: dict[str, float] = {}
    for belief_id, history in retrieval_history_by_lineage(conn, belief_ids).items():
        count = int(history["n"])
        average = history["signal"] / count
        confidence = count / (count + prior_observations)
        boost = average * weight * confidence
        result[belief_id] = min(max(boost, -clamp), clamp)
    return result


def _dedupe_text(value: str) -> str:
    return " ".join(re.findall(r"[\w-]+", value.lower()))


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
    # The full legacy snapshot remains immutable evidence/archive, never a
    # retrieval corpus.  Serving requires a native v1 proposal plus explicit
    # gate decision so catalog paths, plugin caches, transcript pointers, and
    # package-store chatter cannot leak back into current memory on rebuild.
    del row, status
    return False


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
    "GOAL_BELIEF_TYPE",
    "LEGACY_IMPORT_KINDS",
    "NO_COVERAGE_OUTCOME",
    "PROCEDURE_BELIEF_TYPE",
    "RELEVANCE_OUTCOMES",
    "SERVED_OUTCOME",
    "append_core_event",
    "assert_core_v1_inventory",
    "canonical_json",
    "compilation_block_reason",
    "conservative_legacy_scope",
    "core_v1_table_names",
    "evidence_body_ref",
    "get_core_v1_belief",
    "get_core_v1_evidence",
    "init_core_v1",
    "is_core_v1",
    "looks_like_exact_locator",
    "migrate_core_v1_columns",
    "project_core_v1",
    "rebuild_core_v1_search",
    "reclassify_no_coverage_receipts",
    "record_core_v1_evidence",
    "record_core_v1_retrieval",
    "resolve_object_id",
    "retrieval_history_by_lineage",
    "retrieval_served_item_count",
    "search_core_v1",
    "set_core_v1_search_triggers",
    "sha256_text",
    "verify_event_chain",
]
