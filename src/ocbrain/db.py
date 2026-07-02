from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ocbrain.ids import stable_id

DEFAULT_DB_PATH = Path(os.environ.get("OCBRAIN_DB", "~/.ocbrain/ocbrain.sqlite")).expanduser()
PUBLIC_SCOPES = ("workspace", "project", "public")
SCOPE_RANK = {"private": 0, "workspace": 1, "project": 2, "public": 3}


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS evidence (
  id TEXT PRIMARY KEY,
  source_type TEXT NOT NULL,
  source_runtime TEXT,
  source_uri TEXT,
  content_hash TEXT NOT NULL,
  claim TEXT NOT NULL,
  artifact_uri TEXT,
  artifact_hash TEXT,
  verifier_status TEXT CHECK (
    verifier_status IN ('unknown','passed','failed','not_required')
  ) DEFAULT 'unknown',
  loop_tags TEXT,
  project TEXT,
  privacy_scope TEXT CHECK (
    privacy_scope IN ('private','workspace','project','public')
  ) DEFAULT 'workspace',
  occurred_at TEXT,
  ingested_at TEXT NOT NULL,
  UNIQUE(source_uri, content_hash)
);

CREATE TABLE IF NOT EXISTS knowledge (
  id TEXT PRIMARY KEY,
  type TEXT CHECK (type IN ('value','doc','capability')) NOT NULL,
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
  status TEXT CHECK (
    status IN ('candidate','current','superseded','stale','archived')
  ) DEFAULT 'candidate',
  superseded_by TEXT REFERENCES knowledge(id),
  invalidation_reason TEXT,
  gate TEXT CHECK (gate IN ('auto','human')) NOT NULL,
  prescriptive INTEGER DEFAULT 0,
  inject INTEGER DEFAULT 0,
  risk TEXT CHECK (risk IN ('low','medium','high','critical')) DEFAULT 'low',
  confidence REAL,
  content_hash TEXT,
  loop_tags TEXT,
  project TEXT,
  privacy_scope TEXT CHECK (
    privacy_scope IN ('private','workspace','project','public')
  ) DEFAULT 'workspace',
  approved_by TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_evidence (
  knowledge_id TEXT NOT NULL REFERENCES knowledge(id),
  evidence_id TEXT NOT NULL REFERENCES evidence(id),
  relation TEXT CHECK (
    relation IN ('supports','contradicts','derived_from','supersedes')
  ),
  created_at TEXT NOT NULL,
  PRIMARY KEY (knowledge_id, evidence_id, relation)
);

CREATE TABLE IF NOT EXISTS retrieval_uses (
  id TEXT PRIMARY KEY,
  knowledge_id TEXT REFERENCES knowledge(id),
  served_to_runtime TEXT,
  task_ref TEXT,
  affected_decision INTEGER,
  corrected INTEGER,
  outcome TEXT CHECK (
    outcome IN (
      'improved','failed','neutral','unknown',
      'served','helpful','used','irrelevant','ignored','harmful'
    )
  ) DEFAULT 'unknown',
  note TEXT,
  served_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS loop_liveness (
  loop_id TEXT NOT NULL,
  run_id TEXT,
  last_heartbeat_at TEXT,
  last_ledger_write_at TEXT,
  expected_interval_seconds INTEGER,
  deadman_due_at TEXT,
  PRIMARY KEY (loop_id, run_id)
);

CREATE TABLE IF NOT EXISTS family_scores (
  loop_id TEXT NOT NULL,
  family TEXT NOT NULL,
  attempts INTEGER,
  kept INTEGER,
  reverted INTEGER,
  approach_failures INTEGER,
  verifier_pass_rate REAL,
  mean_primary_delta REAL,
  recency TEXT,
  state TEXT CHECK (
    state IN ('promising','exhausted','blocked','risky','stale','untried')
  ),
  refreshed_at TEXT,
  PRIMARY KEY (loop_id, family)
);

CREATE TABLE IF NOT EXISTS brain_events (
  id TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  kind TEXT CHECK (
    kind IN (
      'evidence_recorded',
      'compilation_proposed',
      'compilation_decided',
      'correction_recorded',
      'tombstone_recorded',
      'scope_promoted'
    )
  ) NOT NULL,
  writer TEXT NOT NULL,
  session_id TEXT,
  body_json TEXT NOT NULL,
  body_hash TEXT NOT NULL,
  prev_hash TEXT,
  event_hash TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_brain_events_kind_ts ON brain_events(kind, ts);

CREATE TABLE IF NOT EXISTS current_beliefs (
  belief_id TEXT PRIMARY KEY,
  body TEXT NOT NULL,
  scope_type TEXT NOT NULL,
  scope_id TEXT NOT NULL,
  visibility TEXT NOT NULL,
  egress_policy TEXT NOT NULL,
  confidence REAL,
  confidence_band TEXT,
  evidence_ids TEXT NOT NULL,
  status TEXT CHECK (
    status IN ('current','superseded','retracted','tombstoned')
  ) NOT NULL DEFAULT 'current',
  pinned INTEGER NOT NULL DEFAULT 0,
  approved_event_id TEXT NOT NULL,
  last_event_id TEXT NOT NULL,
  last_compiled_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_current_beliefs_scope
  ON current_beliefs(scope_type, scope_id, status);

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

CREATE VIEW IF NOT EXISTS memory AS
 SELECT id, type, subject, predicate, value_numeric, value_text, value_bool,
        title, body_uri, project, privacy_scope
 FROM knowledge
 WHERE status='current' AND inject=1;

CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
  doc_id UNINDEXED,
  kind,
  title,
  body,
  path UNINDEXED
);
"""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def connect(path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


# Legacy ocbrain tables burned down at startup, ordered so referencing tables are
# dropped before the tables they reference. Each name maps to a distinctive marker
# column plus the full legacy column set; a table is only dropped when it provably
# matches the legacy ocbrain shape (marker column present, or every column drawn
# from the legacy definition), so init can never destroy an unrelated user table
# that merely happens to be named 'events' etc.
_LEGACY_TABLES: dict[str, tuple[str, frozenset[str]]] = {
    "artifact_links": (
        "candidate_id",
        frozenset(
            {"id", "candidate_id", "surface", "uri", "line_start", "line_end",
             "applied_at", "applied_by"}
        ),
    ),
    "invalidations": (
        "old_candidate_id",
        frozenset({"id", "old_candidate_id", "new_candidate_id", "reason", "created_at"}),
    ),
    "candidate_decisions": (
        "candidate_id",
        frozenset(
            {"id", "candidate_id", "action", "actor", "reason", "previous_status",
             "next_status", "created_at"}
        ),
    ),
    "candidates": (
        "hints_json",
        frozenset(
            {"id", "event_id", "target", "title", "body", "confidence", "scope", "risk",
             "status", "claim_key", "hints_json", "evidence_json", "created_at", "updated_at"}
        ),
    ),
    "legacy_evidence": (
        "excerpt",
        frozenset(
            {"id", "event_id", "kind", "uri", "excerpt", "line_start", "line_end", "created_at"}
        ),
    ),
    "events": (
        "triaged_at",
        frozenset(
            {"id", "source_type", "source_uri", "content_hash", "title", "summary", "body",
             "scope", "metadata_json", "created_at", "ingested_at", "triaged_at"}
        ),
    ),
}


def _drop_legacy_tables(conn: sqlite3.Connection) -> None:
    for table, (marker_column, legacy_columns) in _LEGACY_TABLES.items():
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        if not exists:
            continue
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if marker_column in columns or columns <= legacy_columns:
            conn.execute(f"DROP TABLE {table}")


def init_db(conn: sqlite3.Connection) -> None:
    # Drop legacy tables before SCHEMA runs (and before PRAGMA foreign_keys=ON)
    # so legacy cross-table references never block the burn-down.
    _drop_legacy_tables(conn)
    conn.executescript(SCHEMA)
    _migrate_schema(conn)
    conn.commit()


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Reshape legacy tables that CREATE TABLE IF NOT EXISTS cannot touch.

    SCHEMA uses ``CREATE TABLE IF NOT EXISTS`` everywhere, so an existing table
    created by an older schema keeps its old columns forever. These migrations
    bring such tables into line with the canonical SCHEMA definition.
    """
    _migrate_retrieval_uses(conn)


def _canonical_retrieval_uses_ddl() -> str:
    """Extract the canonical ``retrieval_uses`` CREATE statement from SCHEMA.

    Keeping this derived from SCHEMA (rather than duplicated) means the migration
    can never drift from the source-of-truth table definition.
    """
    match = re.search(
        r"CREATE TABLE IF NOT EXISTS retrieval_uses\b.*?\);",
        SCHEMA,
        re.DOTALL,
    )
    if not match:  # pragma: no cover - SCHEMA always defines this table
        raise RuntimeError("retrieval_uses CREATE statement not found in SCHEMA")
    return match.group(0)


# Outcome values the canonical retrieval_uses.outcome CHECK constraint allows.
# Kept in sync with the CHECK in SCHEMA above. Any other value (including NULL or
# legacy 'included') must be sanitized to 'unknown' before copying forward, or the
# INSERT trips ``CHECK constraint failed: outcome IN (...)``.
_RETRIEVAL_OUTCOMES = (
    "improved",
    "failed",
    "neutral",
    "unknown",
    "served",
    "helpful",
    "used",
    "irrelevant",
    "ignored",
    "harmful",
)


def _retrieval_uses_source_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _retrieval_uses_copy_columns(source_cols: set[str]) -> tuple[list[str], list[str]]:
    """Build (canonical_columns, source_exprs) for copying a legacy table forward.

    Defensive on which legacy columns exist: each source expression is assembled
    with explicit ``if 'col' in source_cols`` checks rather than by string-munging a
    template, so the CASE expression that sanitizes ``outcome`` is never mangled.
    Canonical column <- legacy source.
    """
    # outcome: keep only values the canonical CHECK constraint allows; everything
    # else (NULL, legacy 'included', ...) maps to 'unknown'. A CASE expression is
    # used rather than COALESCE because COALESCE preserves invalid non-NULL values.
    if "outcome" in source_cols:
        allowed = ", ".join(f"'{value}'" for value in _RETRIEVAL_OUTCOMES)
        outcome_expr = f"CASE WHEN outcome IN ({allowed}) THEN outcome ELSE 'unknown' END"
    else:
        outcome_expr = "'unknown'"

    # knowledge_id: prefer an existing knowledge_id, else the legacy
    # artifact_or_candidate_id, else NULL.
    knowledge_sources = [
        col for col in ("knowledge_id", "artifact_or_candidate_id") if col in source_cols
    ]
    if not knowledge_sources:
        knowledge_expr = "NULL"
    elif len(knowledge_sources) == 1:
        knowledge_expr = knowledge_sources[0]
    else:
        knowledge_expr = f"COALESCE({', '.join(knowledge_sources)})"

    # served_to_runtime: prefer canonical served_to_runtime, else legacy runtime.
    runtime_sources = [
        col for col in ("served_to_runtime", "runtime") if col in source_cols
    ]
    if not runtime_sources:
        runtime_expr = "NULL"
    elif len(runtime_sources) == 1:
        runtime_expr = runtime_sources[0]
    else:
        runtime_expr = f"COALESCE({', '.join(runtime_sources)})"

    # served_at is NOT NULL in canonical; backfill from legacy created_at (NOT NULL
    # in legacy) whenever served_at is missing or null.
    served_at_sources = [
        col for col in ("served_at", "created_at") if col in source_cols
    ]
    if not served_at_sources:
        served_at_expr = "NULL"
    elif len(served_at_sources) == 1:
        served_at_expr = served_at_sources[0]
    else:
        served_at_expr = f"COALESCE({', '.join(served_at_sources)})"

    def passthrough(col: str) -> str:
        return col if col in source_cols else "NULL"

    target_columns = [
        "id",
        "knowledge_id",
        "served_to_runtime",
        "task_ref",
        "affected_decision",
        "corrected",
        "outcome",
        "note",
        "served_at",
    ]
    source_exprs = [
        "id",
        knowledge_expr,
        runtime_expr,
        passthrough("task_ref"),
        passthrough("affected_decision"),
        passthrough("corrected"),
        outcome_expr,
        passthrough("note"),
        served_at_expr,
    ]
    return target_columns, source_exprs


def _migrate_retrieval_uses(conn: sqlite3.Connection) -> None:
    """Bring a legacy-shaped ``retrieval_uses`` table into the canonical shape.

    Old databases were created by an earlier schema and then column-added via
    ALTER, leaving ``artifact_or_candidate_id``/``created_at``/``runtime``/``query``
    columns behind. ``log_retrieval_use`` only INSERTs the canonical columns, so on
    those DBs the INSERT trips ``NOT NULL constraint failed:
    retrieval_uses.artifact_or_candidate_id``. This rebuilds the table to match
    SCHEMA while preserving every historical row.

    SQLite DDL auto-commits in Python's default isolation mode, so a destroy-before-
    copy rebuild can leave the DB half-migrated if the copy fails (canonical table
    empty + a stray ``retrieval_uses_legacy`` holding the real rows). To stay safe
    and idempotent, this handles three states:

    * **State A (normal legacy)** — ``retrieval_uses`` still has
      ``artifact_or_candidate_id``. Copy rows into a fresh scratch table
      ``retrieval_uses_new`` first; only once that succeeds drop the original and
      rename the scratch into place. The source is never destroyed before the copy.
    * **State B (interrupted)** — ``retrieval_uses`` is already canonical but a stray
      ``retrieval_uses_legacy`` survives from an earlier crashed migration. Recover
      by copying its rows in with ``INSERT OR IGNORE`` (idempotent on the ``id`` PK),
      then drop it.
    * **State C (clean)** — canonical with no stray tables -> no-op.
    """
    # Clear any scratch table left by a prior crashed run before inspecting state.
    conn.execute("DROP TABLE IF EXISTS retrieval_uses_new")

    columns = _retrieval_uses_source_columns(conn, "retrieval_uses")
    has_legacy_stash = bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='retrieval_uses_legacy'"
        ).fetchone()
    )

    is_legacy_shape = "artifact_or_candidate_id" in columns
    if not is_legacy_shape and not has_legacy_stash:
        # State C: already canonical, nothing stashed -> done.
        return

    # The canonical table has knowledge_id REFERENCES knowledge(id). Legacy
    # artifact_or_candidate_id values are NOT canonical knowledge ids, so copying
    # history forward could trip FK enforcement. Disable FK checks for the rebuild
    # (the standard SQLite table-rebuild pattern), then restore the original setting.
    fk_was_on = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        if is_legacy_shape:
            # State A: build a scratch canonical table, copy into it, and only then
            # destroy the source. Derive the scratch DDL from the canonical CREATE so
            # it can never drift from SCHEMA.
            scratch_ddl = _canonical_retrieval_uses_ddl().replace(
                "retrieval_uses", "retrieval_uses_new", 1
            )
            conn.execute(scratch_ddl)
            target_columns, source_exprs = _retrieval_uses_copy_columns(columns)
            conn.execute(
                f"INSERT INTO retrieval_uses_new ({', '.join(target_columns)}) "
                f"SELECT {', '.join(source_exprs)} FROM retrieval_uses"
            )
            conn.execute("DROP TABLE retrieval_uses")
            conn.execute("ALTER TABLE retrieval_uses_new RENAME TO retrieval_uses")
        elif has_legacy_stash:
            # State B: canonical table already exists; recover the stashed rows.
            legacy_cols = _retrieval_uses_source_columns(conn, "retrieval_uses_legacy")
            target_columns, source_exprs = _retrieval_uses_copy_columns(legacy_cols)
            conn.execute(
                f"INSERT OR IGNORE INTO retrieval_uses ({', '.join(target_columns)}) "
                f"SELECT {', '.join(source_exprs)} FROM retrieval_uses_legacy"
            )
            conn.execute("DROP TABLE retrieval_uses_legacy")
    finally:
        # Restore the caller's foreign_keys setting (SCHEMA leaves it ON by default).
        conn.execute(f"PRAGMA foreign_keys={'ON' if fk_was_on else 'OFF'}")


def upsert_evidence(
    conn: sqlite3.Connection,
    *,
    source_type: str,
    claim: str,
    content_hash: str,
    source_uri: str | None = None,
    source_runtime: str | None = None,
    artifact_uri: str | None = None,
    artifact_hash: str | None = None,
    verifier_status: str = "unknown",
    loop_tags: dict[str, Any] | None = None,
    project: str | None = None,
    privacy_scope: str = "workspace",
    occurred_at: str | None = None,
) -> str:
    evidence_id = stable_id("evd", source_uri or "", content_hash)
    conn.execute(
        """
        INSERT INTO evidence (
          id, source_type, source_runtime, source_uri, content_hash, claim,
          artifact_uri, artifact_hash, verifier_status, loop_tags, project,
          privacy_scope, occurred_at, ingested_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_uri, content_hash) DO UPDATE SET
          source_type = excluded.source_type,
          source_runtime = excluded.source_runtime,
          claim = excluded.claim,
          artifact_uri = excluded.artifact_uri,
          artifact_hash = excluded.artifact_hash,
          verifier_status = excluded.verifier_status,
          loop_tags = excluded.loop_tags,
          project = excluded.project,
          privacy_scope = excluded.privacy_scope,
          occurred_at = excluded.occurred_at
        """,
        (
            evidence_id,
            source_type,
            source_runtime,
            source_uri,
            content_hash,
            claim,
            artifact_uri,
            artifact_hash,
            verifier_status,
            json.dumps(loop_tags, sort_keys=True) if loop_tags else None,
            project,
            privacy_scope,
            occurred_at,
            now_iso(),
        ),
    )
    upsert_search_index(
        conn,
        evidence_id,
        f"evidence:{source_type}",
        claim[:160],
        claim,
        source_uri or artifact_uri or evidence_id,
    )
    return evidence_id


def upsert_knowledge(
    conn: sqlite3.Connection,
    *,
    knowledge_type: str,
    gate: str,
    subject: str | None = None,
    predicate: str | None = None,
    value_numeric: float | None = None,
    value_text: str | None = None,
    value_bool: bool | None = None,
    unit: str | None = None,
    target_value: float | None = None,
    slug: str | None = None,
    title: str | None = None,
    body_uri: str | None = None,
    doc_kind: str | None = None,
    status: str = "candidate",
    superseded_by: str | None = None,
    invalidation_reason: str | None = None,
    prescriptive: bool = False,
    inject: bool = False,
    risk: str = "low",
    confidence: float | None = None,
    content_hash: str | None = None,
    loop_tags: dict[str, Any] | None = None,
    project: str | None = None,
    privacy_scope: str = "workspace",
    approved_by: str | None = None,
) -> str:
    validate_knowledge_value(knowledge_type, value_numeric, value_text, value_bool)
    if knowledge_type == "value":
        knowledge_id = stable_id("know", subject or "", predicate or "", project or "")
    else:
        knowledge_id = stable_id("know", slug or title or "", knowledge_type, project or "")
    timestamp = now_iso()
    if prescriptive or knowledge_type == "capability" or risk in {"high", "critical"}:
        gate = "human"
        if status == "current" and not approved_by:
            status = "candidate"
    conn.execute(
        """
        INSERT INTO knowledge (
          id, type, subject, predicate, value_numeric, value_text, value_bool,
          unit, target_value, slug, title, body_uri, doc_kind, status,
          superseded_by, invalidation_reason, gate, prescriptive, inject, risk,
          confidence, content_hash, loop_tags, project, privacy_scope, approved_by,
          created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          value_numeric = excluded.value_numeric,
          value_text = excluded.value_text,
          value_bool = excluded.value_bool,
          unit = excluded.unit,
          target_value = excluded.target_value,
          title = excluded.title,
          body_uri = excluded.body_uri,
          doc_kind = excluded.doc_kind,
          status = excluded.status,
          superseded_by = excluded.superseded_by,
          invalidation_reason = excluded.invalidation_reason,
          gate = excluded.gate,
          prescriptive = excluded.prescriptive,
          inject = excluded.inject,
          risk = excluded.risk,
          confidence = excluded.confidence,
          content_hash = excluded.content_hash,
          loop_tags = excluded.loop_tags,
          privacy_scope = excluded.privacy_scope,
          approved_by = excluded.approved_by,
          updated_at = excluded.updated_at
        """,
        (
            knowledge_id,
            knowledge_type,
            subject,
            predicate,
            value_numeric,
            value_text,
            1 if value_bool is True else 0 if value_bool is False else None,
            unit,
            target_value,
            slug,
            title,
            body_uri,
            doc_kind,
            status,
            superseded_by,
            invalidation_reason,
            gate,
            1 if prescriptive else 0,
            1 if inject else 0,
            risk,
            confidence,
            content_hash,
            json.dumps(loop_tags, sort_keys=True) if loop_tags else None,
            project,
            privacy_scope,
            approved_by,
            timestamp,
            timestamp,
        ),
    )
    upsert_search_index(
        conn,
        knowledge_id,
        f"knowledge:{knowledge_type}",
        title or subject or slug or knowledge_id,
        knowledge_search_body(
            knowledge_type=knowledge_type,
            subject=subject,
            predicate=predicate,
            value_numeric=value_numeric,
            value_text=value_text,
            value_bool=value_bool,
            title=title,
            body_uri=body_uri,
        ),
        body_uri or knowledge_id,
    )
    return knowledge_id


def validate_knowledge_value(
    knowledge_type: str,
    value_numeric: float | None,
    value_text: str | None,
    value_bool: bool | None,
) -> None:
    if knowledge_type != "value":
        return
    set_count = sum(value is not None for value in (value_numeric, value_text, value_bool))
    if set_count != 1:
        raise ValueError("value knowledge must set exactly one typed value field")


def link_knowledge_evidence(
    conn: sqlite3.Connection,
    knowledge_id: str,
    evidence_id: str,
    *,
    relation: str = "supports",
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO knowledge_evidence (
          knowledge_id, evidence_id, relation, created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (knowledge_id, evidence_id, relation, now_iso()),
    )
    apply_composed_privacy_scope(conn, knowledge_id, evidence_id)


def apply_composed_privacy_scope(
    conn: sqlite3.Connection, knowledge_id: str, evidence_id: str
) -> None:
    row = conn.execute(
        """
        SELECT
          knowledge.privacy_scope AS knowledge_scope,
          evidence.privacy_scope AS evidence_scope
        FROM knowledge
        JOIN evidence ON evidence.id = ?
        WHERE knowledge.id = ?
        """,
        (evidence_id, knowledge_id),
    ).fetchone()
    if row is None:
        return
    composed_scope = most_restrictive_scope(row["knowledge_scope"], row["evidence_scope"])
    if composed_scope == row["knowledge_scope"]:
        return
    conn.execute(
        """
        UPDATE knowledge
        SET privacy_scope = ?, updated_at = ?
        WHERE id = ?
        """,
        (composed_scope, now_iso(), knowledge_id),
    )


def most_restrictive_scope(*scopes: str | None) -> str:
    valid_scopes = [scope for scope in scopes if scope in SCOPE_RANK]
    if not valid_scopes:
        return "workspace"
    return min(valid_scopes, key=lambda scope: SCOPE_RANK[scope])


def upsert_search_index(
    conn: sqlite3.Connection,
    doc_id: str,
    kind: str,
    title: str,
    body: str,
    path: str,
) -> None:
    conn.execute("DELETE FROM search_index WHERE doc_id = ?", (doc_id,))
    conn.execute(
        "INSERT INTO search_index (doc_id, kind, title, body, path) VALUES (?, ?, ?, ?, ?)",
        (doc_id, kind, title, body, path),
    )


def knowledge_search_body(**kwargs: Any) -> str:
    return " ".join(str(value) for value in kwargs.values() if value is not None)


def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    scopes: tuple[str, ...] | None = PUBLIC_SCOPES,
    filters: dict[str, Any] | None = None,
) -> list[sqlite3.Row]:
    normalized_query = normalize_fts_query(query)
    if not normalized_query:
        return []
    clauses = ["search_index MATCH ?"]
    params: list[Any] = [normalized_query]
    if scopes:
        placeholders = ",".join("?" for _ in scopes)
        clauses.append(
            f"COALESCE(knowledge.privacy_scope, evidence.privacy_scope) IN ({placeholders})"
        )
        params.extend(scopes)
    filters = filters or {}
    if filters.get("project"):
        clauses.append("(knowledge.project = ? OR evidence.project = ?)")
        params.extend([filters["project"], filters["project"]])
    if filters.get("type"):
        clauses.append("knowledge.type = ?")
        params.append(filters["type"])
    if filters.get("status"):
        clauses.append("knowledge.status = ?")
        params.append(filters["status"])
    if filters.get("loop_id"):
        clauses.append(
            "(knowledge.loop_tags LIKE ? OR evidence.loop_tags LIKE ?)"
        )
        needle = f'%\"loop_id\": \"{filters["loop_id"]}\"%'
        params.extend([needle, needle])
    if filters.get("family"):
        clauses.append(
            "(knowledge.loop_tags LIKE ? OR evidence.loop_tags LIKE ?)"
        )
        needle = f'%\"family\": \"{filters["family"]}\"%'
        params.extend([needle, needle])
    params.append(limit)
    return list(
        conn.execute(
            f"""
            SELECT
              search_index.doc_id,
              search_index.kind,
              search_index.title,
              snippet(search_index, 3, '[', ']', ' ... ', 12) AS snippet,
              search_index.path,
              COALESCE(knowledge.privacy_scope, evidence.privacy_scope) AS scope
            FROM search_index
            LEFT JOIN knowledge ON knowledge.id = search_index.doc_id
            LEFT JOIN evidence ON evidence.id = search_index.doc_id
            WHERE {' AND '.join(clauses)}
            ORDER BY rank
            LIMIT ?
            """,
            params,
        )
    )


def normalize_fts_query(query: str) -> str:
    terms = re.findall(r"[\w-]{2,}", query.lower())
    return " OR ".join(f'"{term}"' for term in terms[:8])


def get_knowledge(conn: sqlite3.Connection, knowledge_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM knowledge WHERE id = ?", (knowledge_id,)).fetchone()


def knowledge_evidence(conn: sqlite3.Connection, knowledge_id: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in conn.execute(
            """
            SELECT
              knowledge_evidence.relation,
              evidence.id,
              evidence.source_type,
              evidence.source_runtime,
              evidence.source_uri,
              evidence.content_hash,
              evidence.claim,
              evidence.artifact_uri,
              evidence.artifact_hash,
              evidence.verifier_status,
              evidence.privacy_scope,
              evidence.occurred_at
            FROM knowledge_evidence
            JOIN evidence ON evidence.id = knowledge_evidence.evidence_id
            WHERE knowledge_evidence.knowledge_id = ?
            ORDER BY knowledge_evidence.created_at ASC, evidence.id ASC
            """,
            (knowledge_id,),
        )
    ]


def list_current_knowledge(
    conn: sqlite3.Connection,
    *,
    project: str | None = None,
    scopes: tuple[str, ...] = PUBLIC_SCOPES,
    limit: int = 20,
    knowledge_type: str | None = None,
    doc_kind: str | None = None,
    inject_only: bool = False,
) -> list[sqlite3.Row]:
    clauses = ["status = 'current'"]
    params: list[Any] = []
    if scopes:
        placeholders = ",".join("?" for _ in scopes)
        clauses.append(f"privacy_scope IN ({placeholders})")
        params.extend(scopes)
    if project:
        clauses.append("(project = ? OR project IS NULL)")
        params.append(project)
    if knowledge_type:
        clauses.append("type = ?")
        params.append(knowledge_type)
    if doc_kind:
        clauses.append("doc_kind = ?")
        params.append(doc_kind)
    if inject_only:
        clauses.append("inject = 1")
    params.append(limit)
    return list(
        conn.execute(
            f"""
            SELECT *
            FROM knowledge
            WHERE {' AND '.join(clauses)}
            ORDER BY
              inject DESC,
              COALESCE(confidence, 0) DESC,
              updated_at DESC,
              id ASC
            LIMIT ?
            """,
            params,
        )
    )


def list_knowledge(
    conn: sqlite3.Connection,
    *,
    status: str | None = None,
    knowledge_type: str | None = None,
    scopes: tuple[str, ...] | None = PUBLIC_SCOPES,
    limit: int = 20,
) -> list[sqlite3.Row]:
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if knowledge_type:
        clauses.append("type = ?")
        params.append(knowledge_type)
    if scopes:
        placeholders = ",".join("?" for _ in scopes)
        clauses.append(f"privacy_scope IN ({placeholders})")
        params.extend(scopes)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    return list(
        conn.execute(
            f"""
            SELECT *
            FROM knowledge
            {where}
            ORDER BY updated_at DESC, id ASC
            LIMIT ?
            """,
            params,
        )
    )


def get_current_doc(
    conn: sqlite3.Connection,
    *,
    slug: str,
    scopes: tuple[str, ...] = PUBLIC_SCOPES,
    doc_kind: str | None = None,
) -> sqlite3.Row | None:
    clauses = ["status = 'current'", "type = 'doc'", "slug = ?"]
    params: list[Any] = [slug]
    if scopes:
        placeholders = ",".join("?" for _ in scopes)
        clauses.append(f"privacy_scope IN ({placeholders})")
        params.extend(scopes)
    if doc_kind:
        clauses.append("doc_kind = ?")
        params.append(doc_kind)
    return conn.execute(
        f"""
        SELECT *
        FROM knowledge
        WHERE {' AND '.join(clauses)}
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        params,
    ).fetchone()


def knowledge_value(row: sqlite3.Row) -> Any:
    if row["value_bool"] is not None:
        return bool(row["value_bool"])
    if row["value_numeric"] is not None:
        if row["unit"]:
            return {"value": row["value_numeric"], "unit": row["unit"]}
        return row["value_numeric"]
    return row["value_text"]


def knowledge_summary(
    row: sqlite3.Row, evidence: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": row["id"],
        "type": row["type"],
        "status": row["status"],
        "gate": row["gate"],
        "risk": row["risk"],
        "confidence": row["confidence"],
        "inject": bool(row["inject"]),
        "project": row["project"],
        "privacy_scope": row["privacy_scope"],
        "updated_at": row["updated_at"],
    }
    if row["type"] == "value":
        payload.update(
            {
                "subject": row["subject"],
                "predicate": row["predicate"],
                "value": knowledge_value(row),
                "target_value": row["target_value"],
            }
        )
    else:
        payload.update(
            {
                "slug": row["slug"],
                "title": row["title"],
                "body_uri": row["body_uri"],
                "doc_kind": row["doc_kind"],
            }
        )
    if evidence is not None:
        payload["evidence"] = evidence
    return payload


def knowledge_digest(
    conn: sqlite3.Connection,
    *,
    project: str | None = None,
    scopes: tuple[str, ...] = PUBLIC_SCOPES,
    limit: int = 12,
) -> dict[str, Any]:
    return {
        "counts": counts(conn),
        "project": project,
        "scopes": list(scopes),
        "memory": [
            knowledge_summary(row)
            for row in list_current_knowledge(
                conn, project=project, scopes=scopes, limit=limit, inject_only=True
            )
        ],
        "values": [
            knowledge_summary(row)
            for row in list_current_knowledge(
                conn, project=project, scopes=scopes, limit=limit, knowledge_type="value"
            )
        ],
        "documents": [
            knowledge_summary(row)
            for row in list_current_knowledge(
                conn, project=project, scopes=scopes, limit=limit, knowledge_type="doc"
            )
        ],
        "capabilities": [
            knowledge_summary(row)
            for row in list_current_knowledge(
                conn, project=project, scopes=scopes, limit=limit, knowledge_type="capability"
            )
        ],
        "loop_families": loop_family_rows(conn, limit=limit),
    }


def loop_family_rows(conn: sqlite3.Connection, *, limit: int = 12) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in conn.execute(
            """
            SELECT *
            FROM family_scores
            ORDER BY refreshed_at DESC, loop_id ASC, family ASC
            LIMIT ?
            """,
            (limit,),
        )
    ]


def render_doc_markdown(conn: sqlite3.Connection, row: sqlite3.Row) -> str:
    evidence = knowledge_evidence(conn, row["id"])
    title = row["title"] or row["slug"] or row["id"]
    lines = [
        f"# {title}",
        "",
        f"- Brain ID: `{row['id']}`",
        f"- Type: `{row['type']}` / `{row['doc_kind'] or 'doc'}`",
        f"- Status: `{row['status']}`",
        f"- Gate: `{row['gate']}`",
        f"- Scope: `{row['privacy_scope']}`",
    ]
    if row["body_uri"]:
        lines.append(f"- Body URI: `{row['body_uri']}`")
    if row["confidence"] is not None:
        lines.append(f"- Confidence: `{row['confidence']}`")
    lines += ["", "## Evidence", ""]
    if not evidence:
        lines.append("- No linked evidence.")
    else:
        for item in evidence:
            source = item["source_uri"] or item["artifact_uri"] or item["id"]
            relation = item["relation"] or "supports"
            lines.append(f"- `{relation}` [{item['id']}] {item['claim']} (`{source}`)")
    return "\n".join(lines).rstrip() + "\n"


def log_retrieval_use(
    conn: sqlite3.Connection,
    knowledge_id: str | None,
    *,
    runtime: str | None = None,
    task_ref: str | None = None,
    outcome: str = "served",
    note: str | None = None,
) -> str:
    created_at = now_iso()
    sequence = conn.execute("SELECT COUNT(*) FROM retrieval_uses").fetchone()[0]
    retrieval_id = stable_id(
        "ret",
        knowledge_id or "",
        runtime or "",
        task_ref or "",
        outcome,
        note or "",
        created_at,
        str(sequence),
    )
    conn.execute(
        """
        INSERT INTO retrieval_uses (
          id, knowledge_id, served_to_runtime, task_ref, outcome, note, served_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (retrieval_id, knowledge_id, runtime, task_ref, outcome, note, created_at),
    )
    return retrieval_id


def update_retrieval_use_feedback(
    conn: sqlite3.Connection,
    retrieval_use_id: str,
    *,
    outcome: str,
    note: str | None = None,
) -> bool:
    cursor = conn.execute(
        """
        UPDATE retrieval_uses
        SET outcome = ?, note = ?
        WHERE id = ?
        """,
        (outcome, note, retrieval_use_id),
    )
    return cursor.rowcount > 0


def mark_knowledge_stale(
    conn: sqlite3.Connection,
    knowledge_id: str,
    *,
    reason: str = "user_request",
) -> bool:
    cursor = conn.execute(
        """
        UPDATE knowledge
        SET status = 'stale',
            invalidation_reason = ?,
            updated_at = ?
        WHERE id = ? AND status != 'archived'
        """,
        (reason, now_iso(), knowledge_id),
    )
    return cursor.rowcount > 0


def approve_knowledge(
    conn: sqlite3.Connection,
    knowledge_id: str,
    *,
    actor: str,
) -> bool:
    cursor = conn.execute(
        """
        UPDATE knowledge
        SET status = 'current',
            approved_by = ?,
            updated_at = ?
        WHERE id = ? AND gate = 'human' AND status = 'candidate'
        """,
        (actor, now_iso(), knowledge_id),
    )
    return cursor.rowcount > 0


def reject_knowledge(
    conn: sqlite3.Connection,
    knowledge_id: str,
    *,
    reason: str = "rejected",
) -> bool:
    cursor = conn.execute(
        """
        UPDATE knowledge
        SET status = 'archived',
            invalidation_reason = ?,
            updated_at = ?
        WHERE id = ? AND gate = 'human' AND status = 'candidate'
        """,
        (reason, now_iso(), knowledge_id),
    )
    return cursor.rowcount > 0


def counts(conn: sqlite3.Connection) -> dict[str, Any]:
    evidence_count = conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    knowledge_count = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
    by_knowledge_type = {
        row["type"]: row["count"]
        for row in conn.execute(
            "SELECT type, COUNT(*) AS count FROM knowledge GROUP BY type ORDER BY type"
        )
    }
    by_status = {
        row["status"]: row["count"]
        for row in conn.execute(
            "SELECT status, COUNT(*) AS count FROM knowledge GROUP BY status ORDER BY status"
        )
    }
    return {
        "evidence": evidence_count,
        "knowledge": knowledge_count,
        "by_knowledge_type": by_knowledge_type,
        "by_status": by_status,
    }
