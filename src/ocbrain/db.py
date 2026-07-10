from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ocbrain.ids import content_hash, stable_id
from ocbrain.text import find_probable_injection, find_probable_secret_leaks

DEFAULT_DB_PATH = Path(os.environ.get("OCBRAIN_DB", "~/.ocbrain/ocbrain.sqlite")).expanduser()
# The brain DB has heavy concurrent writers (light/heavy autopilot cycles,
# stallcheck, MCP feedback). Wait on a lock rather than fail-fast so migrations
# and ledger writes don't crash with "database is locked". Set in the core
# connect() factory so ALL consumers inherit it. Config-overridable via env.
DB_BUSY_TIMEOUT_MS = int(os.environ.get("OCBRAIN_BUSY_TIMEOUT_MS", "5000"))
PUBLIC_SCOPES = ("workspace", "project", "public")
SCOPE_RANK = {"private": 0, "workspace": 1, "project": 2, "public": 3}


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS candidates;
DROP TABLE IF EXISTS artifact_links;
DROP TABLE IF EXISTS invalidations;
DROP TABLE IF EXISTS candidate_decisions;
DROP TABLE IF EXISTS legacy_evidence;

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
CREATE INDEX IF NOT EXISTS idx_retrieval_uses_knowledge_outcome_served
  ON retrieval_uses(knowledge_id, outcome, served_at);

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
CREATE INDEX IF NOT EXISTS idx_brain_events_correction_target
  ON brain_events(json_extract(body_json, '$.target_id'))
  WHERE kind = 'correction_recorded';

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

CREATE INDEX IF NOT EXISTS idx_knowledge_updated_id ON knowledge(updated_at, id);

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

-- v0.2 additive tables (spec §1.2). CREATE ... IF NOT EXISTS keeps init_db idempotent.
CREATE TABLE IF NOT EXISTS signal_events (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  polarity TEXT NOT NULL,
  weight REAL NOT NULL,
  source TEXT NOT NULL,
  source_ref TEXT NOT NULL,
  session_key TEXT,
  knowledge_id TEXT REFERENCES knowledge(id),
  evidence_id TEXT REFERENCES evidence(id),
  details TEXT,
  occurred_at TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sig_knowledge ON signal_events(knowledge_id, created_at);
CREATE INDEX IF NOT EXISTS idx_sig_kind ON signal_events(kind, created_at);
CREATE INDEX IF NOT EXISTS idx_sig_session ON signal_events(session_key);

CREATE TABLE IF NOT EXISTS harvest_watermarks (
  domain TEXT NOT NULL,
  stream TEXT NOT NULL,
  watermark TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (domain, stream)
);

CREATE TABLE IF NOT EXISTS judge_runs (
  id TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  model TEXT NOT NULL,
  status TEXT NOT NULL,
  item_count INTEGER NOT NULL DEFAULT 0,
  request_hash TEXT,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  cost_usd REAL NOT NULL DEFAULT 0,
  response_json TEXT,
  egress_audit_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_judge_ts ON judge_runs(ts);

CREATE TABLE IF NOT EXISTS autopilot_runs (
  id TEXT PRIMARY KEY,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL DEFAULT 'running',
  stages_json TEXT,
  error TEXT
);

CREATE TABLE IF NOT EXISTS dataset_examples (
  id TEXT PRIMARY KEY,
  dataset TEXT CHECK (dataset IN ('sft','dpo','persona')) NOT NULL,
  content_hash TEXT NOT NULL,
  dedup_key TEXT NOT NULL,
  source_kind TEXT CHECK (source_kind IN
    ('openclaw_session','claude_session','codex_session',
     'correction_event','git_commit','authored_doc')) NOT NULL,
  source_uri TEXT,
  source_span TEXT,
  evidence_ids TEXT NOT NULL,
  privacy_scope TEXT CHECK (privacy_scope IN ('private','workspace','project','public'))
    NOT NULL DEFAULT 'workspace',
  quality_label TEXT CHECK (quality_label IN ('good','neutral','bad','excluded')) NOT NULL,
  quality_confidence REAL,
  quality_reasons TEXT,
  n_turns INTEGER,
  n_chars INTEGER,
  example_json TEXT NOT NULL,
  session_id TEXT,
  occurred_at TEXT,
  grade_score REAL,
  grade_json TEXT,
  grade_model TEXT,
  grade_prompt_version TEXT,
  graded_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(dataset, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_dsx_dataset_label ON dataset_examples(dataset, quality_label);
CREATE INDEX IF NOT EXISTS idx_dsx_dedup ON dataset_examples(dataset, dedup_key);
CREATE INDEX IF NOT EXISTS idx_dsx_occurred ON dataset_examples(dataset, occurred_at);

CREATE TABLE IF NOT EXISTS dataset_sources (
  source_uri TEXT NOT NULL,
  dataset TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  mined_at TEXT NOT NULL,
  examples_emitted INTEGER NOT NULL DEFAULT 0,
  status TEXT CHECK (status IN ('mined','skipped','error')) NOT NULL DEFAULT 'mined',
  detail TEXT,
  PRIMARY KEY (source_uri, dataset)
);

CREATE TABLE IF NOT EXISTS dataset_grade_runs (
  id TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  model TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  item_count INTEGER NOT NULL DEFAULT 0,
  error_count INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  request_hash TEXT,
  error TEXT
);
CREATE INDEX IF NOT EXISTS idx_dataset_grade_runs_ts ON dataset_grade_runs(ts);

-- v0.3 additive tables. CREATE ... IF NOT EXISTS keeps init_db idempotent.
-- embed_runs: one row per embedding batch, for daily-cap accounting + auditing.
CREATE TABLE IF NOT EXISTS embed_runs (
  id TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  items INTEGER NOT NULL DEFAULT 0,
  cost_usd REAL NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  error TEXT
);
CREATE INDEX IF NOT EXISTS idx_embed_runs_ts ON embed_runs(ts);

-- projection_cursor: single-row incremental cursor so the projection rebuild can
-- fold only events after last_event_rowid instead of full-folding every event.
CREATE TABLE IF NOT EXISTS projection_cursor (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  last_event_rowid INTEGER,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS dataset_exports (
  id TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  dataset TEXT NOT NULL,
  path TEXT NOT NULL,
  min_scope TEXT NOT NULL,
  min_label TEXT NOT NULL,
  min_grade REAL,
  format TEXT NOT NULL,
  example_count INTEGER NOT NULL,
  excluded_count INTEGER NOT NULL,
  bytes INTEGER NOT NULL,
  payload_hash TEXT NOT NULL,
  manifest_json TEXT NOT NULL,
  egress_audit_id TEXT
);
"""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def connect(path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _migrate_schema(conn)
    conn.commit()


# v0.2 additive columns (spec §1.1). Applied via _ensure_column so both fresh
# and existing DBs converge without any table rebuild or CHECK edits. Labels are
# enforced in code, not by CHECK constraints (additive ALTER cannot add CHECKs).
_V2_KNOWLEDGE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("origin", "TEXT"),
    ("quarantine_reason", "TEXT"),
    ("quality_label", "TEXT"),
    ("quality_confidence", "REAL"),
    ("quality_updated_at", "TEXT"),
    ("promote_score", "REAL"),
)
_V2_EVIDENCE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("injection_scan_status", "TEXT"),
    ("injection_scan_hits", "TEXT"),
)
# v0.3 additive columns. embedding holds the raw vector bytes; embedded_at marks
# when it was computed (NULL == not yet embedded / stale, eligible for re-embed).
_V3_KNOWLEDGE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("embedding", "BLOB"),
    ("embedded_at", "TEXT"),
)

# Dataset-grading additions. Kept additive so the live corpus migrates without
# rebuilding or rewriting the large ``dataset_examples`` table.
_V4_DATASET_EXAMPLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("grade_score", "REAL"),
    ("grade_json", "TEXT"),
    ("grade_model", "TEXT"),
    ("grade_prompt_version", "TEXT"),
    ("graded_at", "TEXT"),
)
_V4_DATASET_EXPORT_COLUMNS: tuple[tuple[str, str], ...] = (("min_grade", "REAL"),)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    """Add ``column`` to ``table`` if absent. Additive-only; never rebuilds."""
    cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


_MEMORY_VIEW_SQL = """
CREATE VIEW memory AS
  SELECT * FROM knowledge
  WHERE status = 'current' AND inject = 1 AND quarantine_reason IS NULL
"""


def _normalized_sql(sql: str) -> str:
    return re.sub(r"\s+", "", sql).lower().rstrip(";")


def _rebuild_memory_view(conn: sqlite3.Connection) -> bool:
    """Ensure the injectable ``memory`` view excludes quarantined rows (§1.3).

    Views hold no data, so DROP + CREATE is additive-safe when migration is
    actually needed. A current definition is left untouched: unconditional DDL
    made every read-oriented CLI startup contend as a schema writer.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'view' AND name = 'memory'"
    ).fetchone()
    if row is not None and _normalized_sql(row["sql"] or "") == _normalized_sql(_MEMORY_VIEW_SQL):
        return False
    conn.execute("DROP VIEW IF EXISTS memory")
    conn.execute(_MEMORY_VIEW_SQL)
    return True


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Reshape legacy tables that CREATE TABLE IF NOT EXISTS cannot touch.

    SCHEMA uses ``CREATE TABLE IF NOT EXISTS`` everywhere, so an existing table
    created by an older schema keeps its old columns forever. These migrations
    bring such tables into line with the canonical SCHEMA definition, then apply
    the additive v0.2 columns and rebuild the quarantine-aware memory view.
    """
    _migrate_retrieval_uses(conn)
    _repair_orphan_retrieval_uses(conn)
    for column, decl in _V2_KNOWLEDGE_COLUMNS:
        _ensure_column(conn, "knowledge", column, decl)
    for column, decl in _V3_KNOWLEDGE_COLUMNS:
        _ensure_column(conn, "knowledge", column, decl)
    for column, decl in _V2_EVIDENCE_COLUMNS:
        _ensure_column(conn, "evidence", column, decl)
    for column, decl in _V4_DATASET_EXAMPLE_COLUMNS:
        _ensure_column(conn, "dataset_examples", column, decl)
    for column, decl in _V4_DATASET_EXPORT_COLUMNS:
        _ensure_column(conn, "dataset_exports", column, decl)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dsx_grade ON dataset_examples(dataset, grade_score)"
    )
    _rebuild_memory_view(conn)


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

    # ``artifact_or_candidate_id`` was polymorphic legacy provenance, not a
    # canonical knowledge foreign key. Never copy it into knowledge_id: doing so
    # creates FK violations for old candidate/event ids. A genuinely canonical
    # knowledge_id is preserved here and validated after the rebuild.
    knowledge_expr = "knowledge_id" if "knowledge_id" in source_cols else "NULL"

    # served_to_runtime: prefer canonical served_to_runtime, else legacy runtime.
    runtime_sources = [col for col in ("served_to_runtime", "runtime") if col in source_cols]
    if not runtime_sources:
        runtime_expr = "NULL"
    elif len(runtime_sources) == 1:
        runtime_expr = runtime_sources[0]
    else:
        runtime_expr = f"COALESCE({', '.join(runtime_sources)})"

    # served_at is NOT NULL in canonical; backfill from legacy created_at (NOT NULL
    # in legacy) whenever served_at is missing or null.
    served_at_sources = [col for col in ("served_at", "created_at") if col in source_cols]
    if not served_at_sources:
        served_at_expr = "NULL"
    elif len(served_at_sources) == 1:
        served_at_expr = served_at_sources[0]
    else:
        served_at_expr = f"COALESCE({', '.join(served_at_sources)})"

    def passthrough(col: str) -> str:
        return col if col in source_cols else "NULL"

    # Preserve the old polymorphic id as explicit provenance rather than losing
    # it or pretending it references knowledge. task_ref is the canonical free-
    # form provenance slot and remains queryable after migration.
    task_ref_expr = passthrough("task_ref")
    if "artifact_or_candidate_id" in source_cols:
        legacy_ref = "'legacy-object:' || artifact_or_candidate_id"
        task_ref_expr = f"COALESCE({task_ref_expr}, {legacy_ref})"

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
        task_ref_expr,
        passthrough("affected_decision"),
        passthrough("corrected"),
        outcome_expr,
        passthrough("note"),
        served_at_expr,
    ]
    return target_columns, source_exprs


def _repair_orphan_retrieval_uses(conn: sqlite3.Connection) -> int:
    """Repair canonical rows whose optional knowledge FK points nowhere.

    Five pre-v0.2 rows in the live database carried candidate/event ids through
    the legacy migration. Preserve that value in ``task_ref`` and clear the
    invalid optional FK; no retrieval-use row is deleted.
    """
    rows = conn.execute(
        """
        SELECT id, knowledge_id
        FROM retrieval_uses
        WHERE knowledge_id IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM knowledge WHERE knowledge.id = retrieval_uses.knowledge_id
          )
        """
    ).fetchall()
    for row in rows:
        conn.execute(
            """
            UPDATE retrieval_uses
            SET task_ref = COALESCE(task_ref, 'legacy-object:' || knowledge_id),
                knowledge_id = NULL
            WHERE id = ?
            """,
            (row["id"],),
        )
    return len(rows)


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
    evidence_existed = (
        conn.execute("SELECT 1 FROM evidence WHERE id = ? LIMIT 1", (evidence_id,)).fetchone()
        is not None
    )
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
        replace=evidence_existed,
    )
    return evidence_id


def _record_clobber_refused(conn: sqlite3.Connection, knowledge_id: str, *, actor: str) -> str:
    """Persist a neutral ``clobber_refused`` signal breadcrumb (spec §5.1-2, §5.2).

    Uses the same stable-id recipe as the autolabel lane's ``record_signal`` so a
    repeated refusal from the same actor collapses to one row (INSERT OR IGNORE).
    """
    source = "events"
    kind = "clobber_refused"
    details = {"actor": actor, "reason": "human_origin_no_clobber"}
    details_json = json.dumps(details, sort_keys=True, separators=(",", ":"))
    signal_id = stable_id("sig", source, knowledge_id, kind, content_hash(details_json))
    timestamp = now_iso()
    conn.execute(
        """
        INSERT OR IGNORE INTO signal_events (
          id, kind, polarity, weight, source, source_ref, session_key,
          knowledge_id, evidence_id, details, occurred_at, created_at
        )
        VALUES (?, ?, 'neutral', 0.1, ?, ?, NULL, ?, NULL, ?, ?, ?)
        """,
        (signal_id, kind, source, knowledge_id, knowledge_id, details_json, timestamp, timestamp),
    )
    return signal_id


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
    origin: str = "autopilot",
    actor: str = "ocbrain",
) -> str:
    validate_knowledge_value(knowledge_type, value_numeric, value_text, value_bool)
    if knowledge_type == "value":
        knowledge_id = stable_id("know", subject or "", predicate or "", project or "")
    else:
        knowledge_id = stable_id("know", slug or title or "", knowledge_type, project or "")
    timestamp = now_iso()

    # Human-provenance no-clobber guard (spec §5.1-2): a non-human writer may not
    # overwrite a row an actual human authored. Record an audit breadcrumb and
    # leave the existing row untouched.
    existing = conn.execute("SELECT origin FROM knowledge WHERE id = ?", (knowledge_id,)).fetchone()
    if existing is not None and existing["origin"] == "human" and not actor.startswith("human"):
        _record_clobber_refused(conn, knowledge_id, actor=actor)
        return knowledge_id

    # Injectable guard (spec §5.1-2): scanning happens only when the caller asks
    # for injection. A hit forces inject=0 and quarantines the new row so it can
    # never reach the memory view. On conflict, quarantine is preserved (never
    # cleared) by the ON CONFLICT clause below, so this stamp lands on inserts.
    inject_flag = bool(inject)
    quarantine_reason: str | None = None
    if inject_flag:
        scan_target = value_text or title or ""
        hits = find_probable_injection(scan_target) + find_probable_secret_leaks(scan_target)
        if hits:
            inject_flag = False
            quarantine_reason = "injection_scan:" + ",".join(hits)

    conn.execute(
        """
        INSERT INTO knowledge (
          id, type, subject, predicate, value_numeric, value_text, value_bool,
          unit, target_value, slug, title, body_uri, doc_kind, status,
          superseded_by, invalidation_reason, gate, prescriptive, inject, risk,
          confidence, content_hash, loop_tags, project, privacy_scope, approved_by,
          origin, quarantine_reason, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
          origin = COALESCE(knowledge.origin, excluded.origin),
          quarantine_reason = knowledge.quarantine_reason,
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
            1 if inject_flag else 0,
            risk,
            confidence,
            content_hash,
            json.dumps(loop_tags, sort_keys=True) if loop_tags else None,
            project,
            privacy_scope,
            approved_by,
            origin,
            quarantine_reason,
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
        replace=existing is not None,
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
    *,
    replace: bool = True,
) -> None:
    # FTS5's UNINDEXED doc_id cannot support a normal lookup index. Avoid an
    # O(all FTS rows) DELETE when the parent row is new; this is the hot path for
    # harvested evidence and safeguard tripwires. Existing parents still replace
    # their one index row so updates remain exact.
    if replace:
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
        clauses.append("(knowledge.loop_tags LIKE ? OR evidence.loop_tags LIKE ?)")
        needle = f'%"loop_id": "{filters["loop_id"]}"%'
        params.extend([needle, needle])
    if filters.get("family"):
        clauses.append("(knowledge.loop_tags LIKE ? OR evidence.loop_tags LIKE ?)")
        needle = f'%"family": "{filters["family"]}"%'
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
            WHERE {" AND ".join(clauses)}
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
    clauses = ["status = 'current'", "quarantine_reason IS NULL"]
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
            WHERE {" AND ".join(clauses)}
            ORDER BY
              inject DESC,
              COALESCE(promote_score, -1) DESC,
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
        WHERE {" AND ".join(clauses)}
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


def admit_knowledge(
    conn: sqlite3.Connection,
    knowledge_id: str,
    *,
    actor: str = "ocbrain-autopilot",
) -> bool:
    """Promote a candidate to ``current`` iff it is not quarantined (spec §5.1-3).

    The canonical admission path in v0.2 — the human gate is gone, so any actor
    may admit, but a quarantined row (``quarantine_reason IS NOT NULL``) can never
    be admitted. Stamps ``approved_by=actor``.
    """
    cursor = conn.execute(
        """
        UPDATE knowledge
        SET status = 'current',
            approved_by = ?,
            updated_at = ?
        WHERE id = ? AND status = 'candidate' AND quarantine_reason IS NULL
        """,
        (actor, now_iso(), knowledge_id),
    )
    return cursor.rowcount > 0


def approve_knowledge(
    conn: sqlite3.Connection,
    knowledge_id: str,
    *,
    actor: str,
) -> bool:
    """Deprecated: thin wrapper over :func:`admit_knowledge` (spec §5.1-3).

    The old ``gate = 'human'`` predicate is gone; admission now hinges only on
    candidate status + no quarantine.
    """
    return admit_knowledge(conn, knowledge_id, actor=actor)


def reject_knowledge(
    conn: sqlite3.Connection,
    knowledge_id: str,
    *,
    reason: str = "rejected",
) -> bool:
    """Deprecated: archive a candidate (spec §5.1-3, ``gate='human'`` predicate dropped)."""
    cursor = conn.execute(
        """
        UPDATE knowledge
        SET status = 'archived',
            invalidation_reason = ?,
            updated_at = ?
        WHERE id = ? AND status = 'candidate'
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
