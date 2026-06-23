from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ocbrain.ids import stable_id
from ocbrain.schema import Candidate, Evidence
from ocbrain.text import claim_key

DEFAULT_DB_PATH = Path(os.environ.get("OCBRAIN_DB", "~/.ocbrain/ocbrain.sqlite")).expanduser()
REVIEWED_OUTPUT_STATUSES = ("approved", "proposed", "applied")


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  source_type TEXT NOT NULL,
  source_uri TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  body TEXT NOT NULL,
  scope TEXT NOT NULL DEFAULT 'workspace',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT,
  ingested_at TEXT NOT NULL,
  triaged_at TEXT,
  UNIQUE(source_uri, content_hash)
);

CREATE TABLE IF NOT EXISTS evidence (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  uri TEXT NOT NULL,
  excerpt TEXT NOT NULL,
  line_start INTEGER,
  line_end INTEGER,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates (
  id TEXT PRIMARY KEY,
  event_id TEXT REFERENCES events(id) ON DELETE SET NULL,
  target TEXT NOT NULL,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  confidence REAL NOT NULL,
  scope TEXT NOT NULL,
  risk TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'draft',
  claim_key TEXT NOT NULL DEFAULT '',
  hints_json TEXT NOT NULL DEFAULT '[]',
  evidence_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(event_id, target, title, body)
);

CREATE TABLE IF NOT EXISTS artifact_links (
  id TEXT PRIMARY KEY,
  candidate_id TEXT NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
  surface TEXT NOT NULL,
  uri TEXT NOT NULL,
  line_start INTEGER,
  line_end INTEGER,
  applied_at TEXT,
  applied_by TEXT
);

CREATE TABLE IF NOT EXISTS retrieval_uses (
  id TEXT PRIMARY KEY,
  artifact_or_candidate_id TEXT NOT NULL,
  runtime TEXT,
  query TEXT,
  outcome TEXT,
  note TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS invalidations (
  id TEXT PRIMARY KEY,
  old_candidate_id TEXT NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
  new_candidate_id TEXT REFERENCES candidates(id) ON DELETE SET NULL,
  reason TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidate_decisions (
  id TEXT PRIMARY KEY,
  candidate_id TEXT NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
  action TEXT NOT NULL,
  actor TEXT NOT NULL,
  reason TEXT NOT NULL,
  previous_status TEXT NOT NULL,
  next_status TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS loop_programs (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  project TEXT,
  owner TEXT,
  objective TEXT NOT NULL,
  primary_metric_name TEXT,
  primary_metric_direction TEXT CHECK (
    primary_metric_direction IN ('higher_is_better','lower_is_better','target','boolean')
  ),
  baseline_value TEXT,
  baseline_measured_at TEXT,
  guardrails_json TEXT,
  allowed_scope_json TEXT,
  blocked_actions_json TEXT,
  budget_json TEXT,
  schedule_json TEXT,
  verifier_ref TEXT,
  status TEXT CHECK (
    status IN ('draft','enabled','paused','disabled','archived')
  ) DEFAULT 'draft',
  risk TEXT CHECK (risk IN ('low','medium','high','critical')) DEFAULT 'medium',
  privacy_scope TEXT CHECK (
    privacy_scope IN ('private','workspace','project','public')
  ) DEFAULT 'workspace',
  definition_uri TEXT,
  content_hash TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS loop_runs (
  id TEXT PRIMARY KEY,
  loop_id TEXT NOT NULL REFERENCES loop_programs(id),
  trigger_type TEXT CHECK (
    trigger_type IN ('heartbeat','cron','manual','taskflow','api','unknown')
  ),
  trigger_ref TEXT,
  orchestrator_session_uri TEXT,
  backlog_snapshot_uri TEXT,
  backlog_snapshot_hash TEXT,
  started_at TEXT,
  ended_at TEXT,
  status TEXT CHECK (
    status IN (
      'planned','running','completed','failed','paused','cancelled','wedged','needs_review'
    )
  ),
  budget_used_json TEXT,
  summary TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS loop_items (
  id TEXT PRIMARY KEY,
  loop_run_id TEXT NOT NULL REFERENCES loop_runs(id),
  external_backlog_id TEXT,
  spec_uri TEXT,
  spec_hash TEXT,
  experiment_family TEXT,
  status TEXT CHECK (
    status IN (
      'pending','claimed','running','done','failed','skipped','reverted','needs_review','stale'
    )
  ),
  claimed_by TEXT,
  worker_session_uri TEXT,
  timeout_seconds INTEGER,
  claimed_at TEXT,
  started_at TEXT,
  ended_at TEXT,
  final_decision TEXT CHECK (
    final_decision IN ('kept','reverted','failed','skipped','needs_review','unknown')
  ),
  failure_reason TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS loop_iterations (
  id TEXT PRIMARY KEY,
  loop_item_id TEXT NOT NULL REFERENCES loop_items(id),
  loop_run_id TEXT NOT NULL REFERENCES loop_runs(id),
  loop_id TEXT NOT NULL REFERENCES loop_programs(id),
  hypothesis TEXT,
  mechanism TEXT,
  experiment_family TEXT,
  change_summary TEXT,
  changed_files_json TEXT,
  eval_command TEXT,
  guardrail_commands_json TEXT,
  verifier_command TEXT,
  verifier_passed INTEGER CHECK (verifier_passed IN (0,1)),
  baseline_value TEXT,
  result_value TEXT,
  delta_value TEXT,
  decision TEXT CHECK (decision IN ('kept','reverted','failed','needs_review','skipped')),
  lesson_summary TEXT,
  next_candidate TEXT,
  started_at TEXT,
  ended_at TEXT,
  duration_seconds INTEGER,
  tokens_used INTEGER,
  cost_estimate REAL,
  tool_profile TEXT,
  privacy_scope TEXT CHECK (
    privacy_scope IN ('private','workspace','project','public')
  ) DEFAULT 'workspace',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS loop_metrics (
  id TEXT PRIMARY KEY,
  loop_iteration_id TEXT REFERENCES loop_iterations(id),
  loop_run_id TEXT REFERENCES loop_runs(id),
  loop_id TEXT NOT NULL REFERENCES loop_programs(id),
  metric_name TEXT NOT NULL,
  metric_kind TEXT CHECK (
    metric_kind IN ('primary','guardrail','cost','latency','quality','safety','operational')
  ),
  direction TEXT CHECK (direction IN ('higher_is_better','lower_is_better','target','boolean')),
  unit TEXT,
  baseline_value REAL,
  result_value REAL,
  delta_value REAL,
  passed INTEGER CHECK (passed IN (0,1)),
  measured_at TEXT NOT NULL,
  evidence_uri TEXT,
  evidence_hash TEXT
);

CREATE TABLE IF NOT EXISTS loop_artifacts (
  id TEXT PRIMARY KEY,
  loop_iteration_id TEXT REFERENCES loop_iterations(id),
  loop_run_id TEXT REFERENCES loop_runs(id),
  loop_id TEXT NOT NULL REFERENCES loop_programs(id),
  kind TEXT CHECK (
    kind IN (
      'diff','patch','benchmark','eval','log','report','screenshot','verifier','model','config',
      'other'
    )
  ),
  uri TEXT NOT NULL,
  hash TEXT,
  size_bytes INTEGER,
  verifier_status TEXT CHECK (
    verifier_status IN ('unknown','passed','failed','not_required')
  ) DEFAULT 'unknown',
  privacy_scope TEXT CHECK (
    privacy_scope IN ('private','workspace','project','public')
  ) DEFAULT 'workspace',
  produced_at TEXT,
  ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS loop_tripwires (
  id TEXT PRIMARY KEY,
  loop_id TEXT REFERENCES loop_programs(id),
  loop_run_id TEXT REFERENCES loop_runs(id),
  loop_item_id TEXT REFERENCES loop_items(id),
  kind TEXT CHECK (
    kind IN (
      'stale_running_item',
      'heartbeat_starved',
      'no_ledger_writes',
      'verifier_failure_streak',
      'approval_gate_reached',
      'cost_spike',
      'regression_streak',
      'dangerous_action_attempt',
      'artifact_missing',
      'config_or_secret_risk',
      'other'
    )
  ),
  severity TEXT CHECK (severity IN ('info','warning','high','critical')),
  status TEXT CHECK (status IN ('open','acknowledged','resolved','ignored')) DEFAULT 'open',
  message TEXT,
  evidence_uri TEXT,
  opened_at TEXT NOT NULL,
  resolved_at TEXT,
  resolved_by TEXT
);

CREATE TABLE IF NOT EXISTS loop_candidate_links (
  id TEXT PRIMARY KEY,
  candidate_id TEXT NOT NULL REFERENCES candidates(id),
  loop_id TEXT REFERENCES loop_programs(id),
  loop_run_id TEXT REFERENCES loop_runs(id),
  loop_iteration_id TEXT REFERENCES loop_iterations(id),
  relation TEXT CHECK (
    relation IN ('derived_from','supports','contradicts','supersedes','invalidates')
  ),
  created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
  doc_id UNINDEXED,
  kind,
  title,
  body,
  path UNINDEXED
);
"""


@dataclass(frozen=True)
class EventInput:
    id: str
    source_type: str
    source_uri: str
    content_hash: str
    title: str
    summary: str
    body: str
    scope: str = "workspace"
    metadata: dict[str, Any] | None = None
    created_at: str | None = None


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def connect(path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    ensure_column(conn, "candidates", "claim_key", "TEXT NOT NULL DEFAULT ''")
    conn.commit()


def ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def upsert_event(conn: sqlite3.Connection, event: EventInput) -> bool:
    ingested_at = now_iso()
    try:
        conn.execute(
            """
            INSERT INTO events (
              id, source_type, source_uri, content_hash, title, summary, body,
              scope, metadata_json, created_at, ingested_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.source_type,
                event.source_uri,
                event.content_hash,
                event.title,
                event.summary,
                event.body,
                event.scope,
                json.dumps(event.metadata or {}, sort_keys=True),
                event.created_at,
                ingested_at,
            ),
        )
    except sqlite3.IntegrityError:
        return False

    conn.execute(
        "INSERT INTO search_index (doc_id, kind, title, body, path) VALUES (?, ?, ?, ?, ?)",
        (event.id, event.source_type, event.title, event.body, event.source_uri),
    )
    return True


def add_evidence(conn: sqlite3.Connection, event_id: str, evidence: Evidence, kind: str) -> str:
    evidence_id = stable_id(
        "evd",
        event_id,
        evidence.uri,
        evidence.excerpt,
        str(evidence.line_start),
        str(evidence.line_end),
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO evidence (
          id, event_id, kind, uri, excerpt, line_start, line_end, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            evidence_id,
            event_id,
            kind,
            evidence.uri,
            evidence.excerpt,
            evidence.line_start,
            evidence.line_end,
            now_iso(),
        ),
    )
    return evidence_id


def log_retrieval_use(
    conn: sqlite3.Connection,
    artifact_or_candidate_id: str,
    *,
    runtime: str | None = None,
    query: str | None = None,
    outcome: str | None = None,
    note: str | None = None,
) -> str:
    created_at = now_iso()
    sequence = conn.execute("SELECT COUNT(*) FROM retrieval_uses").fetchone()[0]
    retrieval_id = stable_id(
        "ret",
        artifact_or_candidate_id,
        runtime or "",
        query or "",
        outcome or "",
        note or "",
        created_at,
        str(sequence),
    )
    conn.execute(
        """
        INSERT INTO retrieval_uses (
          id, artifact_or_candidate_id, runtime, query, outcome, note, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            retrieval_id,
            artifact_or_candidate_id,
            runtime,
            query,
            outcome,
            note,
            created_at,
        ),
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


def insert_candidate(
    conn: sqlite3.Connection, candidate: Candidate, event_id: str | None = None
) -> str | None:
    candidate_id = stable_id(
        "cand",
        event_id or "",
        candidate.target.value,
        candidate.title,
        candidate.body,
    )
    timestamp = now_iso()
    try:
        conn.execute(
            """
            INSERT INTO candidates (
              id, event_id, target, title, body, confidence, scope, risk,
              claim_key, hints_json, evidence_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                event_id,
                candidate.target.value,
                candidate.title,
                candidate.body,
                candidate.confidence,
                candidate.scope.value,
                candidate.risk.value,
                candidate.claim_key or claim_key(f"{candidate.target.value} {candidate.body}"),
                json.dumps(candidate.hints, sort_keys=True),
                json.dumps([item.to_dict() for item in candidate.evidence], sort_keys=True),
                timestamp,
                timestamp,
            ),
        )
    except sqlite3.IntegrityError:
        return None
    return candidate_id


def backfill_candidate_claim_keys(conn: sqlite3.Connection, limit: int | None = None) -> int:
    sql = """
        SELECT id, target, body, evidence_json
        FROM candidates
        WHERE claim_key IS NULL OR claim_key = ''
        ORDER BY created_at ASC
    """
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    updated = 0
    for row in conn.execute(sql, params):
        evidence = json.loads(row["evidence_json"] or "[]")
        excerpt = ""
        for item in evidence:
            excerpt = item.get("excerpt", "")
            if excerpt:
                break
        key_source = excerpt or row["body"]
        key = claim_key(f"{row['target']} {key_source}")
        conn.execute(
            "UPDATE candidates SET claim_key = ?, updated_at = ? WHERE id = ?",
            (key, now_iso(), row["id"]),
        )
        updated += 1
    return updated


def mark_event_triaged(conn: sqlite3.Connection, event_id: str) -> None:
    conn.execute("UPDATE events SET triaged_at = ? WHERE id = ?", (now_iso(), event_id))


def iter_untriaged_events(
    conn: sqlite3.Connection, limit: int | None = None
) -> Iterable[sqlite3.Row]:
    sql = "SELECT * FROM events WHERE triaged_at IS NULL ORDER BY ingested_at ASC"
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    yield from conn.execute(sql, params)


def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    scopes: tuple[str, ...] | None = None,
) -> list[sqlite3.Row]:
    normalized_query = normalize_fts_query(query)
    if not normalized_query:
        return []
    scope_clause = ""
    params: list[Any] = [normalized_query]
    if scopes:
        placeholders = ",".join("?" for _ in scopes)
        scope_clause = f"AND events.scope IN ({placeholders})"
        params.extend(scopes)
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
              events.scope
            FROM search_index
            JOIN events ON events.id = search_index.doc_id
            WHERE search_index MATCH ?
            {scope_clause}
            ORDER BY rank
            LIMIT ?
            """,
            params,
        )
    )


def normalize_fts_query(query: str) -> str:
    terms = re.findall(r"[\w-]{2,}", query.lower())
    return " OR ".join(f'"{term}"' for term in terms[:8])


def list_candidates(
    conn: sqlite3.Connection,
    target: str | None = None,
    status: str | None = None,
    scope: str | None = None,
    limit: int = 20,
    *,
    statuses: tuple[str, ...] | None = None,
) -> list[sqlite3.Row]:
    if status and statuses:
        raise ValueError("pass either status or statuses, not both")
    clauses: list[str] = []
    params: list[Any] = []
    if target:
        clauses.append("target = ?")
        params.append(target)
    if status:
        clauses.append("status = ?")
        params.append(status)
    if statuses:
        placeholders = ",".join("?" for _ in statuses)
        clauses.append(f"status IN ({placeholders})")
        params.extend(statuses)
    if scope:
        clauses.append("scope = ?")
        params.append(scope)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    return list(
        conn.execute(
            f"""
            SELECT * FROM candidates
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        )
    )


def get_candidate(conn: sqlite3.Connection, candidate_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()


def transition_candidate(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    action: str,
    next_status: str,
    actor: str,
    reason: str,
) -> str:
    row = get_candidate(conn, candidate_id)
    if row is None:
        raise ValueError(f"candidate not found: {candidate_id}")
    previous_status = row["status"]
    created_at = now_iso()
    decision_id = stable_id(
        "dec",
        candidate_id,
        action,
        previous_status,
        next_status,
        actor,
        reason,
        created_at,
    )
    conn.execute(
        """
        INSERT INTO candidate_decisions (
          id, candidate_id, action, actor, reason, previous_status, next_status, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            decision_id,
            candidate_id,
            action,
            actor,
            reason,
            previous_status,
            next_status,
            created_at,
        ),
    )
    conn.execute(
        "UPDATE candidates SET status = ?, updated_at = ? WHERE id = ?",
        (next_status, created_at, candidate_id),
    )
    return decision_id


def list_candidate_decisions(conn: sqlite3.Connection, candidate_id: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT * FROM candidate_decisions
            WHERE candidate_id = ?
            ORDER BY created_at ASC
            """,
            (candidate_id,),
        )
    )


def review_groups(
    conn: sqlite3.Connection,
    *,
    status: str = "draft",
    target: str | None = None,
    limit: int = 20,
    include_low_value: bool = False,
) -> list[sqlite3.Row]:
    clauses = ["status = ?", "target != 'ignore'"]
    params: list[Any] = [status]
    if target:
        clauses.append("target = ?")
        params.append(target)
    params.append(max(limit * 20, limit))
    rows = list(
        conn.execute(
            f"""
            SELECT
              COALESCE(NULLIF(claim_key, ''), body) AS claim_key,
              target,
              COUNT(*) AS count,
              MIN(id) AS sample_candidate_id,
              MIN(title) AS sample_title,
              MIN(risk) AS risk,
              MIN(confidence) AS confidence
            FROM candidates
            WHERE {' AND '.join(clauses)}
            GROUP BY target, COALESCE(NULLIF(claim_key, ''), body)
            ORDER BY count DESC, sample_title ASC
            LIMIT ?
            """,
            params,
        )
    )
    if include_low_value:
        return rows[:limit]
    return [row for row in rows if not is_low_value_review_group(row["target"], row["claim_key"])][
        :limit
    ]


def review_group_candidates(
    conn: sqlite3.Connection,
    *,
    target: str,
    claim_key: str,
    status: str = "draft",
    limit: int | None = None,
) -> list[sqlite3.Row]:
    sql = """
        SELECT *
        FROM candidates
        WHERE status = ?
          AND target = ?
          AND COALESCE(NULLIF(claim_key, ''), body) = ?
        ORDER BY confidence DESC, created_at ASC, id ASC
    """
    params: list[Any] = [status, target, claim_key]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return list(conn.execute(sql, params))


def transition_candidate_group(
    conn: sqlite3.Connection,
    *,
    target: str,
    claim_key: str,
    status: str,
    action: str,
    next_status: str,
    actor: str,
    reason: str,
    limit: int | None = None,
) -> list[str]:
    rows = review_group_candidates(
        conn,
        target=target,
        claim_key=claim_key,
        status=status,
        limit=limit,
    )
    decision_ids: list[str] = []
    for row in rows:
        decision_ids.append(
            transition_candidate(
                conn,
                row["id"],
                action=action,
                next_status=next_status,
                actor=actor,
                reason=reason,
            )
        )
    return decision_ids


def is_low_value_review_group(target: str, claim_key: str) -> bool:
    normalized = re.sub(r"\s+", " ", claim_key.lower()).strip()
    if not normalized:
        return True
    target_prefix = f"{target} "
    body = normalized.removeprefix(target_prefix).strip()
    if body in {"", target, "status ok", "openclawbrain"}:
        return True
    if re.fullmatch(r"date \d{4} \d{2} \d{2}", body):
        return True
    if body in {"true", "false", "null", "ok"}:
        return True
    if len(body.split()) <= 1:
        return True
    return any(
        marker in body
        for marker in (
            "brain loaded runtime hook registered",
            "openclawbrain brain not yet loaded",
            "session key",
            "pagetype",
            "openclaw home",
        )
    )


def counts(conn: sqlite3.Connection) -> dict[str, Any]:
    event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    candidate_count = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    by_target = {
        row["target"]: row["count"]
        for row in conn.execute(
            "SELECT target, COUNT(*) AS count FROM candidates GROUP BY target ORDER BY target"
        )
    }
    return {"events": event_count, "candidates": candidate_count, "by_target": by_target}
