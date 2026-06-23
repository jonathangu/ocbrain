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
  knowledge_id TEXT REFERENCES knowledge(id),
  runtime TEXT,
  served_to_runtime TEXT,
  query TEXT,
  task_ref TEXT,
  affected_decision INTEGER,
  corrected INTEGER,
  outcome TEXT,
  note TEXT,
  served_at TEXT,
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
    migrate_legacy_evidence_table(conn)
    conn.executescript(SCHEMA)
    ensure_column(conn, "candidates", "claim_key", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "retrieval_uses", "knowledge_id", "TEXT")
    ensure_column(conn, "retrieval_uses", "served_to_runtime", "TEXT")
    ensure_column(conn, "retrieval_uses", "task_ref", "TEXT")
    ensure_column(conn, "retrieval_uses", "affected_decision", "INTEGER")
    ensure_column(conn, "retrieval_uses", "corrected", "INTEGER")
    ensure_column(conn, "retrieval_uses", "served_at", "TEXT")
    conn.commit()


def ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def migrate_legacy_evidence_table(conn: sqlite3.Connection) -> None:
    columns = table_columns(conn, "evidence")
    if columns and "claim" not in columns:
        conn.execute("ALTER TABLE evidence RENAME TO legacy_evidence")


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


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

    upsert_search_index(
        conn,
        event.id,
        event.source_type,
        event.title,
        event.body,
        event.source_uri,
    )
    return True


def add_evidence(conn: sqlite3.Connection, event_id: str, evidence: Evidence, kind: str) -> str:
    event = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    evidence_id = stable_id(
        "evd",
        evidence.uri,
        event["content_hash"] if event else event_id,
        evidence.excerpt,
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO evidence (
          id, source_type, source_runtime, source_uri, content_hash, claim,
          artifact_uri, artifact_hash, verifier_status, loop_tags, project,
          privacy_scope, occurred_at, ingested_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            evidence_id,
            kind,
            None,
            evidence.uri,
            event["content_hash"] if event else stable_id("hash", evidence.excerpt),
            evidence.excerpt,
            evidence.uri,
            None,
            "not_required",
            None,
            None,
            event["scope"] if event else "workspace",
            event["created_at"] if event else None,
            now_iso(),
        ),
    )
    return evidence_id


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
          claim = excluded.claim,
          artifact_uri = excluded.artifact_uri,
          artifact_hash = excluded.artifact_hash,
          verifier_status = excluded.verifier_status,
          loop_tags = excluded.loop_tags,
          project = excluded.project,
          privacy_scope = excluded.privacy_scope
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
        source_type,
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


def knowledge_search_body(**kwargs: Any) -> str:
    return " ".join(str(value) for value in kwargs.values() if value is not None)


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
          id, artifact_or_candidate_id, knowledge_id, runtime, served_to_runtime,
          query, outcome, note, served_at, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            retrieval_id,
            artifact_or_candidate_id,
            artifact_or_candidate_id if artifact_or_candidate_id.startswith("know_") else None,
            runtime,
            runtime,
            query,
            outcome,
            note,
            created_at,
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
        scope_clause = (
            "AND COALESCE(events.scope, knowledge.privacy_scope, evidence.privacy_scope) "
            f"IN ({placeholders})"
        )
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
              COALESCE(events.scope, knowledge.privacy_scope, evidence.privacy_scope) AS scope
            FROM search_index
            LEFT JOIN events ON events.id = search_index.doc_id
            LEFT JOIN knowledge ON knowledge.id = search_index.doc_id
            LEFT JOIN evidence ON evidence.id = search_index.doc_id
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


def get_knowledge(conn: sqlite3.Connection, knowledge_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM knowledge WHERE id = ?", (knowledge_id,)).fetchone()


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
    evidence_count = conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    knowledge_count = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
    by_target = {
        row["target"]: row["count"]
        for row in conn.execute(
            "SELECT target, COUNT(*) AS count FROM candidates GROUP BY target ORDER BY target"
        )
    }
    by_knowledge_type = {
        row["type"]: row["count"]
        for row in conn.execute(
            "SELECT type, COUNT(*) AS count FROM knowledge GROUP BY type ORDER BY type"
        )
    }
    return {
        "events": event_count,
        "candidates": candidate_count,
        "evidence": evidence_count,
        "knowledge": knowledge_count,
        "by_target": by_target,
        "by_knowledge_type": by_knowledge_type,
    }
