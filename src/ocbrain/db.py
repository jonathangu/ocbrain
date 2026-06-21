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
) -> list[sqlite3.Row]:
    clauses: list[str] = []
    params: list[Any] = []
    if target:
        clauses.append("target = ?")
        params.append(target)
    if status:
        clauses.append("status = ?")
        params.append(status)
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
) -> list[sqlite3.Row]:
    clauses = ["status = ?", "target != 'ignore'"]
    params: list[Any] = [status]
    if target:
        clauses.append("target = ?")
        params.append(target)
    params.append(limit)
    return list(
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
