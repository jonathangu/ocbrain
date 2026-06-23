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


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def connect(path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


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
