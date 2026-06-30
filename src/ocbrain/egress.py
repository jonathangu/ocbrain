from __future__ import annotations

import json
import sqlite3
from typing import Any

from ocbrain.db import now_iso
from ocbrain.events import canonical_json, sha256_text
from ocbrain.ids import stable_id
from ocbrain.scope import ScopeContext, ScopeTag, egress_allowed
from ocbrain.text import redact_secrets


def egress_preview(
    conn: sqlite3.Connection,
    *,
    context: ScopeContext,
    target: str,
    query: str | None = None,
    record: bool = False,
    sample_limit: int = 50,
) -> dict[str, Any]:
    included: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in evidence_events(conn):
        body = json.loads(row["body_json"])
        scope = ScopeTag.from_dict(body.get("scope"))
        allowed, reason = egress_allowed(scope, context, target)
        if query and query.lower() not in body.get("body", "").lower():
            allowed = False
            reason = "query_mismatch"
        item = {
            "event_id": row["id"],
            "evidence_id": body.get("evidence_id"),
            "scope": scope.to_dict(),
            "reason": reason,
        }
        if allowed:
            redacted = redact_secrets(str(body.get("body", "")))
            included.append(item | {"body": redacted})
        else:
            rejected.append(item)
    payload_text = "\n\n".join(item["body"] for item in included)
    payload_hash = sha256_text(payload_text)
    result = {
        "target": target,
        "context": context.to_dict(),
        "query": query,
        "included": included[:sample_limit],
        "rejected": rejected[:sample_limit],
        "included_count": len(included),
        "rejected_count": len(rejected),
        "items_sampled": len(included) > sample_limit or len(rejected) > sample_limit,
        "payload_hash": payload_hash,
    }
    if record:
        result["audit_id"] = record_egress_audit(conn, result)
    return result


def record_egress_audit(conn: sqlite3.Connection, result: dict[str, Any]) -> str:
    ts = now_iso()
    audit_id = stable_id(
        "egress",
        result["target"],
        canonical_json(result["context"]),
        result.get("query") or "",
        result["payload_hash"],
        ts,
    )
    conn.execute(
        """
        INSERT INTO egress_audits (
          id, ts, target, context_json, query, included_json, rejected_json, payload_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            audit_id,
            ts,
            result["target"],
            canonical_json(result["context"]),
            result.get("query"),
            canonical_json(result["included"]),
            canonical_json(result["rejected"]),
            result["payload_hash"],
        ),
    )
    return audit_id


def evidence_events(conn: sqlite3.Connection):
    yield from conn.execute(
        """
        SELECT *
        FROM brain_events
        WHERE kind = 'evidence_recorded'
        ORDER BY rowid ASC
        """
    )
