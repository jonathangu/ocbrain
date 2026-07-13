from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any

from ocbrain.egress import egress_preview
from ocbrain.events import iter_events, propose_compilation
from ocbrain.ids import stable_id
from ocbrain.scope import ScopeContext, ScopeTag, scope_match
from ocbrain.text import compact_whitespace


@dataclass(frozen=True)
class DreamProposal:
    belief_id: str
    proposal_event_id: str
    scope: dict[str, Any]
    evidence_ids: list[str]
    body: str


def dream(
    conn: sqlite3.Connection,
    *,
    context: ScopeContext,
    since_ts: str | None = None,
    target: str = "local_model",
    record_egress: bool = False,
    limit: int = 20,
) -> dict[str, Any]:
    evidence = scoped_evidence(conn, context=context, since_ts=since_ts, limit=limit)
    grouped = group_by_scope(evidence)
    proposed: list[DreamProposal] = []
    conflicts: list[dict[str, Any]] = []
    for scope_id, items in grouped.items():
        scope = ScopeTag.from_dict(items[0]["scope"])
        evidence_ids = [item["evidence_id"] for item in items if item.get("evidence_id")]
        if not evidence_ids:
            continue
        body = dream_body(scope, items)
        belief_id = stable_id("belief", "dream", scope_id, *evidence_ids)
        try:
            proposal_id = propose_compilation(
                conn,
                belief_id=belief_id,
                body=body,
                evidence_ids=evidence_ids,
                scope=scope,
                confidence=0.45,
                teacher_model="local_dream_v0",
                teacher_rationale="Deterministic local consolidation over scoped evidence.",
                reward_band="moderate",
                writer="ocbrain-dream",
            )
        except PermissionError as exc:
            conflicts.append(
                {
                    "belief_id": belief_id,
                    "scope": scope.to_dict(),
                    "reason": str(exc),
                    "evidence_ids": evidence_ids,
                }
            )
            continue
        proposed.append(
            DreamProposal(
                belief_id=belief_id,
                proposal_event_id=proposal_id,
                scope=scope.to_dict(),
                evidence_ids=evidence_ids,
                body=body,
            )
        )
    egress = None
    if record_egress:
        egress = egress_preview(conn, context=context, target=target, record=True)
    return {
        "context": context.to_dict(),
        "since_ts": since_ts,
        "target": target,
        "evidence_seen": len(evidence),
        "proposed": [asdict(item) for item in proposed],
        "conflicts": conflicts,
        "egress": egress,
        "summary": {
            "evidence": len(evidence),
            "proposals": len(proposed),
            "conflicts": len(conflicts),
        },
    }


def scoped_evidence(
    conn: sqlite3.Connection,
    *,
    context: ScopeContext,
    since_ts: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in iter_events(conn):
        if event["kind"] != "evidence_recorded":
            continue
        if since_ts is not None and event["ts"] <= since_ts:
            continue
        body = json.loads(event["body_json"])
        scope = ScopeTag.from_dict(body.get("scope"))
        if scope_match(scope, context, cross_scope=False) == 0:
            continue
        rows.append(
            {
                "event_id": event["id"],
                "ts": event["ts"],
                "evidence_id": body.get("evidence_id"),
                "scope": scope.to_dict(),
                "body": str(body.get("body") or ""),
            }
        )
    return rows[:limit]


def group_by_scope(evidence: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in evidence:
        scope_id = str(item["scope"]["scope_id"])
        grouped.setdefault(scope_id, []).append(item)
    return grouped


def dream_body(scope: ScopeTag, evidence: list[dict[str, Any]]) -> str:
    snippets = [compact_whitespace(item["body"]) for item in evidence if item.get("body")]
    joined = " ".join(snippets)
    if len(joined) > 700:
        joined = joined[:697].rstrip() + "..."
    return f"Scoped consolidation for {scope.scope_id}: {joined}"
