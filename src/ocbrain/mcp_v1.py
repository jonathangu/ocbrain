"""MCP-facing operations for the event-authoritative v1 core.

This module is deliberately separate from the legacy compatibility dispatcher.
It never queries a legacy relational knowledge table or a companion store.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ocbrain.closeout import record_closeout
from ocbrain.core_v1 import (
    CORE_V1_SCHEMA_VERSION,
    append_core_event,
    canonical_json,
    get_core_v1_belief,
    get_core_v1_evidence,
    is_core_v1,
    now_iso,
    record_core_v1_evidence,
    record_core_v1_retrieval,
    resolve_object_id,
    search_core_v1,
    sha256_text,
)
from ocbrain.ids import stable_id
from ocbrain.scope import ScopeContext, ScopeTag, resolve_write_scope, scope_match
from ocbrain.shared_context import issue_source_handles

CONTEXT_SCHEMA_VERSION = "ocbrain.context.v1"
SOURCE_SCHEMA_VERSION = "ocbrain.source.v1"
DIGEST_SCHEMA_VERSION = "ocbrain.digest.v1"


def build_context_v1(
    conn: sqlite3.Connection,
    query: str,
    *,
    context: ScopeContext,
    limit: int,
    cross_scope: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_v1(conn)
    raw = search_core_v1(
        conn,
        query,
        context=context,
        limit=limit,
        cross_scope=cross_scope,
    )
    handles: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    for raw_item in raw["items"]:
        item_handles = _source_handles_for_belief(
            conn,
            str(raw_item["belief_id"]),
            context=context,
        )
        handles.extend(item_handles)
        if not item_handles:
            unavailable.append(
                {"object_id": str(raw_item["belief_id"]), "reason": "no_expandable_source"}
            )
        items.append(
            {
                "id": str(raw_item["belief_id"]),
                "kind": "core_v1",
                "excerpt": str(raw_item.get("body") or ""),
                "scope": dict(raw_item.get("scope") or {}),
                "score": float(raw_item.get("score") or 0.0),
                "relevance": float(raw_item.get("relevance") or 0.0),
                "confidence": float(raw_item.get("confidence") or 0.0),
                "confidence_band": str(raw_item.get("confidence_band") or "unknown"),
                "status": "current",
                "evidence_ids": [str(value) for value in raw_item.get("evidence_ids") or []],
                "sources": [_public_source_handle(value) for value in item_handles],
            }
        )
    handles = _dedupe_handles(handles)
    packet = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "core_schema": CORE_V1_SCHEMA_VERSION,
        "query": query,
        "resolved_context": context.to_dict(),
        "cross_scope": bool(cross_scope),
        "at_ts": None,
        "items": items,
        "contradictions": _visible_contradictions(items),
        "coverage": {
            "requested_limit": limit,
            "returned": len(items),
            "excluded_scope_count": int(raw.get("excluded_count") or 0),
            "excluded_sample": list(raw.get("excluded") or []),
            "estimated_tokens": _estimate_tokens(
                [str(item.get("excerpt") or "") for item in items]
            ),
            "source_handle_count": len(handles),
            "unavailable_sources": unavailable,
        },
    }
    return packet, handles


def record_context_v1(
    conn: sqlite3.Connection,
    packet: dict[str, Any],
    handles: list[dict[str, Any]],
    *,
    context: ScopeContext,
) -> str:
    retrieval_id = record_core_v1_retrieval(
        conn,
        query=str(packet["query"]),
        context=context.to_dict(),
        items=[
            {"belief_id": item["id"], "score": item["score"]}
            for item in packet["items"]
        ],
        runtime=context.runtime or "mcp",
        task_ref=context.task or f"brain.context:{packet['query']}",
        session_id=context.session,
        packet_schema=CONTEXT_SCHEMA_VERSION,
    )
    issue_source_handles(conn, handles, retrieval_use_id=retrieval_id)
    return retrieval_id


def expand_source_v1(
    conn: sqlite3.Connection,
    source_id: str,
    *,
    context: ScopeContext,
    max_chars: int,
) -> dict[str, Any]:
    _require_v1(conn)
    row = conn.execute(
        "SELECT * FROM context_source_handles WHERE id=?", (source_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"source handle not found: {source_id}")
    scope = ScopeTag.from_dict(json.loads(row["scope_json"]))
    if scope_match(scope, context) == 0:
        raise PermissionError("source scope does not match the supplied context")
    locator = json.loads(row["locator_json"])
    if row["source_kind"] == "core_v1_evidence":
        source = get_core_v1_evidence(conn, str(locator["evidence_id"]))
        if source is None:
            raise ValueError("issued evidence source no longer exists")
        content = str(source["body"])
    elif row["source_kind"] == "core_v1_belief":
        source = get_core_v1_belief(conn, str(locator["belief_id"]))
        if source is None:
            raise ValueError("issued belief source no longer exists")
        content = str(source["body"])
    else:
        raise ValueError(f"unsupported v1 source kind: {row['source_kind']}")
    actual_hash = sha256_text(content)
    if actual_hash != row["content_hash"]:
        raise ValueError("source changed after issuance; request a fresh brain.context handle")
    excerpt, truncated = _bounded_excerpt(content, max_chars=max_chars)
    issued_by = [
        str(item["retrieval_use_id"])
        for item in conn.execute(
            "SELECT retrieval_use_id FROM context_source_handle_issues "
            "WHERE source_id=? ORDER BY issued_at, retrieval_use_id",
            (source_id,),
        )
    ]
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "core_schema": CORE_V1_SCHEMA_VERSION,
        "id": str(row["id"]),
        "object_id": str(row["object_id"]),
        "kind": str(row["source_kind"]),
        "uri": row["uri"],
        "scope": scope.to_dict(),
        "content_hash": str(row["content_hash"]),
        "hash_verified": True,
        "content": excerpt,
        "truncated": truncated,
        "characters": len(excerpt),
        "issued_at": str(row["issued_at"]),
        "origin_retrieval_use_id": row["retrieval_use_id"],
        "issued_by_retrieval_use_ids": issued_by,
    }


def search_v1(
    conn: sqlite3.Connection,
    query: str,
    *,
    context: ScopeContext,
    limit: int,
    cross_scope: bool,
) -> dict[str, Any]:
    packet, _handles = build_context_v1(
        conn,
        query,
        context=context,
        limit=limit,
        cross_scope=cross_scope,
    )
    retrieval_id = record_core_v1_retrieval(
        conn,
        query=query,
        context=context.to_dict(),
        items=[
            {"belief_id": item["id"], "score": item["score"]}
            for item in packet["items"]
        ],
        runtime=context.runtime or "mcp",
        task_ref=context.task or f"brain.search:{query}",
        session_id=context.session,
        packet_schema="ocbrain.search.v1",
    )
    return {
        "schema_version": "ocbrain.search.v1",
        "query": query,
        "resolved_context": context.to_dict(),
        "items": packet["items"],
        "contradictions": packet["contradictions"],
        "coverage": packet["coverage"],
        "retrieval_use_id": retrieval_id,
        "retrieval_use_status": "recorded",
    }


def get_v1(
    conn: sqlite3.Connection,
    object_id: str,
    *,
    context: ScopeContext,
    include_candidate: bool = False,
    include_private: bool = False,
    cross_scope: bool = False,
) -> dict[str, Any]:
    belief = get_core_v1_belief(conn, object_id)
    if belief is not None:
        _authorize_get_scope(
            belief["scope"],
            context=context,
            include_private=include_private,
            cross_scope=cross_scope,
        )
        attributes = belief.get("attributes") or {}
        if attributes.get("quarantine_reason"):
            raise PermissionError("quarantined beliefs are not served by brain.get")
        if belief.get("status") != "current" or not belief.get("serve"):
            if not (include_candidate and belief.get("status") == "candidate"):
                raise PermissionError("non-current beliefs are not served by brain.get")
        return {"schema_version": "ocbrain.object.v1", "object_kind": "belief", **belief}
    evidence = get_core_v1_evidence(conn, object_id)
    if evidence is not None:
        _authorize_get_scope(
            evidence["scope"],
            context=context,
            include_private=include_private,
            cross_scope=cross_scope,
        )
        return {"schema_version": "ocbrain.object.v1", "object_kind": "evidence", **evidence}
    raise ValueError(f"object not found: {object_id}")


def _authorize_get_scope(
    raw_scope: dict[str, Any],
    *,
    context: ScopeContext,
    include_private: bool,
    cross_scope: bool,
) -> None:
    scope = ScopeTag.from_dict(raw_scope)
    if scope_match(scope, context, cross_scope=cross_scope) == 0:
        raise PermissionError("object scope does not match the supplied context")
    if scope.confidential and not include_private:
        raise PermissionError("confidential objects require explicit include_private")


def digest_v1(
    conn: sqlite3.Connection,
    *,
    context: ScopeContext,
    limit: int,
) -> dict[str, Any]:
    _require_v1(conn)
    counts = {
        name: int(conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
        for name in (
            "brain_events",
            "evidence_objects",
            "current_beliefs",
            "belief_evidence",
            "retrieval_uses",
            "task_closeouts",
        )
    }
    rows = conn.execute(
        "SELECT * FROM current_beliefs WHERE serve=1 AND status='current' "
        "ORDER BY pinned DESC, last_compiled_at DESC, belief_id LIMIT ?",
        (max(limit * 8, 40),),
    )
    current: list[dict[str, Any]] = []
    excluded = 0
    for row in rows:
        scope = ScopeTag(
            str(row["scope_type"]),
            str(row["scope_id"]),
            visibility=str(row["visibility"]),
            egress_policy=str(row["egress_policy"]),
            provenance=str(row["scope_provenance"]),
        )
        if scope_match(scope, context) == 0:
            excluded += 1
            continue
        current.append(
            {
                "id": str(row["belief_id"]),
                "body": str(row["body"]),
                "scope": scope.to_dict(),
                "confidence": row["confidence"],
                "evidence_ids": json.loads(row["evidence_ids"]),
            }
        )
        if len(current) >= limit:
            break
    return {
        "schema_version": DIGEST_SCHEMA_VERSION,
        "core_schema": CORE_V1_SCHEMA_VERSION,
        "resolved_context": context.to_dict(),
        "counts": counts,
        "current": current,
        "excluded_scope_count": excluded,
    }


def feedback_v1(
    conn: sqlite3.Connection,
    retrieval_use_id: str,
    *,
    outcome: str,
    note: str | None,
) -> dict[str, Any]:
    allowed = {"helpful", "used", "irrelevant", "ignored", "harmful"}
    if outcome not in allowed:
        raise ValueError(f"outcome must be one of: {', '.join(sorted(allowed))}")
    updated = conn.execute(
        "UPDATE retrieval_uses SET outcome=?, note=COALESCE(?, note), "
        "feedback_source='runtime_explicit', feedback_at=? WHERE id=?",
        (outcome, note, now_iso(), retrieval_use_id),
    )
    if updated.rowcount == 0:
        raise ValueError(f"retrieval use not found: {retrieval_use_id}")
    return {"retrieval_use_id": retrieval_use_id, "outcome": outcome}


def ingest_v1(
    conn: sqlite3.Connection,
    *,
    body: str,
    kind: str,
    context: ScopeContext,
    writer: str,
    session_id: str | None,
    artifact_ref: str | None,
) -> dict[str, Any]:
    evidence_id, event_id = record_core_v1_evidence(
        conn,
        body=body,
        kind=kind,
        scope=resolve_write_scope(context),
        writer=writer,
        session_id=session_id,
        artifact_ref=artifact_ref,
    )
    return {
        "event_id": event_id,
        "evidence_id": evidence_id,
        "kind": "evidence_recorded",
    }


def closeout_v1(
    conn: sqlite3.Connection,
    *,
    task_ref: str,
    status: str,
    summary: str,
    context: ScopeContext,
    retrieval_use_ids: list[str],
    decision_impact: str,
    decision_note: str | None,
    artifact_refs: list[dict[str, Any]],
    verifier_refs: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    awaiting: str | None,
    actor: str,
) -> dict[str, Any]:
    return record_closeout(
        conn,
        task_ref=task_ref,
        status=status,
        summary=summary,
        context=context,
        retrieval_use_ids=retrieval_use_ids,
        decision_impact=decision_impact,
        decision_note=decision_note,
        artifact_refs=artifact_refs,
        verifier_refs=verifier_refs,
        actions=actions,
        outcomes=outcomes,
        awaiting=awaiting,
        actor=actor,
    )


def correct_v1(
    conn: sqlite3.Connection,
    *,
    layer: str,
    target: str,
    op: str,
    body: str | None,
    actor: str,
    hard: bool,
) -> dict[str, Any]:
    if layer not in {"evidence", "knowledge", "belief"}:
        raise ValueError("layer must be evidence, knowledge, or belief")
    event_id = append_core_event(
        conn,
        "correction_recorded",
        {
            "schema_version": "ocbrain.correction.v1",
            "subject": {"kind": layer, "id": resolve_object_id(conn, target)},
            "target_layer": layer,
            "target_id": target,
            "op": op,
            "body": body,
            "author": actor,
            "hard": bool(hard),
        },
        writer=actor,
        project=True,
    )
    return {"event_id": event_id, "kind": "correction_recorded"}


def forget_v1(
    conn: sqlite3.Connection,
    *,
    target: str,
    mode: str,
    reason: str | None,
    actor: str,
) -> dict[str, Any]:
    if mode not in {"soft", "shred"}:
        raise ValueError("mode must be soft or shred")
    event_id = append_core_event(
        conn,
        "tombstone_recorded",
        {
            "schema_version": "ocbrain.tombstone.v1",
            "subject": {"kind": "belief", "id": resolve_object_id(conn, target)},
            "target": target,
            "target_hash": sha256_text(target),
            "mode": mode,
            "reason": reason,
            "approved_by": actor,
        },
        writer=actor,
        project=True,
    )
    return {"event_id": event_id, "kind": "tombstone_recorded"}


def proposals_v1(
    conn: sqlite3.Connection,
    *,
    limit: int,
    include_decided: bool,
) -> dict[str, Any]:
    decided = {
        str(json.loads(row["body_json"]).get("proposal_event_id"))
        for row in conn.execute(
            "SELECT body_json FROM brain_events WHERE kind='compilation_decided'"
        )
    }
    result: list[dict[str, Any]] = []
    for row in conn.execute(
        "SELECT * FROM brain_events WHERE kind='compilation_proposed' "
        "ORDER BY rowid DESC LIMIT ?",
        (max(limit * 4, 100),),
    ):
        is_decided = str(row["id"]) in decided
        if is_decided and not include_decided:
            continue
        result.append(
            {
                "proposal_event_id": str(row["id"]),
                "ts": str(row["ts"]),
                "decided": is_decided,
                **json.loads(row["body_json"]),
            }
        )
        if len(result) >= limit:
            break
    return {"schema_version": "ocbrain.proposals.v1", "proposals": result}


def decide_proposal_v1(
    conn: sqlite3.Connection,
    *,
    proposal_event_id: str,
    decision: str,
    actor: str,
    edited_body: str | None,
    reason: str | None,
) -> dict[str, Any]:
    if decision not in {"approve", "reject", "edit", "shadow"}:
        raise ValueError("decision must be approve, reject, edit, or shadow")
    proposal = conn.execute(
        "SELECT 1 FROM brain_events WHERE id=? AND kind='compilation_proposed'",
        (proposal_event_id,),
    ).fetchone()
    if proposal is None:
        raise ValueError(f"proposal not found: {proposal_event_id}")
    existing = conn.execute(
        "SELECT 1 FROM brain_events WHERE kind='compilation_decided' "
        "AND json_extract(body_json, '$.proposal_event_id')=?",
        (proposal_event_id,),
    ).fetchone()
    if existing is not None:
        raise ValueError(f"proposal already decided: {proposal_event_id}")
    event_id = append_core_event(
        conn,
        "compilation_decided",
        {
            "schema_version": "ocbrain.compilation-decision.v1",
            "subject": {"kind": "proposal", "id": proposal_event_id},
            "proposal_event_id": proposal_event_id,
            "decision": decision,
            "actor": actor,
            "edited_body": edited_body,
            "reason": reason,
        },
        writer=actor,
        project=True,
    )
    return {"event_id": event_id, "kind": "compilation_decided", "decision": decision}


def _source_handles_for_belief(
    conn: sqlite3.Connection,
    belief_id: str,
    *,
    context: ScopeContext,
) -> list[dict[str, Any]]:
    canonical_id = resolve_object_id(conn, belief_id)
    handles: list[dict[str, Any]] = []
    rows = conn.execute(
        "SELECT eo.* FROM belief_evidence be "
        "JOIN evidence_objects eo ON eo.evidence_id=be.evidence_id "
        "WHERE be.belief_id=? ORDER BY be.created_at, eo.evidence_id",
        (canonical_id,),
    )
    for row in rows:
        scope = {
            "scope_type": row["scope_type"],
            "scope_id": row["scope_id"],
            "visibility": row["visibility"],
            "egress_policy": row["egress_policy"],
            "provenance": row["scope_provenance"],
        }
        if scope_match(ScopeTag.from_dict(scope), context) == 0:
            continue
        content = str(row["body"])
        handles.append(
            _make_source_handle(
                object_id=canonical_id,
                source_kind="core_v1_evidence",
                uri=row["source_uri"] or row["artifact_uri"] or f"ocbrain://evidence/{row['evidence_id']}",
                content_hash=sha256_text(content),
                scope=scope,
                locator={"evidence_id": str(row["evidence_id"])},
            )
        )
    if handles:
        return _dedupe_handles(handles)
    belief = get_core_v1_belief(conn, canonical_id)
    if belief is None:
        return []
    scope = dict(belief["scope"])
    if scope_match(ScopeTag.from_dict(scope), context) == 0:
        return []
    return [
        _make_source_handle(
            object_id=canonical_id,
            source_kind="core_v1_belief",
            uri=f"ocbrain://belief/{canonical_id}",
            content_hash=sha256_text(str(belief["body"])),
            scope=scope,
            locator={"belief_id": canonical_id},
        )
    ]


def _make_source_handle(
    *,
    object_id: str,
    source_kind: str,
    uri: str | None,
    content_hash: str,
    scope: dict[str, Any],
    locator: dict[str, Any],
) -> dict[str, Any]:
    source_id = stable_id(
        "src",
        object_id,
        source_kind,
        uri or "",
        content_hash,
        canonical_json(scope),
    )
    return {
        "id": source_id,
        "object_id": object_id,
        "source_kind": source_kind,
        "uri": uri,
        "content_hash": content_hash,
        "scope": scope,
        "locator": locator,
    }


def _public_source_handle(handle: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": handle["id"],
        "kind": handle["source_kind"],
        "uri": handle.get("uri"),
        "content_hash": handle["content_hash"],
    }


def _dedupe_handles(handles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list({str(handle["id"]): handle for handle in handles}.values())


def _bounded_excerpt(content: str, *, max_chars: int) -> tuple[str, bool]:
    if len(content) <= max_chars:
        return content, False
    return content[:max_chars], True


def _estimate_tokens(values: list[str]) -> int:
    return sum(max(len(value) // 4, 1) for value in values)


def _visible_contradictions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose obvious negation conflicts inside the bounded packet."""
    result: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        left_terms = _terms(str(item.get("excerpt") or ""))
        for other in items[index + 1 :]:
            right_terms = _terms(str(other.get("excerpt") or ""))
            shared = sorted(left_terms & right_terms)
            if len(shared) < 2:
                continue
            if _has_negation(str(item.get("excerpt") or "")) == _has_negation(
                str(other.get("excerpt") or "")
            ):
                continue
            result.append(
                {
                    "belief_id": item["id"],
                    "other_belief_id": other["id"],
                    "reasons": ["negation_mismatch"],
                    "shared_terms": shared[:12],
                    "body": item["excerpt"],
                    "other_body": other["excerpt"],
                }
            )
    return result[:12]


def _terms(value: str) -> set[str]:
    import re

    return set(re.findall(r"[\w-]{2,}", value.lower()))


def _has_negation(value: str) -> bool:
    return bool(_terms(value) & {"no", "not", "never", "without", "cannot"})


def _require_v1(conn: sqlite3.Connection) -> None:
    if not is_core_v1(conn):
        raise ValueError("operation requires an OCBrain v1 core")


__all__ = [
    "build_context_v1",
    "closeout_v1",
    "correct_v1",
    "decide_proposal_v1",
    "digest_v1",
    "expand_source_v1",
    "feedback_v1",
    "forget_v1",
    "get_v1",
    "ingest_v1",
    "proposals_v1",
    "record_context_v1",
    "search_v1",
]
