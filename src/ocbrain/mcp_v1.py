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
    compilation_block_reason,
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
from ocbrain.scope import (
    HOSTED_MODEL_TARGET,
    LOCAL_MODEL_TARGET,
    ScopeContext,
    ScopeTag,
    egress_allowed,
    normalize_delivery_target,
    resolve_write_scope,
)
from ocbrain.shared_context import issue_source_handles

CONTEXT_SCHEMA_VERSION = "ocbrain.context.v1"
SOURCE_SCHEMA_VERSION = "ocbrain.source.v1"
DIGEST_SCHEMA_VERSION = "ocbrain.digest.v1"
MAX_CONTEXT_PACKET_BYTES = 32_000
MAX_CONTEXT_QUERY_CHARS = 4_000
MAX_ITEM_EXCERPT_CHARS = 1_600
MAX_ITEM_SOURCE_HANDLES = 3
RETRIEVAL_ID_PLACEHOLDER = "ret_0000000000000000"


def build_context_v1(
    conn: sqlite3.Connection,
    query: str,
    *,
    context: ScopeContext,
    limit: int,
    cross_scope: bool,
    delivery_target: str = LOCAL_MODEL_TARGET,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_v1(conn)
    delivery_target = normalize_delivery_target(delivery_target)
    raw = search_core_v1(
        conn,
        query,
        context=context,
        limit=limit,
        cross_scope=cross_scope,
        delivery_target=delivery_target,
    )
    handles: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    delivery_excluded = 0
    for raw_item in raw["items"]:
        if not _scope_allowed_for_delivery(
            raw_item.get("scope"),
            context=context,
            delivery_target=delivery_target,
            cross_scope=cross_scope,
        ):
            delivery_excluded += 1
            continue
        item_handles = _source_handles_for_belief(
            conn,
            str(raw_item["belief_id"]),
            context=context,
            delivery_target=delivery_target,
            cross_scope=cross_scope,
        )
        item_handles = item_handles[:MAX_ITEM_SOURCE_HANDLES]
        handles.extend(item_handles)
        if not item_handles:
            unavailable.append(
                {"object_id": str(raw_item["belief_id"]), "reason": "no_expandable_source"}
            )
        excerpt, excerpt_truncated = _bounded_excerpt(
            str(raw_item.get("body") or ""), max_chars=MAX_ITEM_EXCERPT_CHARS
        )
        items.append(
            {
                "id": str(raw_item["belief_id"]),
                "kind": "core_v1",
                "excerpt": excerpt,
                "excerpt_truncated": excerpt_truncated,
                "scope": dict(raw_item.get("scope") or {}),
                "score": float(raw_item.get("score") or 0.0),
                "relevance": float(raw_item.get("relevance") or 0.0),
                "confidence": float(raw_item.get("confidence") or 0.0),
                "confidence_band": str(raw_item.get("confidence_band") or "unknown"),
                "status": "current",
                "evidence_ids": _evidence_ids_for_delivery(
                    conn,
                    raw_item.get("evidence_ids") or [],
                    context=context,
                    delivery_target=delivery_target,
                    cross_scope=cross_scope,
                ),
                "sources": [_public_source_handle(value) for value in item_handles],
                "ranking": dict(raw_item.get("ranking") or {}),
            }
        )
    handles = _dedupe_handles(handles)
    packet = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "core_schema": CORE_V1_SCHEMA_VERSION,
        "delivery_target": delivery_target,
        "query": query[:MAX_CONTEXT_QUERY_CHARS],
        "resolved_context": context.to_dict(),
        "cross_scope": bool(cross_scope),
        "at_ts": None,
        "items": items,
        "contradictions": _explicit_contradictions(conn, items),
        "coverage": {
            "requested_limit": limit,
            "returned": len(items),
            "excluded_scope_count": int(raw.get("excluded_count") or 0),
            "excluded_delivery_count": (
                int(raw.get("delivery_excluded_count") or 0) + delivery_excluded
            ),
            "exclusion_count_basis": str(
                raw.get("exclusion_count_basis") or "current_serving_inventory"
            ),
            "excluded_sample": (
                [] if delivery_target != LOCAL_MODEL_TARGET else list(raw.get("excluded") or [])
            ),
            "estimated_tokens": 0,
            "serialized_bytes": 0,
            "hard_packet_limit_bytes": MAX_CONTEXT_PACKET_BYTES,
            "source_handle_count": len(handles),
            "unavailable_sources": unavailable,
            "ranking": dict(raw.get("ranking") or {}),
        },
    }
    return _enforce_context_packet_limit(packet, handles)


def record_context_v1(
    conn: sqlite3.Connection,
    packet: dict[str, Any],
    handles: list[dict[str, Any]],
    *,
    context: ScopeContext,
    delivery_target: str = LOCAL_MODEL_TARGET,
) -> str:
    delivery_target = normalize_delivery_target(delivery_target)
    retrieval_id = record_core_v1_retrieval(
        conn,
        query=str(packet["query"]),
        context={**context.to_dict(), "delivery_target": delivery_target},
        items=[{"belief_id": item["id"], "score": item["score"]} for item in packet["items"]],
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
    delivery_target: str = LOCAL_MODEL_TARGET,
) -> dict[str, Any]:
    _require_v1(conn)
    delivery_target = normalize_delivery_target(delivery_target)
    row = conn.execute("SELECT * FROM context_source_handles WHERE id=?", (source_id,)).fetchone()
    if row is None:
        raise ValueError(f"source handle not found: {source_id}")
    scope = ScopeTag.from_dict(json.loads(row["scope_json"]))
    _authorize_delivery_scope(
        scope,
        context=context,
        delivery_target=delivery_target,
        scope_error="source scope does not match the supplied context",
    )
    locator = json.loads(row["locator_json"])
    if row["source_kind"] == "core_v1_evidence":
        source = get_core_v1_evidence(conn, str(locator["evidence_id"]))
        if source is None:
            raise ValueError("issued evidence source no longer exists")
        belief = get_core_v1_belief(conn, str(row["object_id"]))
        if belief is None or belief.get("status") != "current" or not belief.get("serve"):
            raise PermissionError("issued source is no longer linked to a current belief")
        _authorize_delivery_scope(
            ScopeTag.from_dict(belief.get("scope")),
            context=context,
            delivery_target=delivery_target,
            scope_error="source belief scope no longer matches the supplied context",
        )
        _authorize_delivery_scope(
            ScopeTag.from_dict(source.get("scope")),
            context=context,
            delivery_target=delivery_target,
            scope_error="source evidence scope no longer matches the supplied context",
        )
        linked = conn.execute(
            "SELECT 1 FROM belief_evidence WHERE belief_id=? AND evidence_id=? "
            "AND relation='supports'",
            (belief["canonical_id"], source["canonical_id"]),
        ).fetchone()
        if linked is None:
            raise PermissionError("issued evidence is no longer current support for this belief")
        content = str(source["body"])
    elif row["source_kind"] == "core_v1_belief":
        source = get_core_v1_belief(conn, str(locator["belief_id"]))
        if source is None:
            raise ValueError("issued belief source no longer exists")
        if source.get("status") != "current" or not source.get("serve"):
            raise PermissionError("issued belief source is no longer current")
        _authorize_delivery_scope(
            ScopeTag.from_dict(source.get("scope")),
            context=context,
            delivery_target=delivery_target,
            scope_error="source belief scope no longer matches the supplied context",
        )
        content = str(source["body"])
    else:
        raise ValueError(f"unsupported v1 source kind: {row['source_kind']}")
    actual_hash = sha256_text(content)
    if actual_hash != row["content_hash"]:
        raise ValueError("source changed after issuance; request a fresh brain.context handle")
    excerpt, truncated = _bounded_excerpt(content, max_chars=max_chars)
    issued_by_count = int(
        conn.execute(
            "SELECT COUNT(*) FROM context_source_handle_issues WHERE source_id=?",
            (source_id,),
        ).fetchone()[0]
    )
    issued_by = [
        str(item["retrieval_use_id"])
        for item in conn.execute(
            "SELECT retrieval_use_id FROM context_source_handle_issues "
            "WHERE source_id=? ORDER BY issued_at DESC, retrieval_use_id DESC LIMIT 8",
            (source_id,),
        )
    ]
    uri = row["uri"]
    if delivery_target == HOSTED_MODEL_TARGET:
        if row["source_kind"] == "core_v1_evidence":
            uri = f"ocbrain://evidence/{locator['evidence_id']}"
        else:
            uri = f"ocbrain://belief/{locator['belief_id']}"
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "core_schema": CORE_V1_SCHEMA_VERSION,
        "delivery_target": delivery_target,
        "id": str(row["id"]),
        "object_id": str(row["object_id"]),
        "kind": str(row["source_kind"]),
        "uri": uri,
        "scope": scope.to_dict(),
        "content_hash": str(row["content_hash"]),
        "hash_verified": True,
        "content": excerpt,
        "truncated": truncated,
        "characters": len(excerpt),
        "issued_at": str(row["issued_at"]),
        "origin_retrieval_use_id": row["retrieval_use_id"],
        "issued_by_count": issued_by_count,
        "issued_by_retrieval_use_ids": issued_by,
    }


def search_v1(
    conn: sqlite3.Connection,
    query: str,
    *,
    context: ScopeContext,
    limit: int,
    cross_scope: bool,
    delivery_target: str = LOCAL_MODEL_TARGET,
) -> dict[str, Any]:
    packet, handles = build_context_v1(
        conn,
        query,
        context=context,
        limit=limit,
        cross_scope=cross_scope,
        delivery_target=delivery_target,
    )
    payload = {
        "schema_version": "ocbrain.search.v1",
        "delivery_target": packet["delivery_target"],
        "query": packet["query"],
        "resolved_context": context.to_dict(),
        "items": packet["items"],
        "contradictions": packet["contradictions"],
        "coverage": packet["coverage"],
    }
    payload, handles = prepare_retrieval_packet_v1(payload, handles)
    retrieval_id = record_core_v1_retrieval(
        conn,
        query=str(payload["query"]),
        context={**context.to_dict(), "delivery_target": payload["delivery_target"]},
        items=[{"belief_id": item["id"], "score": item["score"]} for item in payload["items"]],
        runtime=context.runtime or "mcp",
        task_ref=context.task or f"brain.search:{payload['query']}",
        session_id=context.session,
        packet_schema="ocbrain.search.v1",
    )
    issue_source_handles(conn, handles, retrieval_use_id=retrieval_id)
    bind_retrieval_id_v1(payload, retrieval_id)
    return payload


def get_v1(
    conn: sqlite3.Connection,
    object_id: str,
    *,
    context: ScopeContext,
    include_candidate: bool = False,
    include_private: bool = False,
    cross_scope: bool = False,
    delivery_target: str = LOCAL_MODEL_TARGET,
) -> dict[str, Any]:
    delivery_target = normalize_delivery_target(delivery_target)
    belief = get_core_v1_belief(conn, object_id)
    if belief is not None:
        _authorize_get_scope(
            belief["scope"],
            context=context,
            include_private=include_private,
            cross_scope=cross_scope,
            delivery_target=delivery_target,
        )
        attributes = belief.get("attributes") or {}
        if attributes.get("quarantine_reason"):
            raise PermissionError("quarantined beliefs are not served by brain.get")
        if belief.get("status") != "current" or not belief.get("serve"):
            if not (include_candidate and belief.get("status") == "candidate"):
                raise PermissionError("non-current beliefs are not served by brain.get")
        public_belief = _belief_for_delivery(belief, delivery_target=delivery_target)
        public_belief["evidence_ids"] = _evidence_ids_for_delivery(
            conn,
            belief.get("evidence_ids") or [],
            context=context,
            delivery_target=delivery_target,
            cross_scope=cross_scope,
        )
        return {
            "schema_version": "ocbrain.object.v1",
            "delivery_target": delivery_target,
            "object_kind": "belief",
            **public_belief,
        }
    evidence = get_core_v1_evidence(conn, object_id)
    if evidence is not None:
        _authorize_get_scope(
            evidence["scope"],
            context=context,
            include_private=include_private,
            cross_scope=cross_scope,
            delivery_target=delivery_target,
        )
        return {
            "schema_version": "ocbrain.object.v1",
            "delivery_target": delivery_target,
            "object_kind": "evidence",
            **_evidence_for_delivery(evidence, delivery_target=delivery_target),
        }
    raise ValueError(f"object not found: {object_id}")


def _authorize_get_scope(
    raw_scope: dict[str, Any],
    *,
    context: ScopeContext,
    include_private: bool,
    cross_scope: bool,
    delivery_target: str,
) -> None:
    scope = ScopeTag.from_dict(raw_scope)
    _authorize_delivery_scope(
        scope,
        context=context,
        delivery_target=delivery_target,
        cross_scope=cross_scope,
        scope_error="object scope does not match the supplied context",
    )
    if scope.confidential and not include_private:
        raise PermissionError("confidential objects require explicit include_private")


def _belief_for_delivery(belief: dict[str, Any], *, delivery_target: str) -> dict[str, Any]:
    if delivery_target == LOCAL_MODEL_TARGET:
        return dict(belief)
    attributes = belief.get("attributes") or {}
    safe_attribute_keys = {
        "title",
        "curated",
        "manifest_schema",
        "curation_sha256",
        "source_quality",
        "lifecycle",
        "content_sha256",
        "contradicts",
        "contradiction_ids",
    }
    safe_attributes = {key: attributes[key] for key in safe_attribute_keys if key in attributes}
    attestations = attributes.get("source_attestations")
    if isinstance(attestations, list):
        safe_attributes["source_attestations"] = [
            {key: value[key] for key in ("ref", "sha256") if key in value}
            for value in attestations
            if isinstance(value, dict)
        ]
    keys = {
        "requested_id",
        "canonical_id",
        "belief_id",
        "body",
        "belief_type",
        "scope",
        "confidence",
        "confidence_band",
        "status",
        "serve",
        "pinned",
        "last_compiled_at",
    }
    return {
        **{key: belief[key] for key in keys if key in belief},
        "attributes": safe_attributes,
    }


def _evidence_for_delivery(evidence: dict[str, Any], *, delivery_target: str) -> dict[str, Any]:
    if delivery_target == LOCAL_MODEL_TARGET:
        return dict(evidence)
    keys = {
        "requested_id",
        "canonical_id",
        "evidence_id",
        "body",
        "kind",
        "content_hash",
        "source_content_hash",
        "verifier_status",
        "occurred_at",
        "recorded_at",
        "scope",
    }
    return {key: evidence[key] for key in keys if key in evidence}


def digest_v1(
    conn: sqlite3.Connection,
    *,
    context: ScopeContext,
    limit: int,
    delivery_target: str = LOCAL_MODEL_TARGET,
) -> dict[str, Any]:
    _require_v1(conn)
    delivery_target = normalize_delivery_target(delivery_target)
    if delivery_target == LOCAL_MODEL_TARGET:
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
    else:
        compatible = sorted(context.compatible_scope_ids())
        placeholders = ",".join("?" for _ in compatible)
        counts = {
            "eligible_current_beliefs": int(
                conn.execute(
                    f"SELECT COUNT(*) FROM current_beliefs WHERE serve=1 "
                    f"AND status='current' AND egress_policy='hosted_ok' "
                    f"AND visibility NOT IN ('confidential','secret') "
                    f"AND scope_id IN ({placeholders})",  # noqa: S608
                    compatible,
                ).fetchone()[0]
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
        allowed, _reason = egress_allowed(scope, context, delivery_target)
        if not allowed:
            excluded += 1
            continue
        current.append(
            {
                "id": str(row["belief_id"]),
                "body": _bounded_excerpt(str(row["body"]), max_chars=MAX_ITEM_EXCERPT_CHARS)[0],
                "scope": scope.to_dict(),
                "confidence": row["confidence"],
                "evidence_ids": _evidence_ids_for_delivery(
                    conn,
                    json.loads(row["evidence_ids"]),
                    context=context,
                    delivery_target=delivery_target,
                ),
            }
        )
        if len(current) >= limit:
            break
    return {
        "schema_version": DIGEST_SCHEMA_VERSION,
        "core_schema": CORE_V1_SCHEMA_VERSION,
        "delivery_target": delivery_target,
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
    if layer not in {"knowledge", "belief"}:
        raise ValueError("layer must be knowledge or belief; evidence corrections are unsupported")
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
        "SELECT * FROM brain_events WHERE kind='compilation_proposed' ORDER BY rowid DESC LIMIT ?",
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
        "SELECT event_seq, body_json FROM brain_events WHERE id=? AND kind='compilation_proposed'",
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
    if decision in {"approve", "edit"}:
        proposal_body = json.loads(proposal["body_json"])
        belief_id = str(proposal_body.get("belief_id") or "")
        reason_blocked = compilation_block_reason(
            conn,
            belief_id,
            proposal_event_seq=int(proposal["event_seq"]),
        )
        if reason_blocked is not None:
            raise PermissionError(f"cannot {decision}: belief is {reason_blocked}: {belief_id}")
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


def _scope_allowed_for_delivery(
    raw_scope: dict[str, Any] | None,
    *,
    context: ScopeContext,
    delivery_target: str,
    cross_scope: bool = False,
) -> bool:
    allowed, _reason = egress_allowed(
        ScopeTag.from_dict(raw_scope),
        context,
        delivery_target,
        cross_scope=cross_scope,
    )
    return allowed


def _evidence_ids_for_delivery(
    conn: sqlite3.Connection,
    evidence_ids: list[Any],
    *,
    context: ScopeContext,
    delivery_target: str,
    cross_scope: bool = False,
) -> list[str]:
    values = [str(value) for value in evidence_ids]
    if delivery_target == LOCAL_MODEL_TARGET:
        return values
    result: list[str] = []
    for evidence_id in values:
        evidence = get_core_v1_evidence(conn, evidence_id)
        if evidence is None:
            continue
        if _scope_allowed_for_delivery(
            evidence.get("scope"),
            context=context,
            delivery_target=delivery_target,
            cross_scope=cross_scope,
        ):
            result.append(evidence_id)
    return result


def _authorize_delivery_scope(
    scope: ScopeTag,
    *,
    context: ScopeContext,
    delivery_target: str,
    scope_error: str,
    cross_scope: bool = False,
) -> None:
    allowed, reason = egress_allowed(
        scope,
        context,
        delivery_target,
        cross_scope=cross_scope,
    )
    if allowed:
        return
    if reason == "scope_mismatch":
        raise PermissionError(scope_error)
    raise PermissionError(f"object is not eligible for {delivery_target} delivery ({reason})")


def _source_handles_for_belief(
    conn: sqlite3.Connection,
    belief_id: str,
    *,
    context: ScopeContext,
    delivery_target: str,
    cross_scope: bool = False,
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
        allowed, _reason = egress_allowed(
            ScopeTag.from_dict(scope),
            context,
            delivery_target,
            cross_scope=cross_scope,
        )
        if not allowed:
            continue
        content = str(row["body"])
        handles.append(
            _make_source_handle(
                object_id=canonical_id,
                source_kind="core_v1_evidence",
                uri=(
                    f"ocbrain://evidence/{row['evidence_id']}"
                    if delivery_target == HOSTED_MODEL_TARGET
                    else row["source_uri"]
                    or row["artifact_uri"]
                    or f"ocbrain://evidence/{row['evidence_id']}"
                ),
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
    allowed, _reason = egress_allowed(
        ScopeTag.from_dict(scope),
        context,
        delivery_target,
        cross_scope=cross_scope,
    )
    if not allowed:
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


def _explicit_contradictions(
    conn: sqlite3.Connection, items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Package only curator/compiler-declared conflicts, never lexical guesses."""
    visible = {str(item["id"]): item for item in items}
    result: list[dict[str, Any]] = []
    emitted: set[tuple[str, str]] = set()
    for belief_id, item in visible.items():
        belief = get_core_v1_belief(conn, belief_id)
        attributes = (belief or {}).get("attributes") or {}
        conflicts = attributes.get("contradicts") or attributes.get("contradiction_ids") or []
        if not isinstance(conflicts, list):
            continue
        for raw_other_id in conflicts:
            other_id = resolve_object_id(conn, str(raw_other_id))
            if other_id not in visible or other_id == belief_id:
                continue
            pair = tuple(sorted((belief_id, other_id)))
            if pair in emitted:
                continue
            emitted.add(pair)
            other = visible[other_id]
            result.append(
                {
                    "belief_id": belief_id,
                    "other_belief_id": other_id,
                    "reason": "explicit_compiler_metadata",
                    "evidence_ids": list(
                        dict.fromkeys(
                            [
                                *[str(value) for value in item.get("evidence_ids") or []],
                                *[str(value) for value in other.get("evidence_ids") or []],
                            ]
                        )
                    )[:8],
                }
            )
    return result[:12]


def prepare_retrieval_packet_v1(
    packet: dict[str, Any],
    handles: list[dict[str, Any]],
    *,
    preview: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reserve the final receipt fields before enforcing the public byte cap."""
    packet["retrieval_use_id"] = RETRIEVAL_ID_PLACEHOLDER
    packet["retrieval_use_status"] = "recorded"
    if preview:
        packet["preview"] = True
    return _enforce_context_packet_limit(packet, handles)


def bind_retrieval_id_v1(packet: dict[str, Any], retrieval_id: str) -> None:
    if len(retrieval_id) != len(RETRIEVAL_ID_PLACEHOLDER):
        raise RuntimeError("retrieval id length changed after packet budgeting")
    packet["retrieval_use_id"] = retrieval_id
    _refresh_packet_accounting(packet)
    if _serialized_bytes(packet) > MAX_CONTEXT_PACKET_BYTES:
        raise RuntimeError("final retrieval packet exceeded the hard serialized limit")


def _enforce_context_packet_limit(
    packet: dict[str, Any], handles: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coverage = packet["coverage"]
    previously_trimmed = int(coverage.get("trimmed_for_packet_limit") or 0)
    trimmed = 0
    # Leave headroom for the final accounting fields themselves.
    while packet["items"] and _serialized_bytes(packet) > MAX_CONTEXT_PACKET_BYTES - 512:
        packet["items"].pop()
        trimmed += 1
    kept_ids = {str(item["id"]) for item in packet["items"]}
    packet["contradictions"] = [
        conflict
        for conflict in packet["contradictions"]
        if conflict["belief_id"] in kept_ids and conflict["other_belief_id"] in kept_ids
    ]
    kept_source_ids = {
        str(source["id"]) for item in packet["items"] for source in item.get("sources") or []
    }
    handles = [handle for handle in handles if str(handle["id"]) in kept_source_ids]
    coverage["returned"] = len(packet["items"])
    coverage["trimmed_for_packet_limit"] = previously_trimmed + trimmed
    coverage["source_handle_count"] = len(handles)
    coverage["unavailable_sources"] = [
        value for value in coverage["unavailable_sources"] if value["object_id"] in kept_ids
    ]
    _refresh_packet_accounting(packet)
    if coverage["serialized_bytes"] > MAX_CONTEXT_PACKET_BYTES:
        raise RuntimeError("context packet accounting exceeded the hard serialized limit")
    return packet, handles


def _refresh_packet_accounting(packet: dict[str, Any]) -> None:
    coverage = packet["coverage"]
    for _attempt in range(8):
        serialized_bytes = _serialized_bytes(packet)
        estimated_tokens = max((serialized_bytes + 3) // 4, 1)
        if (
            coverage.get("serialized_bytes") == serialized_bytes
            and coverage.get("estimated_tokens") == estimated_tokens
        ):
            return
        coverage["serialized_bytes"] = serialized_bytes
        coverage["estimated_tokens"] = estimated_tokens
    raise RuntimeError("packet accounting did not converge")


def _serialized_bytes(value: dict[str, Any]) -> int:
    return len(canonical_json(value).encode("utf-8"))


def _require_v1(conn: sqlite3.Connection) -> None:
    if not is_core_v1(conn):
        raise ValueError("operation requires an OCBrain v1 core")


__all__ = [
    "build_context_v1",
    "bind_retrieval_id_v1",
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
    "prepare_retrieval_packet_v1",
    "record_context_v1",
    "search_v1",
]
