"""MCP-facing operations for the event-authoritative v1 core.

This module is deliberately separate from the legacy compatibility dispatcher.
It never queries a legacy relational knowledge table or a companion store.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from ocbrain.closeout import record_closeout
from ocbrain.config import load_config
from ocbrain.core_v1 import (
    CORE_V1_SCHEMA_VERSION,
    GOAL_BELIEF_TYPE,
    NO_COVERAGE_OUTCOME,
    RELEVANCE_OUTCOMES,
    SERVED_OUTCOME,
    SHA256_TEXT_RE,
    STABLE_OBJECT_ID_RE,
    TERMINAL_ARTIFACT_URI_RE,
    append_core_event,
    canonical_json,
    compilation_block_reason,
    evidence_body_ref,
    get_core_v1_belief,
    get_core_v1_evidence,
    is_core_v1,
    looks_like_exact_locator,
    now_iso,
    record_core_v1_evidence,
    record_core_v1_retrieval,
    resolve_object_id,
    retrieval_served_item_count,
    search_core_v1,
    sha256_text,
)
from ocbrain.deslop import ENFORCED_RULE_IDS, find_slop
from ocbrain.events import SKILL_TELEMETRY_KINDS, validate_skill_telemetry
from ocbrain.history_window import rehydrate_history_window
from ocbrain.hybrid import VECTOR_SCHEMA_VERSION, connection_path, vector_db_path
from ocbrain.ids import stable_id
from ocbrain.provenance import EMPTY_PROVENANCE, Provenance
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
from ocbrain.text import compact_whitespace
from ocbrain.vector import decode_embedding

CONTEXT_SCHEMA_VERSION = "ocbrain.context.v1"
SOURCE_SCHEMA_VERSION = "ocbrain.source.v1"
DIGEST_SCHEMA_VERSION = "ocbrain.digest.v1"
MAX_CONTEXT_PACKET_BYTES = 32_000
MAX_CONTEXT_QUERY_CHARS = 4_000
MAX_ITEM_EXCERPT_CHARS = 1_600
MAX_ITEM_SOURCE_HANDLES = 3
RETRIEVAL_ID_PLACEHOLDER = "ret_0000000000000000"
SUPERSEDE_SCHEMA_VERSION = "ocbrain.supersede.v1"
# A replacement never *gains* authority by replacing. Recency-always-wins is the
# obvious rule and the wrong one: it lets a confidently worded agent guess
# outrank a curated, evidence-backed fact simply by being typed later. The
# successor starts capped, and earns anything above this the same way the
# original did.
SUPERSEDE_CONFIDENCE_CAP = 0.7
SUPERSEDE_TIERS = ("project", "pending_all")
# The one writer entitled to supersede an ordinary belief unattended. Spelled out
# here rather than imported from `curator`, which imports this module; a test
# pins the two strings together so a curator version bump cannot silently drop
# the authority. Matching this string is *necessary* and never sufficient --
# `brain.supersede` takes `actor` straight from client arguments, so an agent can
# type it. The `curator_authored` keyword is the half a client cannot reach.
CURATOR_SUPERSEDE_WRITER = "operator-approved:wiki-curator-v2"
# Follow ``superseded_by`` this far and no further. A chain longer than this is
# a corpus problem, not a read to satisfy, and the bound is what stops a cycle
# from becoming an unbounded walk even before the seen-set catches it.
#
# This bounds the *forward* walk only, and deliberately does not transfer to
# `core_v1._belief_lineage_members`, which walks the same pointer backwards.
# Two reasons, both measured: the forward walk pays a belief read per hop while
# the backward walk loads the era pointers once and traverses in memory; and the
# backward walk is answering "what is this belief's whole record", where the
# deepest serving lineage in the 2026-08-28T19:28:58Z snapshot is 12
# generations, so bounding it here would drop two generations of verdicts out of
# ranking today.
# A test pins the divergence, so tightening one walk cannot silently tighten the
# other.
MAX_RESOLUTION_HOPS = 10
GET_MODES = ("resolve", "as_stored")
# The advisory contradiction pass is O(n^2) over the packet, so it is bounded by
# the packet, not by the corpus: at most 12 items is at most 66 pairs.
MAX_ADVISORY_PAIR_ITEMS = 12
MAX_CONTRADICTIONS = 12
# Two independently compiled beliefs this close in embedding space are saying
# the same thing about the same subject. Deliberately high: this is a serving
# hint, and a false pair costs a reader more than a missed one.
ADVISORY_COSINE_THRESHOLD = 0.90

def shared_continuity_scope(context: ScopeContext) -> ScopeTag:
    """Scope for a closeout summary: broadest shared context, never hosted.

    Continuity across clients means a summary written while Claude Code worked
    should be recallable by Codex or Cursor on the same project. So this prefers
    the widest *shared* scope (project, then repo, then client) rather than the
    narrowest one ``resolve_write_scope`` picks. Egress stays ``local_only`` so
    an unattended write can never reach hosted-model delivery; visibility is
    ``internal`` so same-instance clients share it. Task/session-only or empty
    contexts fall back to the standard narrow write scope.

    The ``auto_compiled`` provenance literal outlives the auto-compile feature
    on purpose: 575 stored closeout evidence rows already carry it, and
    changing the value here would make new writes disagree with the ledger.
    """
    for scope_type, value in (
        ("project", context.project),
        ("repo", context.repo),
        ("client", context.client),
    ):
        if value:
            return ScopeTag(
                scope_type,
                f"{scope_type}:{value}",
                visibility="internal",
                egress_policy="local_only",
                provenance="auto_compiled",
            )
    return resolve_write_scope(context)


def build_context_v1(
    conn: sqlite3.Connection,
    query: str,
    *,
    context: ScopeContext,
    limit: int,
    cross_scope: bool = False,
    delivery_target: str = LOCAL_MODEL_TARGET,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build one context packet. One retrieval, one ranking, no retry.

    Local retrieval ranks the whole serving corpus by scope affinity, so there is
    no narrower pass that could come back empty and need widening. The empty-case
    retry this used to run existed only to patch a filter that no longer exists;
    it fired on 6.75% of queries and cost a second full retrieval to recover a
    fraction of what the filter had discarded.

    ``cross_scope`` is accepted and ignored, so the five live clients that pass
    it keep working unchanged. It is deprecated in the tool schema.
    """
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
                # `confidence` and `confidence_band` used to sit here. They were
                # an authored reliability score with no measurable provenance,
                # and joined to recorded feedback they ran backwards: on the
                # reference corpus, packets judged irrelevant or harmful held
                # items averaging 0.8707 confidence against 0.7263 for packets
                # judged used or helpful. A reader weighting on that field was
                # being pointed at the rows readers liked least. These two are
                # facts about the record instead -- how many evidence objects
                # back it, and when the newest of them was recorded -- so a
                # reader can go and check rather than defer.
                "evidence_count": int(raw_item.get("evidence_count") or 0),
                "evidence_latest_at": raw_item.get("evidence_latest_at"),
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
    # Recomputed from what survived delivery gating rather than reusing the
    # ranker's mix, so the histogram describes the packet the caller is holding.
    scope_mix: dict[str, int] = {}
    for item in items:
        scope_id = str((item.get("scope") or {}).get("scope_id") or "unknown")
        scope_mix[scope_id] = scope_mix.get(scope_id, 0) + 1
    packet = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "core_schema": CORE_V1_SCHEMA_VERSION,
        "delivery_target": delivery_target,
        "query": query[:MAX_CONTEXT_QUERY_CHARS],
        "resolved_context": context.to_dict(),
        # Local packets are ranked over the whole corpus; hosted packets are
        # still selected by an explicit scope IN-list. Saying which one produced
        # this packet is honest in a way that ``cross_scope: false`` was not,
        # since that claimed isolation on packets holding cross-scope items.
        "retrieval_mode": (
            "ranked" if delivery_target == LOCAL_MODEL_TARGET else "scoped"
        ),
        "at_ts": None,
        "items": items,
        "contradictions": _packet_contradictions(conn, items),
        "coverage": {
            "requested_limit": limit,
            "returned": len(items),
            "feedback_needed": len(items) > 0,
            "scope_mix": scope_mix,
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
    provenance: Provenance | None = None,
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
        provenance=provenance,
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
        body_ref = evidence_body_ref(source)
        if body_ref is not None:
            rehydrated = rehydrate_history_window(body_ref)
            if not rehydrated.available:
                # A transcript that has grown, rotated, or been deleted is a
                # normal outcome, not an error: 12.4% of recorded source URIs
                # already dangle. Say so in the result, with the reason and the
                # head excerpt that *was* recorded, rather than raising a tool
                # error that tells the caller nothing.
                return _source_payload(
                    conn,
                    row,
                    scope=scope,
                    locator=locator,
                    delivery_target=delivery_target,
                    source_id=source_id,
                    content="",
                    max_chars=max_chars,
                    hash_verified=False,
                    content_availability="content_unavailable",
                    unavailable_reason=rehydrated.reason,
                    body_ref=body_ref,
                    recorded_head_excerpt=str(source.get("body_head") or ""),
                )
            content = rehydrated.text
        else:
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
    return _source_payload(
        conn,
        row,
        scope=scope,
        locator=locator,
        delivery_target=delivery_target,
        source_id=source_id,
        content=content,
        max_chars=max_chars,
        hash_verified=True,
        content_availability="available",
    )


def _source_payload(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    scope: ScopeTag,
    locator: dict[str, Any],
    delivery_target: str,
    source_id: str,
    content: str,
    max_chars: int,
    hash_verified: bool,
    content_availability: str,
    unavailable_reason: str | None = None,
    body_ref: dict[str, Any] | None = None,
    recorded_head_excerpt: str | None = None,
) -> dict[str, Any]:
    """One shape for both outcomes, so a caller never has to branch on absence."""
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
    payload: dict[str, Any] = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "core_schema": CORE_V1_SCHEMA_VERSION,
        "delivery_target": delivery_target,
        "id": str(row["id"]),
        "object_id": str(row["object_id"]),
        "kind": str(row["source_kind"]),
        "uri": uri,
        "scope": scope.to_dict(),
        "content_hash": str(row["content_hash"]),
        "hash_verified": hash_verified,
        "content_availability": content_availability,
        "content": excerpt,
        "truncated": truncated,
        "characters": len(excerpt),
        "issued_at": str(row["issued_at"]),
        "origin_retrieval_use_id": row["retrieval_use_id"],
        "issued_by_count": issued_by_count,
        "issued_by_retrieval_use_ids": issued_by,
    }
    if body_ref is not None:
        payload["body_storage"] = "pointer"
        payload["unavailable_reason"] = unavailable_reason
        payload["source_uri"] = str(body_ref.get("source_uri") or "")
        payload["source_bytes_at_import"] = body_ref.get("source_bytes")
        # Recorded at import and carried in the hash-chained ledger, so it is
        # verified as *what was written down* -- just not against whatever the
        # file says now. Labelled that way rather than passed off as content.
        payload["recorded_head_excerpt"] = recorded_head_excerpt or ""
        payload["recorded_head_excerpt_note"] = (
            "head recorded at import; not verified against the current source file"
        )
    return payload


EXACT_MATCH_LIMIT = 8
EXACT_MATCH_MAX_QUERY_CHARS = 512
_URI_REFERENCE_RE = re.compile(r"^[a-z][a-z0-9+.-]*:\S+$", re.IGNORECASE)
# One definition of "this query names a record", shared with ``search_core_v1``.
# Two copies is how ``brain.search`` came to short-circuit on a locator while
# ``brain.context`` fell through to dense ranking.
_SHA256_TEXT_RE = SHA256_TEXT_RE
_STABLE_OBJECT_ID_RE = STABLE_OBJECT_ID_RE
_TERMINAL_ARTIFACT_URI_RE = TERMINAL_ARTIFACT_URI_RE
_looks_like_exact_locator = looks_like_exact_locator


def exact_lookup_v1(
    conn: sqlite3.Connection,
    query: str,
    *,
    context: ScopeContext,
    cross_scope: bool = False,
    delivery_target: str = LOCAL_MODEL_TARGET,
    limit: int = EXACT_MATCH_LIMIT,
) -> list[dict[str, Any]]:
    """Exact-locator pre-pass for ``brain.search`` on the v1 core.

    Semantic ranking cannot answer "show me closeout X" or "the artifact with
    hash H": a locator string shares no lexical terms with unrelated belief
    bodies, so stale beliefs outrank the exact record. When the query *is* a
    locator, equality lookups short-circuit ranking. A locator is an event,
    evidence, belief, closeout, or retrieval-use id, an artifact URI or
    SHA-256, or an exact ``task_ref`` on a recorded closeout.

    Matches are metadata-only and scope-gated like any other delivery; expand
    bodies through ``brain.get`` / ``brain.source``. ``retrieval_uses.task_ref``
    is deliberately *not* matched: those refs are auto-derived from past query
    text (``brain.search:<query>``), so matching them would let a repeated
    search hijack itself.
    """
    delivery_target = normalize_delivery_target(delivery_target)
    text = str(query).strip()
    if not text or len(text) > EXACT_MATCH_MAX_QUERY_CHARS:
        return []
    limit = max(int(limit), 1)
    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, object_id: str, matched_by: str, **fields: Any) -> None:
        key = (kind, object_id)
        if key in seen:
            return
        seen.add(key)
        matches.append(
            {
                "kind": kind,
                "id": object_id,
                "matched_by": matched_by,
                **{k: v for k, v in fields.items() if v is not None},
            }
        )

    def _evidence_scope_allowed(evidence: dict[str, Any]) -> bool:
        return _scope_allowed_for_delivery(
            evidence.get("scope"),
            context=context,
            delivery_target=delivery_target,
            cross_scope=cross_scope,
        )

    def _stored_context_allowed(raw: Any, *, task_ref: str | None = None) -> bool:
        """Closeouts and retrieval receipts resolve by exact id, locally.

        Asking for ``close_abc123`` by its id is not a topical search that might
        stray into someone else's material: the caller already holds the
        identifier. Requiring their live context to match the one stored on the
        record meant a receipt id copied out of a handoff note resolved to
        nothing. Hosted delivery still never sees these records.
        """
        return delivery_target == LOCAL_MODEL_TARGET

    def _event_scope_allowed(
        event: sqlite3.Row,
        *,
        visited: frozenset[str] = frozenset(),
    ) -> bool:
        event_id = str(event["id"])
        if event_id in visited or len(visited) >= 4:
            return False
        visited = visited | {event_id}
        try:
            body = json.loads(str(event["body_json"]))
        except (json.JSONDecodeError, TypeError, ValueError):
            return False
        if not isinstance(body, dict):
            return False
        raw_scope = body.get("scope")
        if isinstance(raw_scope, dict):
            return _scope_allowed_for_delivery(
                raw_scope,
                context=context,
                delivery_target=delivery_target,
                cross_scope=cross_scope,
            )
        subject = body.get("subject")
        if not isinstance(subject, dict):
            return False
        subject_id = str(subject.get("id") or "")
        subject_kind = str(subject.get("kind") or "")
        if subject_kind == "evidence":
            evidence = get_core_v1_evidence(conn, subject_id)
            return evidence is not None and _evidence_scope_allowed(evidence)
        if subject_kind == "belief":
            belief = get_core_v1_belief(conn, subject_id)
            return belief is not None and _scope_allowed_for_delivery(
                belief.get("scope"),
                context=context,
                delivery_target=delivery_target,
                cross_scope=cross_scope,
            )
        if subject_kind in {"event", "proposal"}:
            parent = conn.execute(
                "SELECT id, body_json FROM brain_events WHERE id=?",
                (subject_id,),
            ).fetchone()
            return parent is not None and _event_scope_allowed(parent, visited=visited)
        return False

    def _add_evidence(evidence_id: str, matched_by: str) -> None:
        evidence = get_core_v1_evidence(conn, evidence_id)
        if evidence is None or not _evidence_scope_allowed(evidence):
            return
        scope = evidence.get("scope") or {}
        add(
            "evidence",
            str(evidence["evidence_id"]),
            matched_by,
            evidence_kind=str(evidence["kind"]),
            artifact_uri=(
                evidence.get("artifact_uri")
                if delivery_target == LOCAL_MODEL_TARGET
                else f"ocbrain://evidence/{evidence['evidence_id']}"
            ),
            artifact_hash=evidence.get("artifact_hash"),
            content_hash=evidence.get("content_hash"),
            occurred_at=evidence.get("occurred_at"),
            scope_id=scope.get("scope_id"),
        )

    def _add_closeout(row: sqlite3.Row, matched_by: str) -> None:
        if not _stored_context_allowed(row["context_json"], task_ref=str(row["task_ref"])):
            return
        add(
            "closeout",
            str(row["id"]),
            matched_by,
            task_ref=str(row["task_ref"]),
            status=str(row["status"]),
            closed_at=str(row["closed_at"]),
        )

    # Stable-id equality lookups (primary keys / unique columns only).
    event = conn.execute(
        "SELECT id, ts, kind, writer, body_json FROM brain_events WHERE id=?", (text,)
    ).fetchone()
    if event is not None and _event_scope_allowed(event):
        add(
            "event",
            str(event["id"]),
            "id",
            ts=str(event["ts"]),
            event_kind=str(event["kind"]),
            writer=(
                str(event["writer"])
                if delivery_target == LOCAL_MODEL_TARGET
                else None
            ),
        )
    _add_evidence(text, "id")
    belief = get_core_v1_belief(conn, text)
    if belief is not None and _scope_allowed_for_delivery(
        belief.get("scope"),
        context=context,
        delivery_target=delivery_target,
        cross_scope=cross_scope,
    ):
        add(
            "belief",
            str(belief["belief_id"]),
            "id",
            status=str(belief.get("status") or ""),
            belief_type=belief.get("belief_type"),
            confidence=belief.get("confidence"),
            scope_id=(belief.get("scope") or {}).get("scope_id"),
        )
    closeout = conn.execute(
        "SELECT id, task_ref, status, closed_at, context_json "
        "FROM task_closeouts WHERE id=?",
        (text,),
    ).fetchone()
    if closeout is not None:
        _add_closeout(closeout, "id")
    retrieval = conn.execute(
        "SELECT id, task_ref, outcome, served_at, context_json "
        "FROM retrieval_uses WHERE id=?",
        (text,),
    ).fetchone()
    if retrieval is not None and _stored_context_allowed(
        retrieval["context_json"],
        task_ref=str(retrieval["task_ref"] or ""),
    ):
        add(
            "retrieval_use",
            str(retrieval["id"]),
            "id",
            task_ref=retrieval["task_ref"],
            outcome=str(retrieval["outcome"]),
            served_at=str(retrieval["served_at"]),
        )

    # Exact task_ref on recorded closeouts ("show me closeout X").
    for row in conn.execute(
        "SELECT id, task_ref, status, closed_at, context_json FROM task_closeouts "
        "WHERE task_ref=? ORDER BY closed_at DESC LIMIT ?",
        (text, limit),
    ):
        _add_closeout(row, "task_ref")

    # Artifact URI equality (evidence columns, then closeout artifact refs).
    # The uri columns are unindexed, so only scan them for path- or URI-like
    # queries. The broader URI syntax permits exact matches for stored opaque
    # references, while terminal-miss handling remains limited to known forms.
    if "/" in text or _URI_REFERENCE_RE.fullmatch(text):
        for row in conn.execute(
            "SELECT evidence_id FROM evidence_objects "
            "WHERE artifact_uri=? OR source_uri=? LIMIT ?",
            (text, text, limit),
        ):
            _add_evidence(str(row["evidence_id"]), "artifact_uri")
        for row in conn.execute(
            "SELECT id, task_ref, status, closed_at, context_json, artifact_refs_json "
            "FROM task_closeouts WHERE artifact_refs_json LIKE '%' || ? || '%' LIMIT ?",
            (text, limit * 4),
        ):
            refs = json.loads(row["artifact_refs_json"] or "[]")
            if any(
                isinstance(ref, dict) and str(ref.get("uri") or "") == text for ref in refs
            ):
                _add_closeout(row, "artifact_uri")

    # SHA-256 equality (evidence hashes, closeout receipt hash, artifact refs).
    lowered = text.lower()
    if _SHA256_TEXT_RE.match(lowered):
        for row in conn.execute(
            "SELECT evidence_id, artifact_hash, content_hash FROM evidence_objects "
            "WHERE artifact_hash=? OR content_hash=? LIMIT ?",
            (lowered, lowered, limit),
        ):
            matched = (
                "artifact_sha256"
                if str(row["artifact_hash"] or "") == lowered
                else "content_sha256"
            )
            _add_evidence(str(row["evidence_id"]), matched)
        receipt = conn.execute(
            "SELECT id, task_ref, status, closed_at, context_json FROM task_closeouts "
            "WHERE content_hash=?",
            (lowered,),
        ).fetchone()
        if receipt is not None:
            _add_closeout(receipt, "content_sha256")
        for row in conn.execute(
            "SELECT id, task_ref, status, closed_at, context_json, artifact_refs_json "
            "FROM task_closeouts WHERE artifact_refs_json LIKE '%' || ? || '%' LIMIT ?",
            (lowered, limit * 4),
        ):
            refs = json.loads(row["artifact_refs_json"] or "[]")
            if any(
                isinstance(ref, dict) and str(ref.get("sha256") or "").lower() == lowered
                for ref in refs
            ):
                _add_closeout(row, "artifact_sha256")

    return matches[:limit]


def search_v1(
    conn: sqlite3.Connection,
    query: str,
    *,
    context: ScopeContext,
    limit: int,
    cross_scope: bool,
    delivery_target: str = LOCAL_MODEL_TARGET,
    provenance: Provenance | None = None,
) -> dict[str, Any]:
    exact_matches = exact_lookup_v1(
        conn,
        query,
        context=context,
        cross_scope=cross_scope,
        delivery_target=delivery_target,
        limit=min(limit, EXACT_MATCH_LIMIT),
    )
    if exact_matches or _looks_like_exact_locator(query):
        payload = {
            "schema_version": "ocbrain.search.v1",
            "delivery_target": delivery_target,
            "query": query,
            "resolved_context": context.to_dict(),
            "match_mode": "exact",
            "items": [],
            "exact_matches": exact_matches,
            "contradictions": [],
            "coverage": {
                "requested_limit": limit,
                "returned": len(exact_matches),
                "feedback_needed": bool(exact_matches),
            },
        }
        retrieval_id = record_core_v1_retrieval(
            conn,
            query=str(payload["query"]),
            context={**context.to_dict(), "delivery_target": payload["delivery_target"]},
            items=[
                {
                    "object_id": item["id"],
                    "object_kind": item["kind"],
                    "score": 1.0,
                }
                for item in exact_matches
            ],
            runtime=context.runtime or "mcp",
            task_ref=context.task or f"brain.search:{payload['query']}",
            session_id=context.session,
            packet_schema="ocbrain.search.v1",
            provenance=provenance,
        )
        payload["retrieval_use_id"] = retrieval_id
        payload["retrieval_use_status"] = "recorded"
        return payload
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
        provenance=provenance,
    )
    issue_source_handles(conn, handles, retrieval_use_id=retrieval_id)
    bind_retrieval_id_v1(payload, retrieval_id)
    return payload


def _resolve_supersession_chain(
    conn: sqlite3.Connection, belief: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str], str | None]:
    """Walk ``superseded_by`` forward to the belief that is actually serving.

    Returns the head, the ids walked through to reach it, and -- when there is
    no head -- why. Both a seen-set and a hop bound guard the walk: the seen-set
    catches a genuine cycle, and the bound stops a long or corrupted chain from
    turning one read into an unbounded scan.
    """
    walked: list[str] = []
    seen = {str(belief["canonical_id"])}
    current = belief
    for _hop in range(MAX_RESOLUTION_HOPS):
        successor_id = str((current.get("attributes") or {}).get("superseded_by") or "").strip()
        if not successor_id:
            return None, walked, "the chain ends without a successor"
        successor = get_core_v1_belief(conn, successor_id)
        if successor is None:
            return None, walked, f"successor {successor_id} is not in the corpus"
        canonical = str(successor["canonical_id"])
        if canonical in seen:
            return None, walked, f"the chain cycles back to {canonical}"
        seen.add(canonical)
        walked.append(str(current["canonical_id"]))
        if successor.get("status") == "current" and successor.get("serve"):
            return successor, walked, None
        current = successor
    return None, walked, f"the chain is longer than {MAX_RESOLUTION_HOPS} hops"


def get_v1(
    conn: sqlite3.Connection,
    object_id: str,
    *,
    context: ScopeContext,
    include_candidate: bool = False,
    include_private: bool = False,
    cross_scope: bool = False,
    delivery_target: str = LOCAL_MODEL_TARGET,
    mode: str = "resolve",
) -> dict[str, Any]:
    """Return one object by id.

    ``mode="resolve"`` (the default) answers the question a caller holding a
    stale id is actually asking -- *what is true now* -- by following
    ``superseded_by`` to the serving head and saying which ids it came through.
    ``mode="as_stored"`` answers the other one -- *what did we believe then* --
    and is how drift is measured. A retracted belief with no successor stays
    refused in both modes: filtering by default is the point, and a store that
    hands back invalidated facts unless you remember to exclude them is a
    footgun, not a feature.
    """
    if mode not in GET_MODES:
        raise ValueError(f"mode must be one of {', '.join(GET_MODES)}")
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
        served = belief
        resolved_from: list[str] = []
        invalidated = False
        if belief.get("status") != "current" or not belief.get("serve"):
            successor_id = str(attributes.get("superseded_by") or "").strip()
            has_successor = belief.get("status") == "retracted" and bool(successor_id)
            if include_candidate and belief.get("status") == "candidate":
                pass
            elif has_successor and mode == "as_stored":
                invalidated = True
            elif has_successor:
                served, resolved_from, failure = _resolve_supersession_chain(conn, belief)
                if served is None:
                    raise PermissionError(
                        "non-current beliefs are not served by brain.get: "
                        f"{failure}; read it with mode=as_stored to see what was stored"
                    )
                _authorize_get_scope(
                    served["scope"],
                    context=context,
                    include_private=include_private,
                    cross_scope=cross_scope,
                    delivery_target=delivery_target,
                )
            else:
                raise PermissionError("non-current beliefs are not served by brain.get")
        public_belief = _belief_for_delivery(served, delivery_target=delivery_target)
        public_belief["evidence_ids"] = _evidence_ids_for_delivery(
            conn,
            served.get("evidence_ids") or [],
            context=context,
            delivery_target=delivery_target,
            cross_scope=cross_scope,
        )
        payload = {
            "schema_version": "ocbrain.object.v1",
            "delivery_target": delivery_target,
            "object_kind": "belief",
            "mode": mode,
            **public_belief,
        }
        if resolved_from:
            payload["requested_id"] = object_id
            payload["resolved_from"] = resolved_from
            payload["resolution_hops"] = len(resolved_from)
        if invalidated:
            payload["invalidated"] = True
            payload["superseded_by"] = successor_id
            payload["valid_from"] = attributes.get("valid_from")
            payload["valid_until"] = attributes.get("valid_until")
        return payload
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
    since: str | None = None,
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
        # Goals are excluded here for the same reason they are excluded from
        # retrieval: a digest reports current *knowledge*, and a goal is task
        # state. `brain.briefing` serves goals, deterministically and by status.
        "SELECT * FROM current_beliefs WHERE serve=1 AND status='current' "
        "AND COALESCE(belief_type, '') <> ? "
        "ORDER BY pinned DESC, last_compiled_at DESC, belief_id LIMIT ?",
        (GOAL_BELIEF_TYPE, max(limit * 8, 40)),
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
        # An operator who never asks about proposals still needs to know a
        # correction is waiting on them: an unapproved supersession means the
        # corpus is knowingly serving something an agent has already contested.
        "pending_corrections": pending_supersede_count(conn),
        "recent_closeouts": _recent_closeouts_v1(
            conn,
            context=context,
            limit=min(limit, 5),
            since=since,
            delivery_target=delivery_target,
        ),
        "excluded_scope_count": excluded,
    }


def _recent_closeouts_v1(
    conn: sqlite3.Connection,
    *,
    context: ScopeContext,
    limit: int,
    since: str | None,
    delivery_target: str,
) -> list[dict[str, Any]]:
    """Return a tiny recent-work register without promoting receipts to beliefs.

    Closeouts are execution receipts, not current truth.  They are still the
    best source for "what just happened?" before a curator has promoted a
    durable fact.  Keep this lane deliberately narrow: high-signal receipts
    only, newest receipt per task, at most five.
    """
    if delivery_target != LOCAL_MODEL_TARGET:
        return []
    earliest = _digest_since(since)
    rows = conn.execute(
        """
        SELECT receipt_json
        FROM task_closeouts
        WHERE closed_at >= ?
        ORDER BY closed_at DESC, id DESC
        LIMIT 250
        """,
        (earliest.isoformat(timespec="microseconds"),),
    )
    result: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for row in rows:
        receipt = json.loads(row["receipt_json"])
        task_ref = str(receipt.get("task_ref") or "").strip()
        if not task_ref or task_ref in seen_tasks:
            continue
        if not _closeout_matches_context(receipt, context):
            continue
        if not _high_signal_closeout(receipt):
            continue
        seen_tasks.add(task_ref)
        artifacts = []
        for artifact in receipt.get("artifact_refs") or []:
            if not isinstance(artifact, dict) or not artifact.get("uri"):
                continue
            artifacts.append(
                {
                    key: artifact[key]
                    for key in ("uri", "kind", "label", "sha256")
                    if artifact.get(key)
                }
            )
            if len(artifacts) >= 4:
                break
        decision = receipt.get("decision") if isinstance(receipt.get("decision"), dict) else {}
        result.append(
            {
                "id": receipt.get("id"),
                "task_ref": task_ref,
                "status": receipt.get("status"),
                "summary": _human_excerpt(receipt.get("summary"), 600),
                "closed_at": receipt.get("closed_at"),
                "verification_status": receipt.get("verification_status"),
                "decision_impact": decision.get("impact"),
                "decision_note": _human_excerpt(decision.get("note"), 300),
                "artifacts": artifacts,
                "context": {
                    key: value
                    for key, value in (receipt.get("context") or {}).items()
                    if key in {"project", "repo", "client", "task", "runtime"} and value
                },
            }
        )
        if len(result) >= limit:
            break
    return result


def _digest_since(raw: str | None) -> datetime:
    if raw:
        normalized = raw.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError("since must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return datetime.now(UTC) - timedelta(hours=72)


def _closeout_matches_context(receipt: dict[str, Any], context: ScopeContext) -> bool:
    receipt_context = receipt.get("context")
    if not isinstance(receipt_context, dict):
        receipt_context = {}
    scoped = False
    for key in ("project", "repo", "client"):
        expected = getattr(context, key)
        if not expected:
            continue
        scoped = True
        if str(receipt_context.get(key) or "") != expected:
            return False
    if not scoped and context.task:
        return (
            str(receipt.get("task_ref") or "") == context.task
            or str(receipt_context.get("task") or "") == context.task
        )
    return True


def _high_signal_closeout(receipt: dict[str, Any]) -> bool:
    status = str(receipt.get("status") or "")
    if status not in {"completed", "partial", "blocked"}:
        return False
    artifacts = receipt.get("artifact_refs")
    has_artifacts = isinstance(artifacts, list) and any(
        isinstance(item, dict) and item.get("uri") for item in artifacts
    )
    decision = receipt.get("decision")
    impact = str(decision.get("impact") or "") if isinstance(decision, dict) else ""
    verification = str(receipt.get("verification_status") or "")
    return (
        verification == "verified"
        or has_artifacts
        or impact in {"changed", "prevented_error"}
        or (status == "blocked" and bool(receipt.get("awaiting")))
    )


def _human_excerpt(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def feedback_v1(
    conn: sqlite3.Connection,
    retrieval_use_id: str,
    *,
    outcome: str,
    note: str | None,
) -> dict[str, Any]:
    """Record how one retrieval's *served items* turned out.

    Every value here judges items. A retrieval that served nothing has no items
    to judge, and the server refuses one rather than letting "the brain had no
    coverage" be filed as "the brain served junk" -- the two readings share one
    column, and feedback is the only ranking signal the corpus has.

    The rule was written instruction-side first ("when a retrieval returns zero
    items, do not file brain.feedback for it") and it did not hold: in the
    corpus snapshot frozen at 2026-08-28T19:28:58Z, 183 of the 1,086 zero-item
    retrievals carry a relevance verdict anyway, 174 of them ``irrelevant``.
    The zero-item case is now recorded by the server as ``no_coverage`` when the
    receipt is written, from the item count it already holds, and is not a value
    a caller can file.
    """
    if outcome in {NO_COVERAGE_OUTCOME, SERVED_OUTCOME}:
        raise ValueError(
            f"'{outcome}' is recorded by the server when the receipt is written, "
            "from the number of items it served; brain.feedback only takes a "
            f"judgement of served items: {', '.join(RELEVANCE_OUTCOMES)}"
        )
    if outcome not in RELEVANCE_OUTCOMES:
        raise ValueError(f"outcome must be one of: {', '.join(sorted(RELEVANCE_OUTCOMES))}")
    served = retrieval_served_item_count(conn, retrieval_use_id)
    if served is None:
        raise ValueError(f"retrieval use not found: {retrieval_use_id}")
    if served == 0:
        raise ValueError(
            "empty_retrieval_not_feedback_eligible: "
            f"retrieval {retrieval_use_id} served zero items, so there is nothing to "
            f"judge '{outcome}'; the server already recorded it as "
            f"'{NO_COVERAGE_OUTCOME}'. Do not re-poll the same query -- if the gap "
            "matters, write what was missing with brain.ingest, or record the work "
            "with brain.closeout"
        )
    updated = conn.execute(
        "UPDATE retrieval_uses SET outcome=?, note=COALESCE(?, note), "
        "feedback_source='runtime_explicit', feedback_at=? WHERE id=?",
        (outcome, note, now_iso(), retrieval_use_id),
    )
    if updated.rowcount == 0:  # pragma: no cover - existence was checked above.
        raise RuntimeError(f"retrieval feedback update failed: {retrieval_use_id}")
    return {"retrieval_use_id": retrieval_use_id, "outcome": outcome, "served_items": served}


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
    telemetry = kind in SKILL_TELEMETRY_KINDS
    if telemetry:
        envelope = validate_skill_telemetry(body)
        if envelope["kind"] != kind:
            raise ValueError("skill telemetry body kind must match brain.ingest kind")
        body = canonical_json(envelope)
    scope = resolve_write_scope(context)
    evidence_id, event_id = record_core_v1_evidence(
        conn,
        body=body,
        kind=kind,
        scope=scope,
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
    unresolved: str | None = None,
    runtime_detail: str | None = None,
    parent_closeout_id: str | None = None,
    provenance: Provenance | None = None,
) -> dict[str, Any]:
    # Report the writing standard back to whoever wrote the summary. Reporting
    # rather than refusing is the default: the curator gate already stops slop
    # from becoming a served belief, and refusing a closeout throws away the
    # client's work over a style rule. `deslop.reject_closeout_slop` hardens it
    # for an operator who has calibrated the rules against their own corpus --
    # and it is checked here, before anything is written, because a refusal after
    # the receipt exists would leave the closeout recorded and the caller told it
    # failed.
    slop = find_slop(summary, {"lifecycle": "durable"}, rules=ENFORCED_RULE_IDS)
    if slop and load_config().deslop.reject_closeout_slop:
        raise ValueError(
            "closeout summary trips "
            + ", ".join(finding.rule for finding in slop)
            + "; state one durable fact per closeout, or set "
            "deslop.reject_closeout_slop=false to report instead of refuse"
        )
    receipt = record_closeout(
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
        unresolved=unresolved,
        runtime_detail=runtime_detail,
        actor=actor,
        parent_closeout_id=parent_closeout_id,
        provenance=provenance,
    )
    # Always record the summary as scoped evidence, under the shared continuity
    # scope so a closeout written while one client worked is curatable and
    # recallable by the others.
    #
    # Recording evidence is not promotion. An earlier version gated this write
    # on the automatic_activation flag and so lost 567 of 799 closeouts (71%)
    # in one real brain when the flag was off. Closeout summaries are the
    # largest supply of curator-eligible evidence; they are always recorded.
    scope = shared_continuity_scope(context)
    evidence_id, _event_id = record_core_v1_evidence(
        conn,
        body=summary,
        kind="task_closeout_summary",
        scope=scope,
        writer=actor,
        artifact_ref=f"closeout:{receipt['id']}",
    )
    receipt["evidence_id"] = evidence_id
    if slop:
        receipt["slop_findings"] = [finding.to_dict() for finding in slop]
    return receipt


def correct_v1(
    conn: sqlite3.Connection,
    *,
    layer: str,
    target: str,
    op: str,
    body: str | None,
    actor: str,
    hard: bool,
    successor_id: str | None = None,
    attributes_patch: dict[str, Any] | None = None,
    requested_by: str | None = None,
    provenance: Provenance | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Append one correction event.

    ``provenance`` is the connection's server-observed identity. A correction is
    the most consequential thing anyone writes to this ledger and, until now,
    the least attributable: ``actor`` defaulted to the literal string "human"
    and every correction event in one real 719-event corpus carried a NULL
    session id. Recording who and which connection issued it is what makes a
    later audit of "who retired this belief" answerable at all.
    """
    if layer not in {"knowledge", "belief"}:
        raise ValueError("layer must be knowledge or belief; evidence corrections are unsupported")
    provenance = provenance or EMPTY_PROVENANCE
    event_body: dict[str, Any] = {
        "schema_version": "ocbrain.correction.v1",
        "subject": {"kind": layer, "id": resolve_object_id(conn, target)},
        "target_layer": layer,
        "target_id": target,
        "op": op,
        "body": body,
        "author": actor,
        "hard": bool(hard),
        "provenance": provenance.to_dict(),
    }
    if successor_id:
        event_body["successor_id"] = successor_id
    if attributes_patch is not None:
        event_body["attributes_patch"] = attributes_patch
    if requested_by:
        event_body["requested_by"] = requested_by
    event_id = append_core_event(
        conn,
        "correction_recorded",
        event_body,
        writer=actor,
        # Harness-attested beats model-typed: the environment variable the
        # client process was launched with is not something a model can invent.
        session_id=provenance.client_session_hint or session_id,
        project=True,
    )
    return {"event_id": event_id, "kind": "correction_recorded"}



@contextmanager
def _one_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Run a multi-event write as one transaction, or join the caller's.

    ``append_core_event`` opens and closes its own transaction only when it
    finds the connection in autocommit mode, so holding one open around several
    appends is what makes "retire the old belief and serve its replacement" a
    single visible step. Committing stays the caller's job; a failure part-way
    rolls back only the transaction this helper opened.
    """
    opened = not conn.in_transaction
    if opened:
        conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        if opened and conn.in_transaction:
            conn.rollback()
        raise


def _normalized_body_hash(body: str) -> str:
    """Hash a belief body modulo whitespace and case.

    Self-supersession is the failure mode that would make this primitive worse
    than useless: an agent re-typing the stored fact with a different indent
    would retire a good belief and replace it with itself under a new id.
    """
    return sha256_text(compact_whitespace(body).casefold())


# One definition of "nobody has decided this yet", used by the proposal listing,
# the queue depth, the selftest metric and the dedup guard. Two spellings of this
# predicate is how a dedup guard drifts away from the count that is supposed to
# prove it works, so there is one, and it names its outer table `proposal`.
_PROPOSAL_DECIDED_SQL = (
    "EXISTS (SELECT 1 FROM brain_events AS decision "
    "WHERE decision.kind='compilation_decided' "
    "AND json_extract(decision.body_json, '$.proposal_event_id') = proposal.id)"
)
_UNDECIDED_SUPERSEDE_SQL = (
    "FROM brain_events AS proposal WHERE proposal.kind='compilation_proposed' "
    "AND json_extract(proposal.body_json, '$.attributes.supersedes') IS NOT NULL "
    f"AND NOT {_PROPOSAL_DECIDED_SQL}"
)


def is_curator_writer(actor: str) -> bool:
    """Whether this writer string is the scheduled wiki curator's.

    Necessary but never sufficient for curator authority: see
    :data:`CURATOR_SUPERSEDE_WRITER`.
    """
    return str(actor or "").strip() == CURATOR_SUPERSEDE_WRITER


def _supersede_config() -> tuple[str, int, bool]:
    config = load_config().supersede
    tier = str(config.tier or "project").strip().lower()
    if tier not in SUPERSEDE_TIERS:
        tier = "project"
    return tier, int(config.direct_cap), bool(config.curator_direct)


def _recent_supersede_count(
    conn: sqlite3.Connection, *, actor: str, provenance: Provenance
) -> int:
    """Supersessions this caller has already landed in the trailing 24 hours.

    Matched on the harness-attested session hint when there is one, because that
    is the identity an agent cannot type for itself. ``writer`` is the fallback,
    and it is a weaker key: two agents sharing an actor name share a budget.
    """
    since = (datetime.now(UTC) - timedelta(hours=24)).isoformat(timespec="microseconds")
    hint = provenance.client_session_hint
    if hint:
        row = conn.execute(
            "SELECT COUNT(*) FROM brain_events WHERE kind='correction_recorded' AND ts >= ? "
            "AND json_extract(body_json, '$.op')='supersede' "
            "AND json_extract(body_json, '$.provenance.client_session_hint')=?",
            (since, hint),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) FROM brain_events WHERE kind='correction_recorded' AND ts >= ? "
            "AND json_extract(body_json, '$.op')='supersede' AND writer=?",
            (since, actor),
        ).fetchone()
    return int(row[0])


def _supersede_route(
    conn: sqlite3.Connection,
    belief: dict[str, Any],
    *,
    actor: str,
    provenance: Provenance,
    curator_authored: bool = False,
) -> str | None:
    """Why this supersession needs an admin, or ``None`` to land it directly.

    Both routes always exist. The tier flag picks the predicate, never which
    code is compiled, so an operator turning ``pending_all`` on and off again
    exercises paths that were live the whole time.

    The doctrine and pinned rules are checked before any authority is granted,
    so the curator's direct authority covers ordinary beliefs and nothing else.
    The rate cap is the *last* question asked, and only the curator is excused
    from it: it exists to bound an untrusted caller and still does that exactly
    as it did.
    """
    tier, cap, curator_direct = _supersede_config()
    if tier == "pending_all":
        return "tier is pending_all: every runtime supersession is reviewed"
    scope_type = str((belief.get("scope") or {}).get("scope_type") or "")
    scope_id = str((belief.get("scope") or {}).get("scope_id") or "")
    if scope_type == "global":
        return f"target is doctrine ({scope_id}); doctrine is never replaced unattended"
    if belief.get("pinned"):
        return "target is pinned; a pin is a standing operator decision"
    if curator_direct and curator_authored and is_curator_writer(actor):
        # A per-caller cap sized for a runtime agent is the wrong instrument for
        # a scheduled process that recompiles the whole corpus every hour: past
        # the eighth correction it pends everything, forever, and the ledger
        # grows without bound because nothing it pends ever changes the input
        # that produced it. The margin rule and the digest gate are what
        # actually bound the curator, and both still apply above and below here.
        return None
    recent = _recent_supersede_count(conn, actor=actor, provenance=provenance)
    if cap >= 0 and recent >= cap:
        return f"rate cap reached: {recent} supersessions in the trailing 24h (cap {cap})"
    return None


def supersede_v1(
    conn: sqlite3.Connection,
    *,
    target: str,
    body: str,
    reason: str,
    context: ScopeContext,
    actor: str,
    provenance: Provenance | None = None,
) -> dict[str, Any]:
    """Replace one serving belief with a corrected one, atomically.

    This is the primitive the ledger has been missing. Every correction an agent
    has ever issued took the only shape available to it -- soft-retract the wrong
    belief, then type the replacement into the correction's ``body``, a field
    nothing indexes and nothing serves -- so correcting the brain *destroyed*
    knowledge instead of updating it. Here the replacement is a first-class
    belief: it is compiled, scoped, evidenced, searchable, and reachable from the
    id the old fact was known by.

    Three properties are deliberate and are what separate this from "write the
    new thing and hope":

    * **The scope is copied verbatim.** A supersession can never widen reach. A
      project fact's replacement is a project fact; there is no argument that
      moves it to doctrine.
    * **The confidence is capped.** ``min(old, 0.7)``: recency is not authority.
    * **The old row keeps its history.** Feedback and retrieval rows are
      append-only and stay attached to the belief that earned them; the
      successor inherits none of that boost and earns its own.
    """
    _require_v1(conn)
    provenance = provenance or EMPTY_PROVENANCE
    statement = body.strip()
    if not statement:
        raise ValueError("body must carry the replacement statement")
    rationale = reason.strip()
    if not rationale:
        raise ValueError("reason must say why the stored belief is wrong")

    old = get_core_v1_belief(conn, target)
    if old is None:
        raise ValueError(f"belief not found: {target}")
    old_id = str(old["canonical_id"])
    if old.get("status") != "current" or not old.get("serve"):
        raise ValueError(
            f"only a serving belief can be superseded; {old_id} is {old.get('status')} "
            f"and serve={int(bool(old.get('serve')))}"
        )
    if _normalized_body_hash(statement) == _normalized_body_hash(str(old.get("body") or "")):
        raise ValueError(
            "the replacement restates the stored belief; supersession must change the claim"
        )

    inherited = dict(old.get("attributes") or {})
    return supersede_transaction(
        conn,
        old=old,
        statement=statement,
        rationale=rationale,
        attributes={
            key: inherited[key]
            for key in ("key", "title", "category", "lifecycle")
            if key in inherited
        },
        actor=actor,
        provenance=provenance,
        session_id=context.session,
    )


def supersede_transaction(
    conn: sqlite3.Connection,
    *,
    old: dict[str, Any],
    statement: str,
    rationale: str,
    attributes: dict[str, Any],
    actor: str,
    provenance: Provenance,
    session_id: str | None = None,
    evidence_ids: list[str] | None = None,
    confidence_ceiling: float | None = None,
    extra_pending_reason: str | None = None,
    curator_authored: bool = False,
    inherit_confidence: bool = False,
) -> dict[str, Any]:
    """Retire one serving belief and stand its replacement up, atomically.

    Shared by the runtime primitive (:func:`supersede_v1`) and by the wiki
    curator, whose key-collision cascade produces exactly this transaction out
    of a compiled claim instead of out of an agent's correction. Both doors lead
    to the same era stamp, the same confidence cap, the same tier routing, and
    the same paired correction, because a supersession an operator can audit
    must not depend on which one it came through.

    ``attributes`` is the successor's own metadata: the runtime path inherits it
    from the belief being replaced, the curator supplies the claim's. Either way
    ``supersedes`` and ``valid_from`` are stamped here and cannot be passed in.

    ``evidence_ids`` supports the successor. The runtime path leaves it unset
    and the rationale evidence recorded here is the only support it has; the
    curator passes the claim's own quote-validated evidence, and the rationale
    stays reachable through ``attributes.correction_evidence_id``.

    ``confidence_ceiling`` applies on top of ``min(old, 0.7)``, so a caller that
    carries a confidence of its own cannot publish a successor more confident
    than the claim behind it.

    ``extra_pending_reason`` forces the pending route for a caller that has
    decided this supersession is not safe to land unattended even where the tier
    rules would let it.

    ``curator_authored`` is the half of curator authority a client cannot reach.
    ``brain.supersede`` takes its ``actor`` straight from tool arguments, so the
    writer string alone would let any agent buy unlimited unattended supersession
    by typing the curator's name; :func:`supersede_v1` neither accepts this
    keyword nor passes it, and both halves are required.

    ``inherit_confidence`` keeps the successor at the predecessor's confidence
    instead of the ``min(old, 0.7)`` ceiling, and is honoured only for a
    curator-authored supersession that keeps the predecessor's ``key``. The
    ceiling is right for a *contested correction* -- a replacement must not gain
    authority by replacing -- and wrong for the curator refreshing its own fact
    from better evidence, which is the same claim restated. Left capped, every
    scheduled refresh ratcheted the corpus toward 0.7: on the live core, 30 of
    33 pending proposals would have dropped confidence on approval, mean -0.09.
    Inheriting is no-gain as well as no-loss -- a more confident claim still does
    not raise the fact, because arriving later is still not evidence.
    """
    old_id = str(old["canonical_id"])
    scope = ScopeTag.from_dict(dict(old.get("scope") or {}))
    successor_id = stable_id("belief", "sup", statement, scope.scope_id)
    # Content-and-scope addressed, so two agents reaching the same conclusion
    # converge on one belief instead of minting two -- and, because "sup" is part
    # of the digest input, a successor id can never collide with the id of the
    # belief it replaces.
    head_seq = int(
        conn.execute("SELECT COALESCE(MAX(event_seq), 0) FROM brain_events").fetchone()[0]
    )
    # Ask the same question the decision gate will ask, before writing anything,
    # so a previously banned body comes back as a sentence instead of a
    # PermissionError raised half-way through the transaction.
    blocked = compilation_block_reason(conn, successor_id, proposal_event_seq=head_seq)
    if blocked is not None:
        raise ValueError(f"blocked: this content was previously {blocked}")

    stored_confidence = float(old.get("confidence") or SUPERSEDE_CONFIDENCE_CAP)
    curator_call = curator_authored and is_curator_writer(actor)
    old_key = str((old.get("attributes") or {}).get("key") or "")
    new_key = str(attributes.get("key") or "")
    same_key_refresh = bool(old_key) and old_key == new_key
    if inherit_confidence and curator_call and same_key_refresh:
        confidence = stored_confidence
    else:
        ceilings = [stored_confidence, SUPERSEDE_CONFIDENCE_CAP]
        if confidence_ceiling is not None:
            ceilings.append(float(confidence_ceiling))
        confidence = min(ceilings)
    attributes = dict(attributes)
    attributes["supersedes"] = old_id
    attributes["valid_from"] = now_iso()
    slop = find_slop(statement, attributes, rules=ENFORCED_RULE_IDS)
    pending_reason = extra_pending_reason or _supersede_route(
        conn, old, actor=actor, provenance=provenance, curator_authored=curator_authored
    )
    with _one_transaction(conn):
        if pending_reason is not None:
            # Nothing is written for a proposal the ledger already carries -- not the
            # proposal, and not the rationale evidence row either. The lookup runs
            # under the same write lock as the append so concurrent identical callers
            # cannot both observe the pair as absent. The successor id is content-and-
            # scope addressed, so an identical re-derivation is a no-op while a
            # genuinely different replacement body for the same target still mints.
            duplicate = undecided_supersede_proposal(
                conn, superseded_id=old_id, successor_id=successor_id
            )
            if duplicate is not None:
                return {
                    "schema_version": SUPERSEDE_SCHEMA_VERSION,
                    "mode": "pending",
                    "deduped": True,
                    "superseded_id": old_id,
                    "successor_id": successor_id,
                    "scope": scope.to_dict(),
                    "confidence": confidence,
                    "pending_reason": pending_reason,
                    "proposal_event_id": str(duplicate["id"]),
                    "proposed_at": str(duplicate["ts"]),
                    "next_step": (
                        "this supersession is already in the pending ledger, undecided; "
                        "an admin decides it with brain.proposal_decide"
                    ),
                }

        evidence_id, evidence_event_id = record_core_v1_evidence(
            conn,
            body=f"Superseding {old_id}. Reason: {rationale}\n\nReplacement claim: {statement}",
            kind="correction",
            scope=scope,
            writer=actor,
            session_id=provenance.client_session_hint or session_id,
            artifact_ref=f"supersede:{old_id}",
        )
        attributes["correction_evidence_id"] = evidence_id
        proposal_event_id = append_core_event(
            conn,
            "compilation_proposed",
            {
                "schema_version": "ocbrain.compilation.v1",
                "subject": {"kind": "belief", "id": successor_id},
                "belief_id": successor_id,
                "belief_type": old.get("belief_type"),
                "body": statement,
                "evidence_ids": list(evidence_ids) if evidence_ids else [evidence_id],
                "scope": scope.to_dict(),
                "confidence": confidence,
                "attributes": attributes,
                "supersede_reason": rationale,
                "supersede_requested_by": actor,
                # Why this one is waiting, in the ledger rather than only in the
                # return value the caller may not keep. An operator reading the
                # pending queue a week later needs the reason beside the
                # proposal. Deliberately a body field and not an attribute:
                # attributes are written onto the belief when the proposal is
                # approved, and "why it once pended" is not a property of the
                # fact.
                "pending_reason": pending_reason,
            },
            writer=actor,
            session_id=provenance.client_session_hint or session_id,
        )
        payload: dict[str, Any] = {
            "schema_version": SUPERSEDE_SCHEMA_VERSION,
            "mode": "pending" if pending_reason else "direct",
            "superseded_id": old_id,
            "successor_id": successor_id,
            "scope": scope.to_dict(),
            "confidence": confidence,
            "correction_evidence_id": evidence_id,
            "evidence_event_id": evidence_event_id,
            "proposal_event_id": proposal_event_id,
        }
        if slop:
            # Reported, never refused. The curator gate already stops slop from
            # being promoted, and refusing here would leave the agent holding a
            # correction it has nowhere to put -- which is the failure this whole
            # primitive exists to end.
            payload["slop_findings"] = [finding.to_dict() for finding in slop]
        if pending_reason is not None:
            payload["pending_reason"] = pending_reason
            payload["next_step"] = (
                "an admin approves this proposal with brain.proposal_decide; "
                f"{old_id} keeps serving until they do"
            )
            return payload
        decision = decide_proposal_v1(
            conn,
            proposal_event_id=proposal_event_id,
            decision="approve",
            actor=actor,
            edited_body=None,
            reason=f"runtime supersede; {rationale}",
            provenance=provenance,
        )
        payload["decision_event_id"] = decision["event_id"]
        payload["correction_event_id"] = decision.get("correction_event_id")
        return payload


def _complete_supersede_pair(
    conn: sqlite3.Connection,
    *,
    proposal_body: dict[str, Any],
    actor: str,
    provenance: Provenance,
) -> dict[str, Any] | None:
    """Retire the belief a just-approved successor was proposed to replace.

    An undecided proposal carrying ``attributes.supersedes`` *is* the pending
    correction -- there is no second table and no new status. Approving one
    therefore has to finish the pair here, in the same transaction and after the
    decision, or the corpus would serve the old belief and its replacement side
    by side and nothing would ever close the gap.
    """
    attributes = proposal_body.get("attributes")
    if not isinstance(attributes, dict):
        return None
    superseded_id = str(attributes.get("supersedes") or "").strip()
    if not superseded_id:
        return None
    target = get_core_v1_belief(conn, superseded_id)
    if target is None:
        return {"supersede_status": f"target not found: {superseded_id}"}
    if target.get("status") != "current" or not target.get("serve"):
        # Retired already, by a competing approval, a retraction, or a
        # tombstone. Saying so beats appending a correction that projects to no
        # change and leaves an audit trail implying one happened.
        return {"supersede_status": f"target already retired: {superseded_id}"}
    correction = correct_v1(
        conn,
        layer="belief",
        target=str(target["canonical_id"]),
        op="supersede",
        body=str(proposal_body.get("supersede_reason") or "").strip() or None,
        actor=actor,
        hard=False,
        successor_id=str(proposal_body.get("belief_id") or ""),
        requested_by=str(proposal_body.get("supersede_requested_by") or "") or None,
        provenance=provenance,
    )
    return {
        "supersede_status": "retired",
        "superseded_id": str(target["canonical_id"]),
        "correction_event_id": correction["event_id"],
    }


def pending_supersede_count(conn: sqlite3.Connection) -> int:
    """Undecided supersede proposals: the raw depth of the queue."""
    row = conn.execute(f"SELECT COUNT(*) {_UNDECIDED_SUPERSEDE_SQL}").fetchone()
    return int(row[0])


def pending_supersede_targets(conn: sqlite3.Connection) -> int:
    """Distinct beliefs the undecided queue is waiting to replace.

    The number an operator actually has to work through. Raw depth hides
    duplication, and duplication is exactly what an unbounded proposal loop
    produces, so a depth that is not reported beside its distinct count is a
    metric that can grow without bound while looking like ordinary backlog.
    """
    row = conn.execute(
        "SELECT COUNT(DISTINCT json_extract(proposal.body_json, '$.attributes.supersedes')) "
        f"{_UNDECIDED_SUPERSEDE_SQL}"
    ).fetchone()
    return int(row[0])


def undecided_supersede_proposal(
    conn: sqlite3.Connection, *, superseded_id: str, successor_id: str
) -> sqlite3.Row | None:
    """The oldest undecided proposal already carrying this exact supersession.

    Keyed on the ``(target, successor)`` pair rather than the target alone: a
    different replacement body for the same belief is a different proposal and
    an operator needs to see both.
    """
    return conn.execute(
        f"SELECT proposal.id AS id, proposal.ts AS ts {_UNDECIDED_SUPERSEDE_SQL} "
        "AND json_extract(proposal.body_json, '$.attributes.supersedes') = ? "
        "AND json_extract(proposal.body_json, '$.belief_id') = ? "
        "ORDER BY proposal.rowid LIMIT 1",
        (superseded_id, successor_id),
    ).fetchone()


def undecided_compilation_proposal(
    conn: sqlite3.Connection, *, belief_id: str, body: str
) -> sqlite3.Row | None:
    """The oldest undecided proposal already carrying this exact claim.

    The pend path has the same shape as the supersede path and the same failure
    mode: a proposal does not change the input that produced it, so a curator
    that pends a claim on Monday re-derives and re-pends it every hour after
    that. Matched on ``(belief_id, body)`` for the same reason
    :func:`undecided_supersede_proposal` matches on the pair -- a different
    statement about the same fact is a second thing to decide, not a duplicate.
    """
    return conn.execute(
        "SELECT proposal.id AS id, proposal.ts AS ts FROM brain_events AS proposal "
        "WHERE proposal.kind='compilation_proposed' "
        "AND json_extract(proposal.body_json, '$.belief_id') = ? "
        "AND json_extract(proposal.body_json, '$.body') = ? "
        f"AND NOT {_PROPOSAL_DECIDED_SQL} "
        "ORDER BY proposal.rowid LIMIT 1",
        (belief_id, body),
    ).fetchone()


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
    result: list[dict[str, Any]] = []
    for row in conn.execute(
        f"SELECT proposal.*, {_PROPOSAL_DECIDED_SQL} AS is_decided FROM brain_events AS proposal "
        "WHERE proposal.kind='compilation_proposed' ORDER BY proposal.rowid DESC LIMIT ?",
        (max(limit * 4, 100),),
    ):
        is_decided = bool(row["is_decided"])
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
    provenance: Provenance | None = None,
) -> dict[str, Any]:
    """Decide one compilation proposal, completing a supersession pair if it is one.

    A rejection changes nothing: the belief under replacement keeps serving and
    the rationale evidence stays in the corpus, curatable, so a refused
    correction is still a recorded observation rather than a discarded one.
    """
    provenance = provenance or EMPTY_PROVENANCE
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
    proposal_body = json.loads(proposal["body_json"])
    if decision in {"approve", "edit"}:
        belief_id = str(proposal_body.get("belief_id") or "")
        reason_blocked = compilation_block_reason(
            conn,
            belief_id,
            proposal_event_seq=int(proposal["event_seq"]),
        )
        if reason_blocked is not None:
            raise PermissionError(f"cannot {decision}: belief is {reason_blocked}: {belief_id}")
    with _one_transaction(conn):
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
            session_id=provenance.client_session_hint,
            project=True,
        )
        result: dict[str, Any] = {
            "event_id": event_id,
            "kind": "compilation_decided",
            "decision": decision,
        }
        if decision in {"approve", "edit"}:
            paired = _complete_supersede_pair(
                conn,
                proposal_body=proposal_body,
                actor=actor,
                provenance=provenance,
            )
            if paired is not None:
                result.update(paired)
        return result


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
        # A pointer row has no inline body; the hash of the content it will
        # actually serve is the window hash already stored on the row. Hashing
        # the empty body instead would issue a handle that can never verify.
        content_hash = (
            str(row["content_hash"])
            if evidence_body_ref(row) is not None
            else sha256_text(str(row["body"]))
        )
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
                content_hash=content_hash,
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
    return result[:MAX_CONTRADICTIONS]


def _packet_contradictions(
    conn: sqlite3.Connection, items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Declared conflicts first, then the cheap packet-local advisory ones.

    The declared pass has never had a writer, so in practice every packet has
    shipped an empty ``contradictions`` list while carrying visibly conflicting
    items. The advisory pass costs one indexed read and, at the twelve-item cap,
    at most sixty-six comparisons -- bounded by the packet, never by the corpus.
    """
    conflicts = _explicit_contradictions(conn, items)
    seen = {
        tuple(sorted((conflict["belief_id"], conflict["other_belief_id"])))
        for conflict in conflicts
    }
    for conflict in _advisory_contradictions(conn, items):
        pair = tuple(sorted((conflict["belief_id"], conflict["other_belief_id"])))
        if pair in seen:
            continue
        seen.add(pair)
        conflicts.append(conflict)
    return conflicts[:MAX_CONTRADICTIONS]


def _advisory_contradictions(
    conn: sqlite3.Connection, items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Flag same-key and near-duplicate pairs inside one packet.

    Two signals, both local to the packet:

    ``duplicate_key``
        Two serving beliefs claiming the same ``attributes.key``. The key is a
        wiki fact's identity and is supposed to be unique across the corpus, so
        two of them in one packet means the reader is being handed two answers
        to one question.
    ``embedding_similarity``
        Mutual cosine at or above the threshold in the local vector sidecar.
        Stands down silently when the sidecar is missing, stale, or unreadable:
        a retrieval must never fail because an optional index is absent.
    """
    if len(items) < 2 or len(items) > MAX_ADVISORY_PAIR_ITEMS:
        return []
    ids = [str(item["id"]) for item in items]
    evidence = {
        str(item["id"]): [str(value) for value in item.get("evidence_ids") or []]
        for item in items
    }
    placeholders = ",".join("?" for _ in ids)
    keys = {
        str(row["belief_id"]): str(row["attribute_key"] or "").strip()
        for row in conn.execute(
            "SELECT belief_id, json_extract(attributes_json, '$.key') AS attribute_key "
            f"FROM current_beliefs WHERE belief_id IN ({placeholders})",
            ids,
        )
    }
    vectors = _packet_embeddings(conn, ids)
    found: list[dict[str, Any]] = []
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            left_key = keys.get(left) or ""
            if left_key and left_key == (keys.get(right) or ""):
                reason = "duplicate_key"
            elif (
                left in vectors
                and right in vectors
                and _cosine(vectors[left], vectors[right]) >= ADVISORY_COSINE_THRESHOLD
            ):
                reason = "embedding_similarity"
            else:
                continue
            found.append(
                {
                    "belief_id": left,
                    "other_belief_id": right,
                    "reason": reason,
                    "advisory": True,
                    "evidence_ids": list(
                        dict.fromkeys([*evidence.get(left, []), *evidence.get(right, [])])
                    )[:8],
                }
            )
    return found


def _packet_embeddings(
    conn: sqlite3.Connection, belief_ids: list[str]
) -> dict[str, list[float]]:
    """Stored belief vectors for one packet, or nothing at all.

    Reads the sidecar directly and never embeds anything, so this makes no
    network call and cannot be the reason a retrieval is slow. Every failure
    path returns an empty mapping: the advisory pass is a hint, and a hint that
    can break a read is worse than no hint.
    """
    core_path = connection_path(conn)
    if core_path is None:
        return {}
    path = vector_db_path(core_path)
    if not path.is_file():
        return {}
    try:
        sidecar = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return {}
    try:
        sidecar.row_factory = sqlite3.Row
        meta = {str(row[0]): str(row[1]) for row in sidecar.execute("SELECT key, value FROM meta")}
        if meta.get("schema_version") != VECTOR_SCHEMA_VERSION:
            return {}
        placeholders = ",".join("?" for _ in belief_ids)
        vectors: dict[str, list[float]] = {}
        for row in sidecar.execute(
            f"SELECT belief_id, vector FROM belief_vectors WHERE belief_id IN ({placeholders})",
            belief_ids,
        ):
            vector = decode_embedding(row["vector"])
            if vector:
                vectors[str(row["belief_id"])] = vector
        return vectors
    except (OSError, sqlite3.Error, ValueError):
        return {}
    finally:
        sidecar.close()


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


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
    coverage["feedback_needed"] = len(packet["items"]) > 0
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
    "CURATOR_SUPERSEDE_WRITER",
    "shared_continuity_scope",
    "bind_retrieval_id_v1",
    "build_context_v1",
    "closeout_v1",
    "correct_v1",
    "decide_proposal_v1",
    "digest_v1",
    "exact_lookup_v1",
    "expand_source_v1",
    "feedback_v1",
    "forget_v1",
    "get_v1",
    "ingest_v1",
    "is_curator_writer",
    "pending_supersede_count",
    "pending_supersede_targets",
    "prepare_retrieval_packet_v1",
    "proposals_v1",
    "record_context_v1",
    "search_v1",
    "supersede_transaction",
    "supersede_v1",
    "undecided_compilation_proposal",
    "undecided_supersede_proposal",
]
