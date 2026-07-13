from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ocbrain.db import now_iso
from ocbrain.events import canonical_json, sha256_text
from ocbrain.ids import stable_id
from ocbrain.retrieve import retrieve
from ocbrain.scope import ScopeContext, ScopeTag, scope_match

CONTEXT_SCHEMA_VERSION = "ocbrain.context.v1"
SOURCE_SCHEMA_VERSION = "ocbrain.source.v1"
MAX_SOURCE_FILE_BYTES = 512_000


def build_context(
    conn: sqlite3.Connection,
    query: str,
    *,
    context: ScopeContext | None = None,
    limit: int = 12,
    cross_scope: bool = False,
    at_ts: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the stable shared-context envelope and its pending source handles.

    Handle rows are returned separately so the MCP seam can persist them in the
    same transaction as its retrieval-use receipt.  A descriptor is useful only
    after :func:`issue_source_handles` succeeds; callers must remove descriptors
    if that write is unavailable.
    """
    resolved = context or ScopeContext()
    raw = retrieve(
        conn,
        query,
        context=resolved,
        limit=limit,
        cross_scope=cross_scope,
        at_ts=at_ts,
    )
    handles: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    for item in raw["items"]:
        item_handles = _handles_for_item(conn, item, context=resolved)
        handles.extend(item_handles)
        if not item_handles:
            unavailable.append(
                {"object_id": str(item["belief_id"]), "reason": "no_expandable_source"}
            )
        normalized.append(
            {
                "id": str(item["belief_id"]),
                "kind": str(item.get("source") or "event_core"),
                "excerpt": str(item.get("body") or ""),
                "scope": dict(item.get("scope") or {}),
                "score": float(item.get("score") or 0.0),
                "relevance": float(item.get("relevance") or 0.0),
                "confidence": float(item.get("confidence") or 0.0),
                "confidence_band": str(item.get("confidence_band") or "unknown"),
                "status": "current",
                "evidence_ids": [str(value) for value in item.get("evidence_ids") or []],
                "sources": [_public_handle(value) for value in item_handles],
            }
        )
    handles = _dedupe_handles(handles)
    envelope = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "query": query,
        "resolved_context": resolved.to_dict(),
        "cross_scope": bool(cross_scope),
        "at_ts": at_ts,
        "items": normalized,
        "contradictions": list(raw.get("contradictions") or []),
        "coverage": {
            "requested_limit": limit,
            "returned": len(normalized),
            "excluded_scope_count": int(raw.get("excluded_count") or 0),
            "excluded_sample": list(raw.get("excluded") or []),
            "estimated_tokens": int(raw.get("token_budget") or 0),
            "source_handle_count": len(handles),
            "unavailable_sources": unavailable,
        },
    }
    return envelope, handles


def issue_source_handles(
    conn: sqlite3.Connection,
    handles: list[dict[str, Any]],
    *,
    retrieval_use_id: str,
) -> None:
    issued_at = now_iso()
    for handle in _dedupe_handles(handles):
        conn.execute(
            """
            INSERT OR IGNORE INTO context_source_handles (
              id, issued_at, retrieval_use_id, object_id, source_kind, uri,
              content_hash, scope_json, locator_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                handle["id"],
                issued_at,
                retrieval_use_id,
                handle["object_id"],
                handle["source_kind"],
                handle.get("uri"),
                handle["content_hash"],
                canonical_json(handle["scope"]),
                canonical_json(handle["locator"]),
            ),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO context_source_handle_issues (
              source_id, retrieval_use_id, issued_at
            )
            VALUES (?, ?, ?)
            """,
            (handle["id"], retrieval_use_id, issued_at),
        )


def remove_unissued_sources(envelope: dict[str, Any], *, reason: str) -> None:
    """Make failure explicit instead of returning unusable capability tokens."""
    count = 0
    unavailable = envelope["coverage"]["unavailable_sources"]
    for item in envelope["items"]:
        for source in item["sources"]:
            unavailable.append(
                {"object_id": item["id"], "source_id": source["id"], "reason": reason}
            )
            count += 1
        item["sources"] = []
    envelope["coverage"]["source_handle_count"] = 0
    envelope["coverage"]["unissued_source_count"] = count


def expand_source(
    conn: sqlite3.Connection,
    source_id: str,
    *,
    context: ScopeContext | None = None,
    max_chars: int = 8_000,
) -> dict[str, Any]:
    """Expand one previously issued source after exact scope and hash checks."""
    row = conn.execute(
        "SELECT * FROM context_source_handles WHERE id = ?",
        (source_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"source handle not found: {source_id}")
    scope = ScopeTag.from_dict(json.loads(row["scope_json"]))
    resolved = context or ScopeContext()
    if scope_match(scope, resolved) == 0:
        raise PermissionError("source scope does not match the supplied context")
    locator = json.loads(row["locator_json"])
    content = _load_source_content(conn, row["source_kind"], locator, context=resolved)
    actual_hash = sha256_text(content)
    if actual_hash != row["content_hash"]:
        raise ValueError("source changed after issuance; request a fresh brain.context handle")
    excerpt, truncated = _bounded_excerpt(content, locator.get("anchor"), max_chars=max_chars)
    issued_by = [
        str(issue["retrieval_use_id"])
        for issue in conn.execute(
            """
            SELECT retrieval_use_id
            FROM context_source_handle_issues
            WHERE source_id = ?
            ORDER BY issued_at ASC, retrieval_use_id ASC
            """,
            (source_id,),
        )
    ]
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "id": row["id"],
        "object_id": row["object_id"],
        "kind": row["source_kind"],
        "uri": row["uri"],
        "scope": scope.to_dict(),
        "content_hash": row["content_hash"],
        "hash_verified": True,
        "content": excerpt,
        "truncated": truncated,
        "characters": len(excerpt),
        "issued_at": row["issued_at"],
        "origin_retrieval_use_id": row["retrieval_use_id"],
        "issued_by_retrieval_use_ids": issued_by,
    }


def _handles_for_item(
    conn: sqlite3.Connection,
    item: dict[str, Any],
    *,
    context: ScopeContext,
) -> list[dict[str, Any]]:
    object_id = str(item["belief_id"])
    scope = ScopeTag.from_dict(item.get("scope")).to_dict()
    handles: list[dict[str, Any]] = []
    for artifact in item.get("artifact_refs") or []:
        relative = artifact.get("path") if isinstance(artifact, dict) else None
        digest = artifact.get("sha256") if isinstance(artifact, dict) else None
        if not relative or not digest or not context.repo:
            continue
        locator = {
            "repo": str(Path(context.repo).expanduser().resolve()),
            "path": str(relative),
            "anchor": str(item.get("body") or "").partition("\n")[2][:300],
        }
        handles.append(
            _handle(
                object_id=object_id,
                source_kind="repo_file",
                uri=str(relative),
                content_hash=str(digest),
                scope=scope,
                locator=locator,
            )
        )
    for evidence_id in item.get("evidence_ids") or []:
        handle = _evidence_handle(conn, str(evidence_id), object_id=object_id, fallback_scope=scope)
        if handle is not None:
            handles.append(handle)
    handles = [
        handle
        for handle in handles
        if scope_match(ScopeTag.from_dict(handle["scope"]), context) > 0
    ]
    if handles:
        return handles
    belief = conn.execute(
        "SELECT body, scope_type, scope_id, visibility, egress_policy FROM current_beliefs "
        "WHERE belief_id = ?",
        (object_id,),
    ).fetchone()
    if belief is not None:
        body = str(belief["body"])
        belief_scope = {
            "scope_type": belief["scope_type"],
            "scope_id": belief["scope_id"],
            "visibility": belief["visibility"],
            "egress_policy": belief["egress_policy"],
            "provenance": "projection",
        }
        handles = [
            _handle(
                object_id=object_id,
                source_kind="compiled_belief",
                uri=f"ocbrain://belief/{object_id}",
                content_hash=sha256_text(body),
                scope=belief_scope,
                locator={"belief_id": object_id, "anchor": str(item.get("body") or "")[:300]},
            )
        ]
        return [
            handle
            for handle in handles
            if scope_match(ScopeTag.from_dict(handle["scope"]), context) > 0
        ]
    # FTS rows can be self-contained snippets without an independently
    # addressable source. Persist the immutable, hashed retrieval snapshot so
    # expansion remains capability-bound rather than accepting arbitrary paths.
    body = str(item.get("body") or "")
    if body:
        handles = [
            _handle(
                object_id=object_id,
                source_kind="retrieval_snapshot",
                uri=f"ocbrain://retrieval-object/{object_id}",
                content_hash=sha256_text(body),
                scope=scope,
                locator={"content": body, "anchor": body[:300]},
            )
        ]
        return [
            handle
            for handle in handles
            if scope_match(ScopeTag.from_dict(handle["scope"]), context) > 0
        ]
    return []


def _evidence_handle(
    conn: sqlite3.Connection,
    evidence_id: str,
    *,
    object_id: str,
    fallback_scope: dict[str, Any],
) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM evidence WHERE id = ?", (evidence_id,)).fetchone()
    if row is not None:
        claim = str(row["claim"])
        scope = dict(fallback_scope)
        if row["privacy_scope"] == "private":
            if row["project"]:
                scope = ScopeTag(
                    "project",
                    f"project:{row['project']}",
                    visibility="confidential",
                    egress_policy="local_only",
                    provenance="relational_evidence",
                ).to_dict()
            elif scope.get("scope_type") == "global":
                # A private row with no compatible project cannot inherit a
                # globally expandable handle from its compiled parent.
                scope = ScopeTag(
                    "legacy_unscoped",
                    f"private:{evidence_id}",
                    visibility="confidential",
                    egress_policy="local_only",
                    provenance="quarantined",
                ).to_dict()
            else:
                scope["visibility"] = "confidential"
                scope["egress_policy"] = "local_only"
        return _handle(
            object_id=object_id,
            source_kind="relational_evidence",
            uri=row["source_uri"] or row["artifact_uri"] or f"ocbrain://evidence/{evidence_id}",
            content_hash=sha256_text(claim),
            scope=scope,
            locator={"evidence_id": evidence_id, "anchor": claim[:300]},
        )
    event = conn.execute(
        """
        SELECT id, body_json
        FROM brain_events
        WHERE kind = 'evidence_recorded'
          AND json_extract(body_json, '$.evidence_id') = ?
        ORDER BY rowid DESC
        LIMIT 1
        """,
        (evidence_id,),
    ).fetchone()
    if event is None:
        return None
    event_body = json.loads(event["body_json"])
    body = str(event_body.get("body") or "")
    event_scope = ScopeTag.from_dict(event_body.get("scope")).to_dict()
    return _handle(
        object_id=object_id,
        source_kind="event_evidence",
        uri=event_body.get("artifact_ref") or f"ocbrain://event/{event['id']}",
        content_hash=sha256_text(body),
        scope=event_scope,
        locator={"event_id": event["id"], "evidence_id": evidence_id, "anchor": body[:300]},
    )


def _handle(
    *,
    object_id: str,
    source_kind: str,
    uri: str | None,
    content_hash: str,
    scope: dict[str, Any],
    locator: dict[str, Any],
) -> dict[str, Any]:
    handle_id = stable_id(
        "src",
        object_id,
        source_kind,
        uri or "",
        content_hash,
        canonical_json(scope),
    )
    return {
        "id": handle_id,
        "object_id": object_id,
        "source_kind": source_kind,
        "uri": uri,
        "content_hash": content_hash,
        "scope": scope,
        "locator": locator,
    }


def _public_handle(handle: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": handle["id"],
        "kind": handle["source_kind"],
        "uri": handle.get("uri"),
        "content_hash": handle["content_hash"],
    }


def _dedupe_handles(handles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list({str(handle["id"]): handle for handle in handles}.values())


def _load_source_content(
    conn: sqlite3.Connection,
    source_kind: str,
    locator: dict[str, Any],
    *,
    context: ScopeContext,
) -> str:
    if source_kind == "repo_file":
        root = Path(str(locator["repo"])).expanduser().resolve()
        if not context.repo or Path(context.repo).expanduser().resolve() != root:
            raise PermissionError("source repository does not match the supplied context")
        path = (root / str(locator["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PermissionError("source path escapes its issued repository") from exc
        payload = path.read_bytes()
        if len(payload) > MAX_SOURCE_FILE_BYTES:
            payload = payload[:MAX_SOURCE_FILE_BYTES]
        return payload.decode("utf-8", errors="replace")
    if source_kind == "event_evidence":
        row = conn.execute(
            "SELECT body_json FROM brain_events WHERE id = ? AND kind = 'evidence_recorded'",
            (locator["event_id"],),
        ).fetchone()
        if row is None:
            raise ValueError("issued event evidence no longer exists")
        return str(json.loads(row["body_json"]).get("body") or "")
    if source_kind == "relational_evidence":
        row = conn.execute(
            "SELECT claim FROM evidence WHERE id = ?",
            (locator["evidence_id"],),
        ).fetchone()
        if row is None:
            raise ValueError("issued evidence no longer exists")
        return str(row["claim"])
    if source_kind == "compiled_belief":
        row = conn.execute(
            "SELECT body FROM current_beliefs WHERE belief_id = ?",
            (locator["belief_id"],),
        ).fetchone()
        if row is None:
            raise ValueError("issued belief no longer exists")
        return str(row["body"])
    if source_kind == "retrieval_snapshot":
        return str(locator["content"])
    raise ValueError(f"unsupported issued source kind: {source_kind}")


def _bounded_excerpt(content: str, anchor: Any, *, max_chars: int) -> tuple[str, bool]:
    if len(content) <= max_chars:
        return content, False
    anchor_text = str(anchor or "").strip()
    index = content.find(anchor_text[:200]) if anchor_text else -1
    if index < 0:
        index = 0
    start = max(0, min(index - max_chars // 4, len(content) - max_chars))
    return content[start : start + max_chars], True
