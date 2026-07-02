"""File-based brain sharing between machines: export/import evidence bundles.

Bundles are plain JSON files moved by hand (USB stick, AirDrop, rsync of a
file the human already trusts) — this module contains no network code. Export
runs through the human_export egress gate and secret redaction; import appends
scoped evidence_recorded events only, so recipients compile beliefs locally
(evidence before belief).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ocbrain.db import now_iso
from ocbrain.egress import evidence_events, record_egress_audit
from ocbrain.events import append_event, canonical_json, sha256_text
from ocbrain.scope import ScopeContext, ScopeTag, egress_allowed
from ocbrain.text import redact_secrets

BUNDLE_SCHEMA_VERSION = "ocbrain.bundle.v1"
EXPORT_WRITER = "ocbrain-export"
EXPORT_TARGET = "human_export"
# Imported evidence is never more egress-permissive than approval_required: a
# friend's hosted_ok must not silently re-egress from the recipient's machine.
IMPORT_EGRESS_CAP = "approval_required"
_CONTEXT_SCOPE_TYPES = {"project", "repo", "client", "task", "session"}


class BundleExportError(ValueError):
    """Export refused: the selection would egress prohibited evidence."""


class BundleImportError(ValueError):
    """Import refused: the bundle is malformed, unsupported, or tampered."""


def export_bundle(
    conn: sqlite3.Connection,
    *,
    db_path: Path | str,
    label: str = "ocbrain",
    scopes: list[tuple[str, str]] | None = None,
    query: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Select exportable evidence, gate it, and build a shareable bundle.

    Raises BundleExportError if any selected item has egress_policy
    'prohibited' (checked before the limit is applied — a limit must never
    hide a refusal). local_only items are skipped and reported, never
    exported. Records an egress_audits row for the successful export.
    """
    selected = [(str(scope_type), str(scope_id)) for scope_type, scope_id in scopes or []]
    included: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    prohibited: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in evidence_events(conn):
        body = json.loads(row["body_json"])
        scope = ScopeTag.from_dict(body.get("scope"))
        if selected and (scope.scope_type, scope.scope_id) not in selected:
            continue
        text = str(body.get("body") or "")
        if query and query.lower() not in text.lower():
            continue
        evidence_id = str(body.get("evidence_id") or "")
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        allowed, reason = export_decision(scope)
        report = {"evidence_id": evidence_id, "scope": scope.to_dict(), "reason": reason}
        if scope.egress_policy == "prohibited":
            prohibited.append(report)
        elif not allowed:
            skipped.append(report)
        else:
            included.append(bundle_evidence_item(row, body, scope))
    if prohibited:
        sample = ", ".join(item["evidence_id"] for item in prohibited[:5])
        raise BundleExportError(
            f"refusing to export: {len(prohibited)} evidence item(s) carry "
            f"egress_policy='prohibited' (e.g. {sample}); narrow the selection "
            "with --scope-type/--scope-id or --query so they are excluded"
        )
    if limit is not None:
        included = included[:limit]
    evidence = sorted(included, key=canonical_json)
    payload_hash = bundle_payload_hash(evidence)
    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at": now_iso(),
        "origin": {
            "writer": EXPORT_WRITER,
            "hostname_free_label": label,
            "db_path_hash": sha256_text(str(db_path)),
        },
        "evidence": evidence,
        "count": len(evidence),
        "payload_hash": payload_hash,
    }
    audit_id = record_egress_audit(
        conn,
        {
            "target": EXPORT_TARGET,
            "context": {"bundle_label": label, "scopes": selected},
            "query": query,
            "included": [
                {"evidence_id": item["evidence_id"], "scope": item["scope"]}
                for item in evidence[:50]
            ],
            "rejected": skipped[:50],
            "payload_hash": payload_hash,
        },
    )
    return {
        "bundle": bundle,
        "count": len(evidence),
        "skipped": skipped,
        "skipped_count": len(skipped),
        "audit_id": audit_id,
        "payload_hash": payload_hash,
    }


def export_decision(scope: ScopeTag) -> tuple[bool, str]:
    """Apply the human_export egress gate to one explicitly selected item.

    Reuses egress_allowed with a context synthesized to match the item's own
    scope: bundle export selects scopes explicitly, so context matching must
    not veto — the scope's egress_policy is the deciding gate.
    """
    allowed, reason = egress_allowed(scope, matching_context(scope), EXPORT_TARGET)
    if not allowed and reason == "scope_mismatch":
        # Scope types with no ScopeContext field (personal_finance,
        # legacy_unscoped) can never context-match; fall through to the same
        # policy gate egress_allowed applies for human_export.
        if scope.egress_policy in {"hosted_ok", "approval_required"}:
            return True, "allowed_export"
        return False, f"egress_policy:{scope.egress_policy}"
    return allowed, reason


def matching_context(scope: ScopeTag) -> ScopeContext:
    if scope.scope_type in _CONTEXT_SCOPE_TYPES:
        _, _, name = scope.scope_id.partition(":")
        return ScopeContext(**{scope.scope_type: name or scope.scope_id})
    return ScopeContext()


def bundle_evidence_item(
    row: sqlite3.Row, body: dict[str, Any], scope: ScopeTag
) -> dict[str, Any]:
    return {
        "evidence_id": str(body.get("evidence_id")),
        "kind": str(body.get("kind") or "observation"),
        "body": redact_secrets(str(body.get("body") or "")),
        "artifact_ref": body.get("artifact_ref"),
        "scope": {
            "scope_type": scope.scope_type,
            "scope_id": scope.scope_id,
            "visibility": scope.visibility,
            "egress_policy": scope.egress_policy,
        },
        "writer": row["writer"],
        "ts": row["ts"],
    }


def bundle_payload_hash(evidence: list[Any]) -> str:
    """Order-independent content hash over the bundle's evidence array."""
    return sha256_text(canonical_json(sorted(evidence, key=canonical_json)))


def load_bundle(path: Path | str) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise BundleImportError(f"cannot read bundle {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BundleImportError("bundle must be a JSON object")
    schema_version = data.get("schema_version")
    if schema_version != BUNDLE_SCHEMA_VERSION:
        raise BundleImportError(
            f"unsupported schema_version {schema_version!r}; expected {BUNDLE_SCHEMA_VERSION!r}"
        )
    evidence = data.get("evidence")
    if not isinstance(evidence, list):
        raise BundleImportError("bundle field 'evidence' must be a list")
    expected = data.get("payload_hash")
    actual = bundle_payload_hash(evidence)
    if actual != expected:
        raise BundleImportError(
            f"payload_hash mismatch: bundle declares {expected!r} but evidence hashes "
            f"to {actual!r}; the bundle was modified after export"
        )
    return data


def import_bundle(
    conn: sqlite3.Connection,
    path: Path | str,
    *,
    dry_run: bool = False,
    actor: str = "human:jonathan",
) -> dict[str, Any]:
    """Append a bundle's evidence as scoped evidence_recorded events.

    Dedups on the content-derived evidence id already present in the event
    core; caps egress_policy at approval_required; never imports beliefs —
    the recipient compiles locally via event-dream + event-decide. With
    dry_run=True nothing is written.
    """
    bundle = load_bundle(path)
    origin = bundle.get("origin")
    origin_label = "ocbrain"
    if isinstance(origin, dict) and origin.get("hostname_free_label"):
        origin_label = str(origin["hostname_free_label"])
    writer = f"import:{origin_label}"
    existing = existing_evidence_ids(conn)
    new_items: list[dict[str, Any]] = []
    duplicates = 0
    skipped: list[dict[str, Any]] = []
    imported_event_ids: list[str] = []
    for index, item in enumerate(bundle["evidence"]):
        if not isinstance(item, dict) or not item.get("evidence_id") or not item.get("body"):
            skipped.append({"index": index, "reason": "invalid_item"})
            continue
        evidence_id = str(item["evidence_id"])
        try:
            scope = ScopeTag.from_dict(item.get("scope"))
        except ValueError as exc:
            skipped.append({"evidence_id": evidence_id, "reason": f"invalid_scope:{exc}"})
            continue
        if scope.egress_policy == "prohibited":
            # A well-formed exporter refuses prohibited items, so their
            # presence signals a hand-built bundle; never ingest them.
            skipped.append({"evidence_id": evidence_id, "reason": "egress_policy:prohibited"})
            continue
        if evidence_id in existing:
            duplicates += 1
            continue
        capped_scope = cap_scope_for_import(scope)
        plan = {
            "evidence_id": evidence_id,
            "kind": str(item.get("kind") or "observation"),
            "scope": capped_scope.to_dict(),
            "original_egress_policy": scope.egress_policy,
        }
        if not dry_run:
            imported_event_ids.append(
                append_event(
                    conn,
                    "evidence_recorded",
                    {
                        "evidence_id": evidence_id,
                        "kind": plan["kind"],
                        "body": str(item["body"]),
                        "artifact_ref": item.get("artifact_ref"),
                        "scope": capped_scope.to_dict(),
                    },
                    writer=writer,
                    session_id=actor,
                )
            )
        existing.add(evidence_id)
        new_items.append(plan)
    return {
        "dry_run": dry_run,
        "origin_label": origin_label,
        "writer": writer,
        "actor": actor,
        "payload_hash": bundle["payload_hash"],
        "counts": {
            "new": len(new_items),
            "duplicates": duplicates,
            "skipped": len(skipped),
        },
        "skipped": skipped,
        "sample": new_items[:20],
        "imported_event_ids": imported_event_ids,
    }


def cap_scope_for_import(scope: ScopeTag) -> ScopeTag:
    """Preserve the origin scope except for the egress cap."""
    if scope.egress_policy != "hosted_ok":
        return scope
    return ScopeTag(
        scope_type=scope.scope_type,
        scope_id=scope.scope_id,
        visibility=scope.visibility,
        egress_policy=IMPORT_EGRESS_CAP,
        provenance=scope.provenance,
    )


def existing_evidence_ids(conn: sqlite3.Connection) -> set[str]:
    ids: set[str] = set()
    for row in evidence_events(conn):
        evidence_id = json.loads(row["body_json"]).get("evidence_id")
        if evidence_id:
            ids.add(str(evidence_id))
    return ids


def write_bundle_file(path: Path, bundle: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
