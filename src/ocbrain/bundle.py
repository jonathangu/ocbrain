"""Explicit, file-only evidence exchange for the strict OCBrain v1 core.

Bundles contain redacted evidence projections only.  They never contain beliefs,
retrieval receipts, closeouts, credentials, network instructions, or remote IDs
that are authoritative on import.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from ocbrain.core_v1 import (
    CORE_V1_EVENT_SCHEMA,
    append_core_event,
    canonical_json,
    is_core_v1,
    project_core_v1,
    sha256_text,
)
from ocbrain.db import now_iso
from ocbrain.egress import record_egress_audit
from ocbrain.ids import stable_id
from ocbrain.scope import ScopeContext, ScopeTag, egress_allowed
from ocbrain.text import find_probable_secret_leaks, redact_secrets

BUNDLE_SCHEMA = "ocbrain.bundle.v1"
BUNDLE_IMPORT_PROVENANCE = "bundle_import"
MAX_BUNDLE_BYTES = 20_000_000
MAX_BUNDLE_ITEMS = 1_000
MAX_ITEM_BODY_CHARS = 200_000
_ENVELOPE_KEYS = {"schema_version", "created_at", "item_count", "payload_hash", "items"}
_ITEM_KEYS = {"source_evidence_id", "kind", "body", "scope"}
_SCOPE_KEYS = {"scope_type", "scope_id", "visibility", "egress_policy", "provenance"}


class BundleError(ValueError):
    """Base class for a refused or malformed bundle operation."""


class BundleExportError(BundleError):
    """Raised before an unauthorized or unsafe export can publish a file."""


class BundleImportError(BundleError):
    """Raised before malformed bundle content can append any event."""


def export_bundle(
    conn: sqlite3.Connection,
    output_path: Path,
    *,
    evidence_ids: list[str],
    context: ScopeContext,
    approve_egress: bool = False,
) -> dict[str, Any]:
    """Export explicitly selected strict-v1 evidence to one fresh local file."""
    if not is_core_v1(conn):
        raise BundleExportError("export-bundle requires an initialized strict-v1 core")
    if conn.in_transaction:
        raise BundleExportError("export-bundle requires a connection without an open transaction")
    selected_ids = sorted({item.strip() for item in evidence_ids if item.strip()})
    if not selected_ids:
        raise BundleExportError("at least one --evidence-id is required")
    if len(selected_ids) > MAX_BUNDLE_ITEMS:
        raise BundleExportError(f"bundle exceeds {MAX_BUNDLE_ITEMS} selected items")

    destination = _absolute_destination(output_path)
    if os.path.lexists(destination):
        raise BundleExportError(f"refusing to overwrite existing bundle path: {destination}")
    if not destination.parent.is_dir():
        raise BundleExportError(f"bundle parent directory does not exist: {destination.parent}")

    published_identity: tuple[int, int] | None = None
    conn.execute("BEGIN IMMEDIATE")
    try:
        items, audit_items = _export_items(
            conn,
            selected_ids,
            context=context,
            approve_egress=approve_egress,
        )
        payload_hash = sha256_text(canonical_json(items))
        bundle = {
            "schema_version": BUNDLE_SCHEMA,
            "created_at": now_iso(),
            "item_count": len(items),
            "payload_hash": payload_hash,
            "items": items,
        }
        audit = {
            "target": "human_export",
            "context": context.to_dict(),
            "query": "explicit evidence bundle",
            "included": audit_items,
            "rejected": [],
            "payload_hash": payload_hash,
        }
        audit_id = record_egress_audit(conn, audit)
        published_identity = _atomic_write_new(destination, canonical_json(bundle) + "\n")
        conn.commit()
    except Exception:
        conn.rollback()
        if published_identity is not None:
            _unlink_if_same(destination, published_identity)
        raise

    return {
        "schema_version": BUNDLE_SCHEMA,
        "path": str(destination),
        "item_count": len(items),
        "payload_hash": payload_hash,
        "egress_audit_id": audit_id,
        "mode": "owner_only",
    }


def import_bundle(
    conn: sqlite3.Connection | None,
    bundle_path: Path,
    *,
    project: str,
    apply: bool = False,
) -> dict[str, Any]:
    """Validate a bundle, defaulting to a DB-free dry run; append only on apply."""
    project_name = project.strip()
    if not project_name:
        raise BundleImportError("a non-empty destination project is required")
    bundle = load_bundle(bundle_path)
    prepared, duplicate_items = _prepare_import_items(bundle, project=project_name)
    base = {
        "schema_version": BUNDLE_SCHEMA,
        "payload_hash": bundle["payload_hash"],
        "source_item_count": bundle["item_count"],
        "unique_item_count": len(prepared),
        "duplicate_items": duplicate_items,
        "destination_scope": {
            "scope_type": "project",
            "scope_id": f"project:{project_name}",
            "visibility": "confidential",
            "egress_policy": "local_only",
            "provenance": BUNDLE_IMPORT_PROVENANCE,
        },
        "local_evidence_ids": [item["evidence_id"] for item in prepared],
    }
    if not apply:
        return base | {
            "dry_run": True,
            "database_touched": False,
            "would_append": len(prepared),
        }
    if conn is None or not is_core_v1(conn):
        raise BundleImportError("--apply requires an initialized strict-v1 core")
    if conn.in_transaction:
        raise BundleImportError("import-bundle requires a connection without an open transaction")

    appended = 0
    deduped = duplicate_items
    conn.execute("BEGIN IMMEDIATE")
    try:
        for item in prepared:
            if _evidence_event_exists(conn, item["evidence_id"]):
                deduped += 1
                continue
            append_core_event(
                conn,
                "evidence_recorded",
                item["event_body"],
                writer="ocbrain-bundle-import",
                project=False,
            )
            appended += 1
        projection = project_core_v1(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return base | {
        "dry_run": False,
        "applied": True,
        "appended": appended,
        "deduped": deduped,
        "projection": projection,
    }


def load_bundle(path: Path) -> dict[str, Any]:
    """Read and fully validate an untrusted bundle before any DB transaction."""
    source = path.expanduser()
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise BundleImportError(f"cannot read bundle: {exc}") from exc
    if size > MAX_BUNDLE_BYTES:
        raise BundleImportError(f"bundle exceeds {MAX_BUNDLE_BYTES} bytes")
    try:
        raw = source.read_text(encoding="utf-8")
        value = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, BundleImportError) as exc:
        if isinstance(exc, BundleImportError):
            raise
        raise BundleImportError(f"invalid bundle JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleImportError("bundle root must be an object")
    if set(value) != _ENVELOPE_KEYS:
        raise BundleImportError("bundle envelope keys do not match the schema")
    if value.get("schema_version") != BUNDLE_SCHEMA:
        raise BundleImportError(f"unsupported bundle schema: {value.get('schema_version')!r}")
    _validate_timestamp(value.get("created_at"))
    count = value.get("item_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise BundleImportError("item_count must be a positive integer")
    if count > MAX_BUNDLE_ITEMS:
        raise BundleImportError(f"bundle exceeds {MAX_BUNDLE_ITEMS} items")
    items = value.get("items")
    if not isinstance(items, list) or len(items) != count:
        raise BundleImportError("item_count does not match items")
    for index, item in enumerate(items):
        _validate_item(item, index=index)
    payload_hash = value.get("payload_hash")
    if not _is_sha256(payload_hash):
        raise BundleImportError("payload_hash must be a lowercase SHA-256 digest")
    expected = sha256_text(canonical_json(items))
    if payload_hash != expected:
        raise BundleImportError("bundle payload hash mismatch")
    return value


def _export_items(
    conn: sqlite3.Connection,
    evidence_ids: list[str],
    *,
    context: ScopeContext,
    approve_egress: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exported: list[dict[str, Any]] = []
    audit_items: list[dict[str, Any]] = []
    refused: list[dict[str, str]] = []
    for evidence_id in evidence_ids:
        row = conn.execute(
            "SELECT * FROM evidence_objects WHERE evidence_id=?",
            (evidence_id,),
        ).fetchone()
        if row is None:
            refused.append({"evidence_id": evidence_id, "reason": "not_found"})
            continue
        scope = ScopeTag(
            scope_type=str(row["scope_type"]),
            scope_id=str(row["scope_id"]),
            visibility=str(row["visibility"]),
            egress_policy=str(row["egress_policy"]),
            provenance=str(row["scope_provenance"]),
        )
        allowed, reason = egress_allowed(scope, context, "human_export")
        if not allowed:
            refused.append({"evidence_id": evidence_id, "reason": reason})
            continue
        if scope.egress_policy == "approval_required" and not approve_egress:
            refused.append({"evidence_id": evidence_id, "reason": "approval_required"})
            continue
        raw_body = str(row["body"])
        if not raw_body.strip() or len(raw_body) > MAX_ITEM_BODY_CHARS:
            refused.append({"evidence_id": evidence_id, "reason": "invalid_body_size"})
            continue
        body = redact_secrets(raw_body)
        residue = find_probable_secret_leaks(body)
        if residue:
            refused.append(
                {"evidence_id": evidence_id, "reason": f"secret_residue:{','.join(residue)}"}
            )
            continue
        item = {
            "source_evidence_id": evidence_id,
            "kind": str(row["kind"]),
            "body": body,
            "scope": scope.to_dict(),
        }
        exported.append(item)
        audit_items.append(
            {
                "evidence_id": evidence_id,
                "scope": scope.to_dict(),
                "reason": (
                    "approved_export" if scope.egress_policy == "approval_required" else reason
                ),
            }
        )
    if refused:
        detail = "; ".join(f"{item['evidence_id']}:{item['reason']}" for item in refused)
        raise BundleExportError(f"bundle export refused selected evidence: {detail}")
    return exported, audit_items


def _prepare_import_items(
    bundle: dict[str, Any], *, project: str
) -> tuple[list[dict[str, Any]], int]:
    scope = ScopeTag(
        "project",
        f"project:{project}",
        visibility="confidential",
        egress_policy="local_only",
        provenance=BUNDLE_IMPORT_PROVENANCE,
    )
    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0
    for item in bundle["items"]:
        body = redact_secrets(item["body"])
        residue = find_probable_secret_leaks(body)
        if residue:
            raise BundleImportError(f"item retains probable secret after redaction: {residue}")
        body_hash = sha256_text(body)
        artifact_ref = f"ocbrain-bundle:sha256:{body_hash}"
        evidence_id = stable_id(
            "evd",
            body,
            "bundle_import",
            artifact_ref,
            scope.scope_id,
        )
        if evidence_id in seen:
            duplicates += 1
            continue
        seen.add(evidence_id)
        prepared.append(
            {
                "evidence_id": evidence_id,
                "event_body": {
                    "schema_version": CORE_V1_EVENT_SCHEMA,
                    "subject": {"kind": "evidence", "id": evidence_id},
                    "evidence_id": evidence_id,
                    "kind": "bundle_import",
                    "body": body,
                    "artifact_ref": artifact_ref,
                    "scope": scope.to_dict(),
                    "bundle_provenance": {
                        "schema_version": BUNDLE_SCHEMA,
                        "payload_hash": bundle["payload_hash"],
                        "source_evidence_id": item["source_evidence_id"],
                        "source_kind": item["kind"],
                    },
                },
            }
        )
    return prepared, duplicates


def _validate_item(item: Any, *, index: int) -> None:
    if not isinstance(item, dict) or set(item) != _ITEM_KEYS:
        raise BundleImportError(f"item {index} keys do not match the schema")
    source_id = item.get("source_evidence_id")
    if not isinstance(source_id, str) or not source_id or len(source_id) > 512:
        raise BundleImportError(f"item {index} has invalid source_evidence_id")
    kind = item.get("kind")
    if not isinstance(kind, str) or not kind.strip() or len(kind) > 256:
        raise BundleImportError(f"item {index} has invalid kind")
    body = item.get("body")
    if not isinstance(body, str) or not body.strip() or len(body) > MAX_ITEM_BODY_CHARS:
        raise BundleImportError(f"item {index} has invalid body")
    scope = item.get("scope")
    if not isinstance(scope, dict) or set(scope) != _SCOPE_KEYS:
        raise BundleImportError(f"item {index} has invalid source scope")
    try:
        ScopeTag.from_dict(scope)
    except ValueError as exc:
        raise BundleImportError(f"item {index} has invalid source scope: {exc}") from exc


def _validate_timestamp(value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise BundleImportError("created_at must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BundleImportError("created_at must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BundleImportError("created_at must include a timezone offset")


def _evidence_event_exists(conn: sqlite3.Connection, evidence_id: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM brain_events WHERE kind='evidence_recorded' "
            "AND json_extract(body_json, '$.evidence_id')=? LIMIT 1",
            (evidence_id,),
        ).fetchone()
        is not None
    )


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleImportError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _absolute_destination(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.parent.resolve() / expanded.name


def _atomic_write_new(path: Path, text: str) -> tuple[int, int]:
    """Publish fully written bytes at ``path`` without any overwrite race."""
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        identity = temporary.stat()
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise BundleExportError(f"refusing to overwrite existing bundle path: {path}") from exc
        os.chmod(path, 0o600, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return identity.st_dev, identity.st_ino
    finally:
        if fd >= 0:
            os.close(fd)
        temporary.unlink(missing_ok=True)


def _unlink_if_same(path: Path, identity: tuple[int, int]) -> None:
    try:
        stat = path.stat(follow_symlinks=False)
        if (stat.st_dev, stat.st_ino) == identity:
            path.unlink()
    except OSError:
        return


__all__ = [
    "BUNDLE_SCHEMA",
    "BundleError",
    "BundleExportError",
    "BundleImportError",
    "export_bundle",
    "import_bundle",
    "load_bundle",
]
