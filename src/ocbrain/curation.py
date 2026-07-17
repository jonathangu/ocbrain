"""Explicit, source-hash-verified curated-memory ingestion.

Curated manifests are small human-reviewable inputs.  Applying one appends
evidence, proposal, and approval events; it never writes a projection row
directly and is idempotent when the current fact and source hashes match.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from ocbrain.core_v1 import (
    append_core_event,
    canonical_json,
    get_core_v1_belief,
    is_core_v1,
    record_core_v1_evidence,
    sha256_text,
)
from ocbrain.mcp_v1 import decide_proposal_v1
from ocbrain.scope import ScopeTag

CURATED_MANIFEST_SCHEMA = "ocbrain.curated-memory.v1"


def apply_curated_manifest(
    conn: sqlite3.Connection,
    manifest_path: Path,
    *,
    actor: str = "human-curated:operator",
    allow_hosted_egress: bool = False,
) -> dict[str, Any]:
    if not is_core_v1(conn):
        raise ValueError("curated manifests require an OCBrain v1 core")
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CURATED_MANIFEST_SCHEMA:
        raise ValueError("unsupported curated manifest schema")
    project = _required_text(manifest, "project")
    manifest_sha256 = _file_sha256(manifest_path)
    sources = _verify_sources(manifest.get("sources"), base_dir=manifest_path.parent)
    facts = manifest.get("facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError("manifest facts must be a non-empty list")
    seen_fact_ids: set[str] = set()
    hosted_fact_ids: list[str] = []
    for raw in facts:
        if not isinstance(raw, dict):
            raise ValueError("each curated fact must be an object")
        fact_id = _required_text(raw, "id")
        if fact_id in seen_fact_ids:
            raise ValueError(f"duplicate curated fact id: {fact_id}")
        seen_fact_ids.add(fact_id)
        visibility = str(raw.get("visibility") or "internal")
        egress_policy = str(raw.get("egress_policy") or "local_only")
        if egress_policy == "hosted_ok":
            if visibility in {"confidential", "secret"}:
                raise ValueError(
                    f"fact {fact_id} cannot combine hosted_ok with {visibility} visibility"
                )
            hosted_fact_ids.append(fact_id)
    if hosted_fact_ids and not allow_hosted_egress:
        joined = ", ".join(hosted_fact_ids)
        raise ValueError(
            "manifest authorizes hosted-model delivery for facts "
            f"{joined}; review their exact bodies and pass --allow-hosted-egress"
        )

    prepared_facts = _prepare_facts(facts, project=project, sources=sources)
    applied: list[str] = []
    unchanged: list[str] = []
    conn.execute("SAVEPOINT curated_manifest_apply")
    try:
        for fact in prepared_facts:
            body = fact["body"]
            referenced = fact["referenced"]
            scope = fact["scope"]
            belief_id = fact["belief_id"]
            confidence = fact["confidence"]
            attributes = fact["attributes"]
            current = get_core_v1_belief(conn, belief_id)
            if (
                current is not None
                and current.get("status") == "current"
                and bool(current.get("serve"))
                and str(current.get("body")) == body
                and current.get("belief_type") == "curated_fact"
                and float(current.get("confidence") or 0.0) == confidence
                and current.get("attributes") == attributes
                and current.get("scope") == scope.to_dict()
            ):
                unchanged.append(belief_id)
                continue

            evidence_ids: list[str] = []
            for source in referenced:
                evidence_id, _event_id = record_core_v1_evidence(
                    conn,
                    body=body,
                    kind="curated_source_attestation",
                    scope=scope,
                    writer=actor,
                    artifact_ref=str(source["path"]),
                )
                evidence_ids.append(evidence_id)
            proposal_id = append_core_event(
                conn,
                "compilation_proposed",
                {
                    "schema_version": "ocbrain.compilation.v1",
                    "subject": {"kind": "belief", "id": belief_id},
                    "belief_id": belief_id,
                    "belief_type": "curated_fact",
                    "body": body,
                    "evidence_ids": list(dict.fromkeys(evidence_ids)),
                    "scope": scope.to_dict(),
                    "confidence": confidence,
                    "reward_band": "strong",
                    "attributes": attributes,
                },
                writer=actor,
            )
            decide_proposal_v1(
                conn,
                proposal_event_id=proposal_id,
                decision="approve",
                actor=actor,
                edited_body=None,
                reason="source-hash-verified curated manifest",
            )
            applied.append(belief_id)
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT curated_manifest_apply")
        conn.execute("RELEASE SAVEPOINT curated_manifest_apply")
        raise
    else:
        conn.execute("RELEASE SAVEPOINT curated_manifest_apply")
    conn.commit()
    return {
        "status": "ok",
        "schema_version": CURATED_MANIFEST_SCHEMA,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "sources_verified": len(sources),
        "facts_total": len(facts),
        "hosted_egress_acknowledged": bool(hosted_fact_ids and allow_hosted_egress),
        "applied": applied,
        "unchanged": unchanged,
    }


def _prepare_facts(
    facts: list[Any],
    *,
    project: str,
    sources: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """Validate and normalize every fact before the first ledger write."""
    result: list[dict[str, Any]] = []
    for raw in facts:
        fact_id = _required_text(raw, "id")
        body = _required_text(raw, "body")
        source_refs = raw.get("source_refs")
        if not isinstance(source_refs, list) or not source_refs:
            raise ValueError(f"fact {fact_id} must name at least one source_ref")
        referenced: list[dict[str, str]] = []
        for ref in source_refs:
            if str(ref) not in sources:
                raise ValueError(f"fact {fact_id} references unknown source {ref}")
            referenced.append(sources[str(ref)])
        scope = ScopeTag(
            "project",
            f"project:{project}",
            visibility=str(raw.get("visibility") or "internal"),
            egress_policy=str(raw.get("egress_policy") or "local_only"),
            provenance="explicit_curated_manifest",
        )
        belief_id = f"curated:{project}:{fact_id}"
        confidence = float(raw.get("confidence", 0.9))
        attributes = {
            "title": str(raw.get("title") or fact_id),
            "curated": True,
            "manifest_schema": CURATED_MANIFEST_SCHEMA,
            "source_attestations": [
                {"ref": source["ref"], "sha256": source["sha256"]}
                for source in referenced
            ],
            "source_quality": float(raw.get("source_quality", 0.95)),
            "lifecycle": str(raw.get("lifecycle") or "durable"),
            "content_sha256": sha256_text(body),
        }
        attributes["curation_sha256"] = sha256_text(
            canonical_json(
                {
                    "belief_id": belief_id,
                    "body": body,
                    "belief_type": "curated_fact",
                    "scope": scope.to_dict(),
                    "confidence": confidence,
                    "attributes": attributes,
                }
            )
        )
        result.append(
            {
                "fact_id": fact_id,
                "body": body,
                "referenced": referenced,
                "scope": scope,
                "belief_id": belief_id,
                "confidence": confidence,
                "attributes": attributes,
            }
        )
    return result


def _verify_sources(value: Any, *, base_dir: Path) -> dict[str, dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("manifest sources must be a non-empty list")
    result: dict[str, dict[str, str]] = {}
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("each curated source must be an object")
        ref = _required_text(raw, "ref")
        path = Path(_required_text(raw, "path")).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve()
        expected = _required_text(raw, "sha256").lower()
        if ref in result:
            raise ValueError(f"duplicate curated source ref: {ref}")
        if not path.is_file():
            raise ValueError(f"curated source does not exist: {path}")
        actual = _file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"curated source hash mismatch for {ref}: expected {expected}, got {actual}"
            )
        result[ref] = {"ref": ref, "path": str(path), "sha256": actual}
    return result


def _required_text(value: dict[str, Any], key: str) -> str:
    result = str(value.get(key) or "").strip()
    if not result:
        raise ValueError(f"{key} is required")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["CURATED_MANIFEST_SCHEMA", "apply_curated_manifest"]
