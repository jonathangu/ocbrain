from __future__ import annotations

import json
import sqlite3
from typing import Any

from ocbrain.db import now_iso
from ocbrain.events import canonical_json
from ocbrain.ids import content_hash, stable_id
from ocbrain.scope import ScopeContext

CLOSEOUT_SCHEMA_VERSION = "ocbrain.closeout.v1"
ACTION_SCHEMA_VERSION = "ocbrain.action.v1"
OUTCOME_SCHEMA_VERSION = "ocbrain.outcome.v1"
CLOSEOUT_STATUSES = {"completed", "partial", "blocked", "failed", "cancelled"}
DECISION_IMPACTS = {"none", "informed", "changed", "prevented_error", "unknown"}
VERIFIER_STATUSES = {"passed", "failed", "unknown", "not_required"}


def record_closeout(
    conn: sqlite3.Connection,
    *,
    task_ref: str,
    status: str,
    summary: str,
    context: ScopeContext | None = None,
    retrieval_use_ids: list[str] | None = None,
    decision_impact: str = "unknown",
    decision_note: str | None = None,
    artifact_refs: list[dict[str, Any]] | None = None,
    verifier_refs: list[dict[str, Any]] | None = None,
    actions: list[dict[str, Any]] | None = None,
    outcomes: list[dict[str, Any]] | None = None,
    awaiting: str | None = None,
    actor: str = "agent",
) -> dict[str, Any]:
    """Append a generic execution outcome receipt without promoting knowledge."""
    task_ref = _required_text(task_ref, "task_ref")
    summary = _required_text(summary, "summary")
    actor = _required_text(actor, "actor")
    if status not in CLOSEOUT_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(sorted(CLOSEOUT_STATUSES))}")
    if decision_impact not in DECISION_IMPACTS:
        raise ValueError(
            f"decision_impact must be one of: {', '.join(sorted(DECISION_IMPACTS))}"
        )
    if status == "blocked" and not (awaiting and awaiting.strip()):
        raise ValueError("blocked closeouts require awaiting")
    retrieval_ids = _dedupe_text(retrieval_use_ids or [])
    _validate_retrieval_ids(conn, retrieval_ids)
    artifacts = [_normalize_artifact_ref(value) for value in artifact_refs or []]
    verifiers = [_normalize_verifier_ref(value) for value in verifier_refs or []]
    normalized_actions = [_normalize_action(value) for value in actions or []]
    normalized_outcomes = [_normalize_outcome(value) for value in outcomes or []]
    verification_status = _verification_status(verifiers)
    resolved = context or ScopeContext()
    closed_at = now_iso()
    provenance = {
        "source": "agent_reported",
        "actor": actor,
        "runtime": resolved.runtime or "mcp",
        "session_id": resolved.session,
        "reported_at": closed_at,
    }
    base_receipt: dict[str, Any] = {
        "schema_version": CLOSEOUT_SCHEMA_VERSION,
        "closed_at": closed_at,
        "task_ref": task_ref,
        "status": status,
        "summary": summary,
        "decision": {
            "impact": decision_impact,
            "note": decision_note.strip() if decision_note and decision_note.strip() else None,
        },
        "retrieval_use_ids": retrieval_ids,
        "artifact_refs": artifacts,
        "verifier_refs": verifiers,
        "actions": normalized_actions,
        "outcomes": normalized_outcomes,
        "verification_status": verification_status,
        "awaiting": awaiting.strip() if awaiting and awaiting.strip() else None,
        "context": resolved.to_dict(),
        "provenance": provenance,
    }
    digest = content_hash(canonical_json(base_receipt))
    closeout_id = stable_id("close", task_ref, closed_at, digest)
    receipt = {"id": closeout_id, "content_hash": digest, **base_receipt}
    conn.execute(
        """
        INSERT INTO task_closeouts (
          id, schema_version, closed_at, task_ref, status, summary,
          decision_impact, decision_note, awaiting, runtime, session_id,
          context_json, artifact_refs_json, verifier_refs_json, provenance_json,
          receipt_json, content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            closeout_id,
            CLOSEOUT_SCHEMA_VERSION,
            closed_at,
            task_ref,
            status,
            summary,
            decision_impact,
            base_receipt["decision"]["note"],
            base_receipt["awaiting"],
            provenance["runtime"],
            provenance["session_id"],
            canonical_json(base_receipt["context"]),
            canonical_json(artifacts),
            canonical_json(verifiers),
            canonical_json(provenance),
            canonical_json(receipt),
            digest,
        ),
    )
    for retrieval_use_id in retrieval_ids:
        conn.execute(
            "INSERT INTO task_closeout_retrievals (closeout_id, retrieval_use_id) "
            "VALUES (?, ?)",
            (closeout_id, retrieval_use_id),
        )
        conn.execute(
            "UPDATE retrieval_uses SET affected_decision = ? WHERE id = ?",
            (_affected_decision(decision_impact), retrieval_use_id),
        )
    return receipt


def get_closeout(conn: sqlite3.Connection, closeout_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT receipt_json FROM task_closeouts WHERE id = ?",
        (closeout_id,),
    ).fetchone()
    return json.loads(row["receipt_json"]) if row is not None else None


def _validate_retrieval_ids(conn: sqlite3.Connection, retrieval_ids: list[str]) -> None:
    if not retrieval_ids:
        return
    placeholders = ",".join("?" for _ in retrieval_ids)
    found = {
        str(row["id"])
        for row in conn.execute(
            f"SELECT id FROM retrieval_uses WHERE id IN ({placeholders})",  # noqa: S608
            retrieval_ids,
        )
    }
    missing = [value for value in retrieval_ids if value not in found]
    if missing:
        raise ValueError(f"retrieval use not found: {', '.join(missing)}")


def _normalize_artifact_ref(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("artifact_refs entries must be objects")
    uri = _required_text(value.get("uri"), "artifact_refs[].uri")
    result: dict[str, Any] = {"uri": uri}
    for key in ("kind", "sha256", "label"):
        item = value.get(key)
        if item is not None:
            result[key] = _required_text(item, f"artifact_refs[].{key}")
    return result


def _normalize_verifier_ref(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("verifier_refs entries must be objects")
    uri = _required_text(value.get("uri"), "verifier_refs[].uri")
    status = str(value.get("status") or "unknown")
    if status not in VERIFIER_STATUSES:
        raise ValueError(f"verifier status must be one of: {', '.join(sorted(VERIFIER_STATUSES))}")
    result: dict[str, Any] = {"uri": uri, "status": status}
    for key in ("kind", "sha256", "detail"):
        item = value.get(key)
        if item is not None:
            result[key] = _required_text(item, f"verifier_refs[].{key}")
    return result


def _normalize_action(value: Any) -> dict[str, Any]:
    """Preserve a portable action envelope without pretending it is a reward."""
    if not isinstance(value, dict):
        raise ValueError("actions entries must be objects")
    target = _json_object(value.get("target"), "actions[].target", required=True)
    result: dict[str, Any] = {
        "schema_version": ACTION_SCHEMA_VERSION,
        "mechanism": _required_text(value.get("mechanism"), "actions[].mechanism"),
        "semantic_role": _required_text(
            value.get("semantic_role"), "actions[].semantic_role"
        ),
        "target": target,
    }
    for key in ("action_id", "occurred_at"):
        item = value.get(key)
        if item is not None:
            result[key] = _required_text(item, f"actions[].{key}")
    for key in ("context_before", "policy", "cost", "provenance", "features"):
        item = value.get(key)
        if item is not None:
            result[key] = _json_object(item, f"actions[].{key}", required=False)
    if "features" in result:
        result["feature_schema"] = _required_text(
            value.get("feature_schema"), "actions[].feature_schema"
        )
    elif value.get("feature_schema") is not None:
        raise ValueError("actions[].feature_schema requires actions[].features")
    return result


def _normalize_outcome(value: Any) -> dict[str, Any]:
    """Keep outcome components and local meaning instead of one scalar reward."""
    if not isinstance(value, dict):
        raise ValueError("outcomes entries must be objects")
    if "value" not in value:
        raise ValueError("outcomes[].value is required")
    result: dict[str, Any] = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "metric": _required_text(value.get("metric"), "outcomes[].metric"),
        "value": _json_value(value["value"], "outcomes[].value"),
        "role": _required_text(value.get("role") or "primary", "outcomes[].role"),
        "interpretation": _required_text(
            value.get("interpretation"), "outcomes[].interpretation"
        ),
    }
    for key in ("unit", "observed_at"):
        item = value.get(key)
        if item is not None:
            result[key] = _required_text(item, f"outcomes[].{key}")
    for key in (
        "observation_window",
        "baseline",
        "counterfactual",
        "attribution",
        "uncertainty",
        "features",
    ):
        item = value.get(key)
        if item is not None:
            result[key] = _json_value(item, f"outcomes[].{key}")
    if "features" in result:
        if not isinstance(result["features"], dict):
            raise ValueError("outcomes[].features must be an object")
        result["feature_schema"] = _required_text(
            value.get("feature_schema"), "outcomes[].feature_schema"
        )
    elif value.get("feature_schema") is not None:
        raise ValueError("outcomes[].feature_schema requires outcomes[].features")
    return result


def _json_object(value: Any, name: str, *, required: bool) -> dict[str, Any]:
    if not isinstance(value, dict) or (required and not value):
        suffix = "a non-empty object" if required else "an object"
        raise ValueError(f"{name} must be {suffix}")
    return _json_value(value, name)


def _json_value(value: Any, name: str) -> Any:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite JSON") from exc
    return json.loads(encoded)


def _verification_status(verifiers: list[dict[str, Any]]) -> str:
    if any(value["status"] == "failed" for value in verifiers):
        return "failed"
    if verifiers and all(value["status"] == "passed" for value in verifiers):
        return "verified"
    return "agent_reported"


def _affected_decision(decision_impact: str) -> int | None:
    if decision_impact in {"informed", "changed", "prevented_error"}:
        return 1
    if decision_impact == "none":
        return 0
    return None


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _dedupe_text(values: list[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = _required_text(value, "retrieval_use_ids[]")
        if text not in result:
            result.append(text)
    return result
