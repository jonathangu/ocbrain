from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocbrain.ids import content_hash, stable_id
from ocbrain.text import claim_key, compact_whitespace

SCHEMA_VERSION = "ocbrain.loop_result.v1"

DECISIONS = {"kept", "reverted", "failed", "needs_review", "skipped"}
DIRECTIONS = {"higher_is_better", "lower_is_better", "target", "boolean"}
TARGETS = {"memory", "wiki", "skill", "policy", "ignore"}
SCOPES = {"private", "workspace", "project", "public"}


@dataclass(frozen=True)
class LoopIngestOptions:
    loop_id: str
    run_id: str
    artifacts_root: Path
    ledger: Path | None = None
    backlog: Path | None = None
    dry_run: bool = True


def dry_run_loop_ingest(options: LoopIngestOptions) -> dict[str, Any]:
    if not options.dry_run:
        raise ValueError("loop ingest writes are not implemented yet; use --dry-run")

    result_paths = sorted(options.artifacts_root.rglob("result.json"))
    envelopes = [read_result_envelope(path, options) for path in result_paths]
    errors = [error for envelope in envelopes for error in envelope["errors"]]
    valid_envelopes = [envelope for envelope in envelopes if not envelope["errors"]]
    tripwires = [
        tripwire
        for envelope in valid_envelopes
        for tripwire in artifact_tripwires(envelope["data"], envelope["path"])
    ]
    families = experiment_family_summaries(valid_envelopes)
    primary = primary_metric_summary(valid_envelopes)
    candidates = candidate_summaries(valid_envelopes, families, primary)
    status = derive_run_status(valid_envelopes, errors, tripwires)

    return {
        "schema_version": "ocbrain.loop_ingest.dry_run.v1",
        "dry_run": True,
        "loop_id": options.loop_id,
        "run_id": options.run_id,
        "run_status": status,
        "inputs": {
            "artifacts": str(options.artifacts_root),
            "ledger": str(options.ledger) if options.ledger else None,
            "backlog": str(options.backlog) if options.backlog else None,
        },
        "summary": summary_counts(valid_envelopes),
        "metrics": {"primary": primary},
        "tripwires": tripwires,
        "experiment_families": families,
        "candidates": candidates,
        "envelopes": {
            "seen": len(result_paths),
            "valid": len(valid_envelopes),
            "invalid": len(envelopes) - len(valid_envelopes),
            "error_count": len(errors),
            "errors": errors,
        },
    }


def read_result_envelope(path: Path, options: LoopIngestOptions) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": path, "data": {}, "errors": [error(path, "invalid_json", str(exc))]}

    for field in ("schema_version", "loop_id", "run_id", "item_id", "eval", "verifier"):
        if field not in data:
            errors.append(error(path, "missing_field", field))
    if errors:
        return {"path": path, "data": data, "errors": errors}

    if data["schema_version"] != SCHEMA_VERSION:
        errors.append(error(path, "schema_version", data["schema_version"]))
    if data["loop_id"] != options.loop_id:
        errors.append(error(path, "loop_id_mismatch", data["loop_id"]))
    if data["run_id"] != options.run_id:
        errors.append(error(path, "run_id_mismatch", data["run_id"]))

    decision = data.get("decision")
    if decision is not None and decision not in DECISIONS:
        errors.append(error(path, "invalid_decision", str(decision)))

    eval_payload = data.get("eval") or {}
    for field in (
        "command",
        "metric_name",
        "direction",
        "baseline_value",
        "result_value",
        "delta_value",
        "passed",
    ):
        if field not in eval_payload:
            errors.append(error(path, "missing_eval_field", field))
    if eval_payload.get("direction") not in DIRECTIONS:
        errors.append(error(path, "invalid_metric_direction", str(eval_payload.get("direction"))))

    verifier = data.get("verifier") or {}
    for field in ("command", "passed", "evidence_uri"):
        if field not in verifier:
            errors.append(error(path, "missing_verifier_field", field))

    for index, candidate in enumerate(data.get("lesson_candidates") or []):
        target = candidate.get("target")
        if target not in TARGETS:
            errors.append(error(path, "invalid_lesson_target", f"{index}:{target}"))

    return {"path": path, "data": data, "errors": errors}


def error(path: Path, kind: str, message: str) -> dict[str, str]:
    return {"path": str(path), "kind": kind, "message": message}


def artifact_tripwires(data: dict[str, Any], result_path: Path) -> list[dict[str, Any]]:
    tripwires = []
    for uri in data.get("artifact_uris") or []:
        artifact_path = resolve_artifact_uri(uri, result_path)
        if not artifact_path.exists():
            tripwires.append(
                {
                    "id": stable_id("trip", data["loop_id"], data["run_id"], data["item_id"], uri),
                    "kind": "artifact_missing",
                    "severity": "warning",
                    "status": "open",
                    "message": f"Loop item {data['item_id']} references missing artifact {uri}",
                    "evidence_uri": str(result_path),
                    "loop_id": data["loop_id"],
                    "loop_run_id": data["run_id"],
                    "loop_item_id": data["item_id"],
                }
            )
    return tripwires


def resolve_artifact_uri(uri: str, result_path: Path) -> Path:
    path = Path(uri).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return result_path.parent / path


def summary_counts(envelopes: list[dict[str, Any]]) -> dict[str, int]:
    decisions = Counter(envelope["data"].get("decision", "unknown") for envelope in envelopes)
    return {
        "items": len(envelopes),
        "pending": 0,
        "running": 0,
        "done": sum(decisions[decision] for decision in ("kept", "reverted", "skipped")),
        "failed": decisions["failed"],
        "kept": decisions["kept"],
        "reverted": decisions["reverted"],
        "skipped": decisions["skipped"],
        "needs_review": decisions["needs_review"],
    }


def primary_metric_summary(envelopes: list[dict[str, Any]]) -> dict[str, Any] | None:
    metrics = [envelope["data"]["eval"] for envelope in envelopes]
    if not metrics:
        return None
    metric_name = metrics[0]["metric_name"]
    direction = metrics[0]["direction"]
    same_metric = [metric for metric in metrics if metric["metric_name"] == metric_name]
    best = best_metric_value(same_metric, direction)
    first = same_metric[0]
    return {
        "name": metric_name,
        "baseline": first["baseline_value"],
        "best": best["result_value"],
        "delta": best["delta_value"],
        "direction": direction,
        "passed": bool(best.get("passed")),
        "evidence_uri": best.get("evidence_uri"),
    }


def best_metric_value(metrics: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    if direction == "lower_is_better":
        return min(metrics, key=lambda item: numeric_or_inf(item.get("result_value")))
    if direction == "higher_is_better":
        return max(metrics, key=lambda item: numeric_or_inf(item.get("result_value")))
    return metrics[-1]


def numeric_or_inf(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def experiment_family_summaries(envelopes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for envelope in envelopes:
        data = envelope["data"]
        groups[data.get("experiment_family") or "unknown"].append(data)

    summaries = []
    for name, rows in sorted(groups.items()):
        attempts = len(rows)
        kept = sum(1 for row in rows if row.get("decision") == "kept")
        failed = sum(1 for row in rows if row.get("decision") == "failed")
        reverted = sum(1 for row in rows if row.get("decision") == "reverted")
        guardrail_failures = sum(
            1
            for row in rows
            for guardrail in row.get("guardrails") or []
            if not guardrail.get("passed")
        )
        deltas = [numeric_delta(row.get("eval", {}).get("delta_value")) for row in rows]
        mean_delta = round(sum(deltas) / len(deltas), 3) if deltas else None
        summaries.append(
            {
                "name": name,
                "attempts": attempts,
                "kept": kept,
                "failed": failed,
                "reverted": reverted,
                "guardrail_failures": guardrail_failures,
                "mean_delta": mean_delta,
                "status": classify_family(attempts, kept, failed, guardrail_failures),
            }
        )
    return summaries


def numeric_delta(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def classify_family(attempts: int, kept: int, failed: int, guardrail_failures: int) -> str:
    if guardrail_failures:
        return "risky"
    if attempts >= 3 and kept == 0 and failed >= 2:
        return "exhausted"
    if kept >= 2:
        return "promising"
    if kept == 1:
        return "needs_more_evidence"
    return "unclear"


def candidate_summaries(
    envelopes: list[dict[str, Any]],
    families: list[dict[str, Any]],
    primary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    if primary and primary.get("delta") not in (None, 0, "0"):
        first_data = envelopes[0]["data"]
        body = (
            f"{primary['name']} changed from {primary['baseline']} to {primary['best']} "
            f"({primary['delta']}, {primary['direction']}) in loop {first_data['loop_id']} "
            f"run {first_data['run_id']}."
        )
        candidates.append(candidate("memory", "Loop metric baseline changed", body, 0.82))
        seen.add(("memory", claim_key(body)))

    for family in families:
        if family["status"] in {"promising", "exhausted", "risky"}:
            target = "wiki" if family["status"] == "promising" else "memory"
            body = (
                f"Experiment family {family['name']} is {family['status']} after "
                f"{family['attempts']} attempts, {family['kept']} kept, "
                f"{family['failed']} failed, and {family['reverted']} reverted."
            )
            key = (target, claim_key(body))
            if key not in seen:
                candidates.append(
                    candidate(target, f"Loop family: {family['name']}", body, 0.8)
                )
                seen.add(key)
        if family["kept"] >= 3 and family["guardrail_failures"] == 0:
            body = (
                f"Repeatable loop procedure candidate: experiment family {family['name']} "
                f"kept {family['kept']} verified changes across {family['attempts']} attempts "
                "without guardrail failures."
            )
            key = ("skill", claim_key(body))
            if key not in seen:
                candidates.append(
                    candidate(
                        "skill",
                        f"Procedure: {family['name']}",
                        body,
                        0.81,
                    )
                )
                seen.add(key)

    for envelope in envelopes:
        data = envelope["data"]
        for lesson in data.get("lesson_candidates") or []:
            target = lesson["target"]
            body = compact_whitespace(str(lesson.get("body", "")))
            if not body:
                continue
            key = (target, claim_key(body))
            if key in seen:
                continue
            candidates.append(
                candidate(
                    target,
                    title_from_body(body),
                    body,
                    lesson_confidence(target, data.get("decision")),
                    scope=data.get("privacy_scope", "workspace"),
                    evidence_uri=str(envelope["path"]),
                )
            )
            seen.add(key)
    return candidates


def candidate(
    target: str,
    title: str,
    body: str,
    confidence: float,
    *,
    scope: str = "workspace",
    evidence_uri: str | None = None,
) -> dict[str, Any]:
    if scope not in SCOPES:
        scope = "workspace"
    payload = {
        "target": target,
        "title": title[:120],
        "body": body,
        "confidence": confidence,
        "risk": "medium" if target in {"skill", "policy"} else "low",
        "privacy_scope": scope,
        "status": "proposal_only" if target in {"skill", "policy"} else "staged",
    }
    if evidence_uri:
        payload["evidence_uri"] = evidence_uri
    return payload


def title_from_body(body: str) -> str:
    return body.split(".")[0][:120] or "Loop lesson"


def lesson_confidence(target: str, decision: str | None) -> float:
    base = 0.74
    if decision == "kept":
        base += 0.06
    if target in {"skill", "policy"}:
        base -= 0.02
    return round(base, 2)


def derive_run_status(
    envelopes: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    tripwires: list[dict[str, Any]],
) -> str:
    if errors:
        return "needs_review"
    if any(tripwire["severity"] in {"high", "critical"} for tripwire in tripwires):
        return "needs_review"
    decisions = {envelope["data"].get("decision") for envelope in envelopes}
    if "needs_review" in decisions:
        return "needs_review"
    if "failed" in decisions and decisions <= {"failed"}:
        return "failed"
    if envelopes:
        return "completed"
    return "planned"


def result_hash(path: Path) -> str:
    return content_hash(path.read_text(encoding="utf-8", errors="replace"))
