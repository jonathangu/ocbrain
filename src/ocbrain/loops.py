from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocbrain.db import (
    link_knowledge_evidence,
    now_iso,
    upsert_evidence,
    upsert_knowledge,
)
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


def loop_run_row_id(loop_id: str, run_id: str) -> str:
    return stable_id("lrun", loop_id, run_id)


def dry_run_loop_ingest(options: LoopIngestOptions) -> dict[str, Any]:
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
        "dry_run": options.dry_run,
        "loop_id": options.loop_id,
        "run_id": options.run_id,
        "loop_run_row_id": loop_run_row_id(options.loop_id, options.run_id),
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


def write_loop_ingest(conn: sqlite3.Connection, options: LoopIngestOptions) -> dict[str, Any]:
    result = dry_run_loop_ingest(options)
    if result["envelopes"]["invalid"]:
        raise ValueError("cannot apply invalid loop envelopes")

    timestamp = now_iso()
    envelopes = [
        read_result_envelope(path, options)
        for path in sorted(options.artifacts_root.rglob("result.json"))
    ]
    valid_envelopes = [envelope for envelope in envelopes if not envelope["errors"]]
    loop_id = options.loop_id
    run_row_id = result["loop_run_row_id"]
    evidence_ids = []
    for envelope in valid_envelopes:
        evidence_ids.extend(write_envelope_rows(conn, envelope, loop_id, run_row_id, timestamp))
    for tripwire in result["tripwires"]:
        evidence_ids.append(
            upsert_evidence(
                conn,
                source_type="loop_tripwire",
                source_uri=tripwire["evidence_uri"],
                content_hash=content_hash(json.dumps(tripwire, sort_keys=True)),
                claim=tripwire["message"],
                verifier_status="not_required",
                loop_tags={
                    "loop_id": loop_id,
                    "run_id": options.run_id,
                    "item_id": tripwire["loop_item_id"],
                    "tripwire": tripwire["kind"],
                },
                privacy_scope="workspace",
                occurred_at=timestamp,
            ),
        )
    for candidate_summary in result["candidates"]:
        knowledge_from_candidate_summary(
            conn,
            candidate_summary,
            loop_id=loop_id,
            run_id=options.run_id,
            content_hash_value=content_hash(json.dumps(candidate_summary, sort_keys=True)),
        )
    refresh_family_scores(conn, loop_id, result["experiment_families"], timestamp)
    conn.commit()
    result["dry_run"] = False
    knowledge_candidate_count = len(result["candidates"]) + (
        1 if result["metrics"]["primary"] else 0
    )
    result["applied"] = {
        "loop_id": loop_id,
        "loop_run_id": run_row_id,
        "evidence": len(set(evidence_ids)),
        "knowledge_candidates": knowledge_candidate_count,
        "tripwires": len(result["tripwires"]),
    }
    return result


def write_envelope_rows(
    conn: sqlite3.Connection,
    envelope: dict[str, Any],
    loop_id: str,
    run_row_id: str,
    timestamp: str,
) -> list[str]:
    data = envelope["data"]
    external_item_id = data["item_id"]
    eval_payload = data["eval"]
    verifier = data["verifier"]
    loop_tags = {
        "loop_id": loop_id,
        "run_id": data["run_id"],
        "item_id": external_item_id,
        "family": data.get("experiment_family"),
    }
    evidence_id = upsert_evidence(
        conn,
        source_type="loop_iteration",
        source_runtime="openclaw",
        source_uri=str(envelope["path"]),
        content_hash=result_hash(envelope["path"]),
        claim=loop_iteration_claim(data),
        artifact_uri=str(envelope["path"]),
        artifact_hash=result_hash(envelope["path"]),
        verifier_status="passed" if verifier.get("passed") else "failed",
        loop_tags=loop_tags,
        project=data.get("project"),
        privacy_scope=data.get("privacy_scope", "workspace"),
        occurred_at=data.get("created_at") or timestamp,
    )
    metric_knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject=f"loop:{loop_id}:family:{data.get('experiment_family') or 'unknown'}",
        predicate=eval_payload["metric_name"],
        value_numeric=numeric_delta(eval_payload.get("result_value")),
        unit=eval_payload.get("unit"),
        target_value=numeric_delta(eval_payload.get("baseline_value")),
        status="current" if eval_payload.get("passed") else "candidate",
        confidence=0.82 if eval_payload.get("passed") else 0.62,
        content_hash=result_hash(envelope["path"]),
        loop_tags=loop_tags,
        project=data.get("project"),
        privacy_scope=data.get("privacy_scope", "workspace"),
    )
    link_knowledge_evidence(conn, metric_knowledge_id, evidence_id, relation="derived_from")
    evidence_ids = [evidence_id]
    for uri in data.get("artifact_uris") or []:
        artifact_path = resolve_artifact_uri(uri, envelope["path"])
        if artifact_path.exists():
            evidence_ids.append(
                upsert_evidence(
                    conn,
                    source_type="loop_artifact",
                    source_runtime="openclaw",
                    source_uri=str(artifact_path),
                    content_hash=file_hash(artifact_path),
                    claim=f"Loop artifact for {loop_id}/{data['run_id']}/{external_item_id}: {uri}",
                    artifact_uri=str(artifact_path),
                    artifact_hash=file_hash(artifact_path),
                    verifier_status="passed" if verifier.get("passed") else "failed",
                    loop_tags={**loop_tags, "artifact_kind": infer_artifact_kind(uri)},
                    project=data.get("project"),
                    privacy_scope=data.get("privacy_scope", "workspace"),
                    occurred_at=data.get("created_at") or timestamp,
                )
            )
    for lesson in data.get("lesson_candidates") or []:
        knowledge_id = knowledge_from_lesson(conn, lesson, data, loop_tags, envelope["path"])
        if knowledge_id:
            link_knowledge_evidence(conn, knowledge_id, evidence_id, relation="derived_from")
    return evidence_ids


def loop_iteration_claim(data: dict[str, Any]) -> str:
    eval_payload = data["eval"]
    return (
        f"Loop {data['loop_id']} run {data['run_id']} item {data['item_id']} "
        f"{data.get('decision', 'unknown')}; {eval_payload['metric_name']} "
        f"{eval_payload.get('baseline_value')} -> {eval_payload.get('result_value')} "
        f"({eval_payload.get('delta_value')})."
    )


def knowledge_from_lesson(
    conn: sqlite3.Connection,
    lesson: dict[str, Any],
    data: dict[str, Any],
    loop_tags: dict[str, Any],
    result_path: Path,
) -> str | None:
    body = compact_whitespace(str(lesson.get("body", "")))
    if not body:
        return None
    target = lesson.get("target")
    if target == "ignore":
        return None
    if target == "memory":
        return upsert_knowledge(
            conn,
            knowledge_type="value",
            gate="auto",
            subject=f"loop:{data['loop_id']}",
            predicate=claim_key(body, limit=80),
            value_text=body,
            status="current" if data.get("decision") == "kept" else "candidate",
            inject=True,
            confidence=0.78,
            content_hash=result_hash(result_path),
            loop_tags=loop_tags,
            project=data.get("project"),
            privacy_scope=data.get("privacy_scope", "workspace"),
        )
    if target in {"wiki", "policy"}:
        return upsert_knowledge(
            conn,
            knowledge_type="doc",
            gate="human" if target == "policy" else "auto",
            slug=stable_id("doc", data["loop_id"], body),
            title=title_from_body(body),
            body_uri=str(result_path),
            doc_kind="procedure" if target == "policy" else "wiki",
            status="candidate",
            prescriptive=target == "policy",
            risk="medium" if target == "policy" else "low",
            confidence=0.76,
            content_hash=result_hash(result_path),
            loop_tags=loop_tags,
            project=data.get("project"),
            privacy_scope=data.get("privacy_scope", "workspace"),
        )
    if target == "skill":
        return upsert_knowledge(
            conn,
            knowledge_type="capability",
            gate="human",
            slug=stable_id("cap", data["loop_id"], body),
            title=title_from_body(body),
            body_uri=str(result_path),
            status="candidate",
            risk="medium",
            confidence=0.74,
            content_hash=result_hash(result_path),
            loop_tags=loop_tags,
            project=data.get("project"),
            privacy_scope=data.get("privacy_scope", "workspace"),
        )
    return None


def knowledge_from_candidate_summary(
    conn: sqlite3.Connection,
    candidate_summary: dict[str, Any],
    *,
    loop_id: str,
    run_id: str,
    content_hash_value: str,
) -> str | None:
    target = candidate_summary["target"]
    body = candidate_summary["body"]
    loop_tags = {"loop_id": loop_id, "run_id": run_id}
    if target == "memory":
        return upsert_knowledge(
            conn,
            knowledge_type="value",
            gate="auto",
            subject=f"loop:{loop_id}",
            predicate=claim_key(body, limit=80),
            value_text=body,
            status="candidate",
            inject=True,
            confidence=candidate_summary.get("confidence"),
            content_hash=content_hash_value,
            loop_tags=loop_tags,
            privacy_scope=candidate_summary.get("privacy_scope", "workspace"),
        )
    if target in {"wiki", "policy"}:
        return upsert_knowledge(
            conn,
            knowledge_type="doc",
            gate="human" if target == "policy" else "auto",
            slug=stable_id("doc", loop_id, body),
            title=candidate_summary.get("title") or title_from_body(body),
            body_uri=candidate_summary.get("evidence_uri"),
            doc_kind="procedure" if target == "policy" else "wiki",
            status="candidate",
            prescriptive=target == "policy",
            risk=candidate_summary.get("risk", "low"),
            confidence=candidate_summary.get("confidence"),
            content_hash=content_hash_value,
            loop_tags=loop_tags,
            privacy_scope=candidate_summary.get("privacy_scope", "workspace"),
        )
    if target == "skill":
        return upsert_knowledge(
            conn,
            knowledge_type="capability",
            gate="human",
            slug=stable_id("cap", loop_id, body),
            title=candidate_summary.get("title") or title_from_body(body),
            body_uri=candidate_summary.get("evidence_uri"),
            status="candidate",
            risk=candidate_summary.get("risk", "medium"),
            confidence=candidate_summary.get("confidence"),
            content_hash=content_hash_value,
            loop_tags=loop_tags,
            privacy_scope=candidate_summary.get("privacy_scope", "workspace"),
        )
    return None


def refresh_family_scores(
    conn: sqlite3.Connection,
    loop_id: str,
    families: list[dict[str, Any]],
    timestamp: str,
) -> None:
    conn.execute("DELETE FROM family_scores WHERE loop_id = ?", (loop_id,))
    for family in families:
        conn.execute(
            """
            INSERT INTO family_scores (
              loop_id, family, attempts, kept, reverted, approach_failures,
              verifier_pass_rate, mean_primary_delta, recency, state, refreshed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                loop_id,
                family["name"],
                family["attempts"],
                family["kept"],
                family["reverted"],
                family["failed"],
                None,
                family["mean_delta"],
                timestamp,
                family_state(family["status"]),
                timestamp,
            ),
        )


def family_state(status: str) -> str:
    if status in {"promising", "risky", "stale"}:
        return status
    if status == "exhausted":
        return "exhausted"
    return "untried"


def infer_artifact_kind(uri: str) -> str:
    suffix = Path(uri).suffix.lower()
    if suffix == ".patch":
        return "patch"
    if suffix in {".diff"}:
        return "diff"
    if suffix in {".json", ".jsonl"}:
        return "eval"
    if suffix in {".log", ".txt"}:
        return "log"
    if suffix in {".md"}:
        return "report"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "screenshot"
    return "other"


def file_hash(path: Path) -> str:
    return content_hash(path.read_text(encoding="utf-8", errors="replace"))


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
