from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocbrain.db import now_iso
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
    run_summary = json.dumps(result["summary"], sort_keys=True)
    first = valid_envelopes[0]["data"] if valid_envelopes else {}

    conn.execute(
        """
        INSERT INTO loop_programs (
          id, name, project, owner, objective, primary_metric_name,
          primary_metric_direction, baseline_value, verifier_ref, status,
          risk, privacy_scope, definition_uri, content_hash, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          name = excluded.name,
          project = excluded.project,
          objective = excluded.objective,
          primary_metric_name = excluded.primary_metric_name,
          primary_metric_direction = excluded.primary_metric_direction,
          baseline_value = excluded.baseline_value,
          verifier_ref = excluded.verifier_ref,
          privacy_scope = excluded.privacy_scope,
          definition_uri = excluded.definition_uri,
          content_hash = excluded.content_hash,
          updated_at = excluded.updated_at
        """,
        (
            loop_id,
            loop_id,
            first.get("project"),
            None,
            first.get("objective") or f"Loop program {loop_id}",
            first.get("eval", {}).get("metric_name"),
            first.get("eval", {}).get("direction"),
            str(first.get("eval", {}).get("baseline_value")),
            first.get("verifier", {}).get("command"),
            "draft",
            "medium",
            first.get("privacy_scope", "workspace"),
            str(options.artifacts_root),
            content_hash(json.dumps(result, sort_keys=True)),
            timestamp,
            timestamp,
        ),
    )
    conn.execute(
        """
        INSERT INTO loop_runs (
          id, loop_id, trigger_type, trigger_ref, backlog_snapshot_uri,
          started_at, ended_at, status, budget_used_json, summary, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          trigger_ref = excluded.trigger_ref,
          backlog_snapshot_uri = excluded.backlog_snapshot_uri,
          status = excluded.status,
          budget_used_json = excluded.budget_used_json,
          summary = excluded.summary,
          updated_at = excluded.updated_at
        """,
        (
            run_row_id,
            loop_id,
            "unknown",
            options.run_id,
            str(options.backlog) if options.backlog else None,
            first.get("started_at"),
            first.get("ended_at"),
            result["run_status"],
            None,
            run_summary,
            timestamp,
            timestamp,
        ),
    )
    delete_run_children(conn, run_row_id)
    for envelope in valid_envelopes:
        write_envelope_rows(conn, envelope, loop_id, run_row_id, timestamp)
    for tripwire in result["tripwires"]:
        conn.execute(
            """
            INSERT OR REPLACE INTO loop_tripwires (
              id, loop_id, loop_run_id, loop_item_id, kind, severity, status,
              message, evidence_uri, opened_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tripwire["id"],
                loop_id,
                run_row_id,
                stable_id("litem", loop_id, options.run_id, tripwire["loop_item_id"]),
                tripwire["kind"],
                tripwire["severity"],
                tripwire["status"],
                tripwire["message"],
                tripwire["evidence_uri"],
                timestamp,
            ),
        )
    conn.commit()
    result["dry_run"] = False
    result["applied"] = {
        "loop_program_id": loop_id,
        "loop_run_id": run_row_id,
        "items": len(valid_envelopes),
        "tripwires": len(result["tripwires"]),
    }
    return result


def delete_run_children(conn: sqlite3.Connection, run_row_id: str) -> None:
    for table in (
        "loop_candidate_links",
        "loop_tripwires",
        "loop_artifacts",
        "loop_metrics",
        "loop_iterations",
        "loop_items",
    ):
        conn.execute(f"DELETE FROM {table} WHERE loop_run_id = ?", (run_row_id,))


def write_envelope_rows(
    conn: sqlite3.Connection,
    envelope: dict[str, Any],
    loop_id: str,
    run_row_id: str,
    timestamp: str,
) -> None:
    data = envelope["data"]
    external_item_id = data["item_id"]
    item_id = stable_id("litem", loop_id, data["run_id"], external_item_id)
    iteration_id = stable_id("liter", loop_id, data["run_id"], external_item_id)
    decision = data.get("decision") or "unknown"
    status = {
        "kept": "done",
        "reverted": "reverted",
        "failed": "failed",
        "skipped": "skipped",
        "needs_review": "needs_review",
    }.get(decision, "needs_review")
    eval_payload = data["eval"]
    verifier = data["verifier"]

    conn.execute(
        """
        INSERT INTO loop_items (
          id, loop_run_id, external_backlog_id, spec_uri, spec_hash, experiment_family,
          status, claimed_by, worker_session_uri, timeout_seconds, claimed_at, started_at,
          ended_at, final_decision, failure_reason, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item_id,
            run_row_id,
            external_item_id,
            str(envelope["path"]),
            result_hash(envelope["path"]),
            data.get("experiment_family"),
            status,
            None,
            data.get("worker_session_uri"),
            None,
            None,
            data.get("started_at"),
            data.get("ended_at"),
            decision,
            data.get("failure_reason"),
            timestamp,
            timestamp,
        ),
    )
    conn.execute(
        """
        INSERT INTO loop_iterations (
          id, loop_item_id, loop_run_id, loop_id, hypothesis, mechanism,
          experiment_family, change_summary, changed_files_json, eval_command,
          guardrail_commands_json, verifier_command, verifier_passed, baseline_value,
          result_value, delta_value, decision, lesson_summary, next_candidate,
          started_at, ended_at, duration_seconds, tokens_used, cost_estimate,
          tool_profile, privacy_scope, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            iteration_id,
            item_id,
            run_row_id,
            loop_id,
            data.get("hypothesis"),
            data.get("mechanism"),
            data.get("experiment_family"),
            data.get("change_summary"),
            json.dumps(data.get("changed_files") or [], sort_keys=True),
            eval_payload.get("command"),
            json.dumps(data.get("guardrails") or [], sort_keys=True),
            verifier.get("command"),
            1 if verifier.get("passed") else 0,
            str(eval_payload.get("baseline_value")),
            str(eval_payload.get("result_value")),
            str(eval_payload.get("delta_value")),
            decision,
            "; ".join(
                compact_whitespace(str(item.get("body", "")))
                for item in data.get("lesson_candidates") or []
                if item.get("body")
            ),
            "; ".join(data.get("next_candidates") or []),
            data.get("started_at"),
            data.get("ended_at"),
            data.get("duration_seconds"),
            data.get("tokens_used"),
            data.get("cost_estimate"),
            (data.get("safety") or {}).get("tool_profile"),
            data.get("privacy_scope", "workspace"),
            timestamp,
            timestamp,
        ),
    )
    conn.execute(
        """
        INSERT INTO loop_metrics (
          id, loop_iteration_id, loop_run_id, loop_id, metric_name, metric_kind,
          direction, unit, baseline_value, result_value, delta_value, passed,
          measured_at, evidence_uri, evidence_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            stable_id(
                "lmet",
                loop_id,
                data["run_id"],
                external_item_id,
                eval_payload["metric_name"],
            ),
            iteration_id,
            run_row_id,
            loop_id,
            eval_payload["metric_name"],
            "primary",
            eval_payload["direction"],
            eval_payload.get("unit"),
            numeric_delta(eval_payload.get("baseline_value")),
            numeric_delta(eval_payload.get("result_value")),
            numeric_delta(eval_payload.get("delta_value")),
            1 if eval_payload.get("passed") else 0,
            data.get("ended_at") or data.get("created_at") or timestamp,
            eval_payload.get("evidence_uri"),
            None,
        ),
    )
    for uri in data.get("artifact_uris") or []:
        artifact_path = resolve_artifact_uri(uri, envelope["path"])
        conn.execute(
            """
            INSERT INTO loop_artifacts (
              id, loop_iteration_id, loop_run_id, loop_id, kind, uri, hash,
              size_bytes, verifier_status, privacy_scope, produced_at, ingested_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stable_id("lart", loop_id, data["run_id"], external_item_id, uri),
                iteration_id,
                run_row_id,
                loop_id,
                infer_artifact_kind(uri),
                uri,
                file_hash(artifact_path) if artifact_path.exists() else None,
                artifact_path.stat().st_size if artifact_path.exists() else None,
                "passed" if verifier.get("passed") else "failed",
                data.get("privacy_scope", "workspace"),
                data.get("ended_at"),
                timestamp,
            ),
        )


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
