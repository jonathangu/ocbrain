"""Local-only LLM grading for mined dataset examples.

The training corpus never leaves the machine. This module enforces that policy
at the transport boundary by refusing every non-loopback endpoint, then stores a
small normalized rubric result both in dedicated columns and in each example's
``metadata.llm_grade`` object. The example body/content hash is unchanged.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from ocbrain.config import DatasetGradingConfig, OcbrainConfig, load_config
from ocbrain.events import canonical_json, sha256_text
from ocbrain.fsutil import db_side_dir, file_lock
from ocbrain.ids import stable_id

DATASET_RUBRICS: dict[str, tuple[str, ...]] = {
    "sft": ("correctness", "usefulness", "instruction_following", "clarity"),
    "dpo": ("preference_validity", "chosen_quality", "rejected_defect", "contrast_strength"),
    "persona": ("voice_fidelity", "taste_alignment", "naturalness", "specificity"),
}

GradeTransport = Callable[[str, str, list[dict[str, str]], int], Any]


def require_loopback_endpoint(endpoint: str) -> str:
    """Return a normalized endpoint or raise before any corpus text is read."""
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("dataset grader endpoint must use http or https")
    if parsed.username or parsed.password:
        raise ValueError("dataset grader endpoint must not contain credentials")
    if (parsed.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("dataset grader endpoint must be loopback-only")
    return endpoint


def _body_without_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "metadata"}


def _messages(dataset: str, record: dict[str, Any]) -> list[dict[str, str]]:
    dimensions = DATASET_RUBRICS[dataset]
    rubric = ", ".join(dimensions)
    system = (
        "You grade one local fine-tuning example. Return JSON only. "
        "Score each requested dimension and overall_score from 0.0 to 1.0. "
        "Use verdict pass, review, or fail. flags is a short list of lowercase slugs; "
        "explanation is at most 240 characters. Do not reproduce the example. "
        "Judge only what the example supports; uncertainty lowers correctness."
    )
    user = canonical_json(
        {
            "dataset": dataset,
            "dimensions": dimensions,
            "required_schema": {
                "overall_score": "number 0..1",
                "dimensions": {name: "number 0..1" for name in dimensions},
                "verdict": "pass|review|fail",
                "flags": ["short_slug"],
                "explanation": "short string",
            },
            "example": _body_without_metadata(record),
            "rubric_summary": rubric,
        }
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _ollama_transport(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
) -> Any:
    payload = canonical_json(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        envelope = json.loads(response.read().decode("utf-8"))
    message = envelope.get("message") if isinstance(envelope, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("local grader returned no message content")
    return json.loads(content)


def _score(value: Any, name: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name} score") from exc
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} score outside 0..1")
    return round(score, 4)


def normalize_grade(dataset: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("local grader response is not an object")
    dimensions_raw = raw.get("dimensions")
    if not isinstance(dimensions_raw, dict):
        raise ValueError("local grader response has no dimensions object")
    dimensions = {
        name: _score(dimensions_raw.get(name), name) for name in DATASET_RUBRICS[dataset]
    }
    overall = _score(raw.get("overall_score"), "overall")
    verdict = str(raw.get("verdict") or "").lower()
    if verdict not in {"pass", "review", "fail"}:
        verdict = "pass" if overall >= 0.8 else "review" if overall >= 0.6 else "fail"
    flags_raw = raw.get("flags")
    flags = []
    if isinstance(flags_raw, list):
        flags = [str(flag)[:64] for flag in flags_raw[:8] if str(flag).strip()]
    explanation = str(raw.get("explanation") or "")[:240]
    return {
        "overall_score": overall,
        "dimensions": dimensions,
        "verdict": verdict,
        "flags": flags,
        "explanation": explanation,
    }


def _daily_items(conn: sqlite3.Connection, day: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(SUM(item_count), 0) AS n FROM dataset_grade_runs "
        "WHERE substr(ts, 1, 10) = ?",
        (day,),
    ).fetchone()
    return int(row["n"] if row is not None else 0)


def _repair_interrupted_runs(conn: sqlite3.Connection) -> int:
    """Close run rows left ``running`` by a killed/lock-blocked grader.

    The public entry point holds the single-grader file lock, so a running row
    seen here cannot belong to another healthy local grading process.
    """
    try:
        cursor = conn.execute(
            """
            UPDATE dataset_grade_runs
            SET status = 'interrupted',
                error = COALESCE(error, '{"Interrupted":1}')
            WHERE status = 'running'
            """
        )
        conn.commit()
    except sqlite3.OperationalError:
        conn.rollback()
        return -1
    return int(cursor.rowcount)


def _candidate_rows(
    conn: sqlite3.Connection,
    datasets: Sequence[str],
    *,
    model: str,
    prompt_version: str,
    force: bool,
    limit: int,
) -> list[sqlite3.Row]:
    placeholders = ",".join("?" for _ in datasets)
    clauses = [
        f"dataset IN ({placeholders})",  # noqa: S608 - placeholders only
        "quality_label IN ('good','neutral')",
    ]
    params: list[Any] = list(datasets)
    if not force:
        clauses.append(
            "(grade_model IS NULL OR grade_model != ? OR grade_prompt_version != ?)"
        )
        params.extend([model, prompt_version])
    params.append(limit)
    return list(
        conn.execute(
            f"SELECT id, dataset, example_json FROM dataset_examples "  # noqa: S608
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY dataset, COALESCE(occurred_at, ''), id LIMIT ?",
            params,
        )
    )


def _store_grade(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    grade: dict[str, Any],
    *,
    model: str,
    prompt_version: str,
    graded_at: str,
) -> None:
    record = json.loads(row["example_json"])
    metadata = record.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        record["metadata"] = metadata
    persisted = {
        **grade,
        "model": model,
        "prompt_version": prompt_version,
        "graded_at": graded_at,
        "local_only": True,
    }
    metadata["llm_grade"] = persisted
    example_json = canonical_json(record)
    conn.execute(
        """
        UPDATE dataset_examples
        SET grade_score = ?, grade_json = ?, grade_model = ?,
            grade_prompt_version = ?, graded_at = ?, example_json = ?,
            n_chars = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            grade["overall_score"],
            canonical_json(persisted),
            model,
            prompt_version,
            graded_at,
            example_json,
            len(example_json),
            graded_at,
            row["id"],
        ),
    )


def _store_grade_error(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    error_type: str,
    model: str,
    prompt_version: str,
    graded_at: str,
) -> None:
    persisted = {
        "status": "error",
        "error_type": error_type,
        "model": model,
        "prompt_version": prompt_version,
        "graded_at": graded_at,
        "local_only": True,
    }
    conn.execute(
        """
        UPDATE dataset_examples
        SET grade_score = NULL, grade_json = ?, grade_model = ?,
            grade_prompt_version = ?, graded_at = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            canonical_json(persisted),
            model,
            prompt_version,
            graded_at,
            graded_at,
            row["id"],
        ),
    )


def _grade_examples_unlocked(
    conn: sqlite3.Connection,
    *,
    cfg: OcbrainConfig | None = None,
    datasets: Sequence[str] | None = None,
    limit: int | None = None,
    endpoint: str | None = None,
    model: str | None = None,
    force: bool = False,
    now: datetime | None = None,
    transport: GradeTransport | None = None,
) -> dict[str, Any]:
    """Grade a bounded batch and persist normalized metadata.

    Endpoint validation happens before selecting or decoding any example, so a
    bad configuration cannot accidentally place corpus text on a remote request.
    """
    cfg = cfg or load_config()
    grade_cfg: DatasetGradingConfig = cfg.dataset_grading
    endpoint = require_loopback_endpoint(endpoint or grade_cfg.endpoint)
    model = model or grade_cfg.model
    if not model:
        raise ValueError("dataset grader model is required")
    wanted = tuple(datasets or DATASET_RUBRICS)
    unknown = sorted(set(wanted) - set(DATASET_RUBRICS))
    if unknown:
        raise ValueError(f"unknown datasets: {', '.join(unknown)}")

    repaired_runs = _repair_interrupted_runs(conn)
    if repaired_runs < 0:
        return {
            "action": "dataset-grade",
            "changed": 0,
            "graded": 0,
            "errors": 0,
            "status": "blocked",
            "skipped": "database_lock",
            "local_only": True,
            "ledger_pending": False,
        }

    instant = now or datetime.now(UTC)
    timestamp = instant.isoformat(timespec="microseconds")
    used_today = _daily_items(conn, timestamp[:10])
    daily_remaining = max(0, grade_cfg.daily_item_cap - used_today)
    requested = grade_cfg.per_run_item_cap if limit is None else max(0, limit)
    batch_limit = min(requested, grade_cfg.per_run_item_cap, daily_remaining)
    if batch_limit == 0:
        return {
            "action": "dataset-grade",
            "changed": 0,
            "graded": 0,
            "errors": 0,
            "skipped": "item_cap",
            "daily_items": used_today,
            "daily_item_cap": grade_cfg.daily_item_cap,
            "repaired_runs": repaired_runs,
        }

    rows = _candidate_rows(
        conn,
        wanted,
        model=model,
        prompt_version=grade_cfg.prompt_version,
        force=force,
        limit=batch_limit,
    )
    if not rows:
        return {
            "action": "dataset-grade",
            "changed": 0,
            "graded": 0,
            "errors": 0,
            "skipped": "no_candidates",
            "daily_items": used_today,
            "daily_item_cap": grade_cfg.daily_item_cap,
            "repaired_runs": repaired_runs,
        }

    request_hash = sha256_text(
        canonical_json(
            {
                "ids": [row["id"] for row in rows],
                "model": model,
                "prompt_version": grade_cfg.prompt_version,
            }
        )
    )
    run_id = stable_id("dsgrade", timestamp, model, request_hash)
    conn.execute(
        """
        INSERT INTO dataset_grade_runs (
          id, ts, endpoint, model, prompt_version, item_count,
          error_count, status, request_hash, error
        ) VALUES (?, ?, ?, ?, ?, 0, 0, 'running', ?, NULL)
        """,
        (run_id, timestamp, endpoint, model, grade_cfg.prompt_version, request_hash),
    )
    conn.commit()

    call = transport or _ollama_transport
    graded = 0
    errors = 0
    attempted = 0
    blocked = False
    ledger_pending = False
    error_types: dict[str, int] = {}
    for row in rows:
        attempted += 1
        try:
            record = json.loads(row["example_json"])
            raw = call(
                endpoint,
                model,
                _messages(row["dataset"], record),
                grade_cfg.timeout_seconds,
            )
            grade = normalize_grade(row["dataset"], raw)
            _store_grade(
                conn,
                row,
                grade,
                model=model,
                prompt_version=grade_cfg.prompt_version,
                graded_at=timestamp,
            )
            graded += 1
        except Exception as exc:  # one malformed local response must not lose the batch
            # A failed write must not poison the progress-ledger update below.
            conn.rollback()
            errors += 1
            name = type(exc).__name__
            error_types[name] = error_types.get(name, 0) + 1
            # Model/response failures are deterministic for this grader version
            # and should not poison every future batch. SQLite infrastructure
            # failures are transient and must remain eligible for a normal retry.
            if not isinstance(exc, sqlite3.Error):
                _store_grade_error(
                    conn,
                    row,
                    error_type=name,
                    model=model,
                    prompt_version=grade_cfg.prompt_version,
                    graded_at=timestamp,
                )
        try:
            conn.execute(
                """
                UPDATE dataset_grade_runs
                SET item_count = ?, error_count = ?, error = ?
                WHERE id = ?
                """,
                (
                    attempted,
                    errors,
                    canonical_json(error_types) if error_types else None,
                    run_id,
                ),
            )
            # Release the write lock before the next local inference call. A
            # slow example must not block MCP feedback or scheduled autopilot.
            conn.commit()
        except sqlite3.OperationalError:
            conn.rollback()
            blocked = True
            ledger_pending = True
            break

    status = "blocked" if blocked else "ok" if errors == 0 else "partial" if graded else "error"
    if not ledger_pending:
        try:
            conn.execute(
                "UPDATE dataset_grade_runs SET status = ? WHERE id = ?",
                (status, run_id),
            )
            conn.commit()
        except sqlite3.OperationalError:
            conn.rollback()
            status = "blocked"
            ledger_pending = True
    return {
        "action": "dataset-grade",
        "changed": graded,
        "graded": graded,
        "errors": errors,
        "error_types": error_types,
        "status": status,
        "run_id": run_id,
        "model": model,
        "prompt_version": grade_cfg.prompt_version,
        "local_only": True,
        "daily_items": used_today + attempted,
        "daily_item_cap": grade_cfg.daily_item_cap,
        "repaired_runs": repaired_runs,
        "ledger_pending": ledger_pending,
    }


def grade_examples(
    conn: sqlite3.Connection,
    *,
    cfg: OcbrainConfig | None = None,
    datasets: Sequence[str] | None = None,
    limit: int | None = None,
    endpoint: str | None = None,
    model: str | None = None,
    force: bool = False,
    now: datetime | None = None,
    transport: GradeTransport | None = None,
) -> dict[str, Any]:
    """Acquire the DB-adjacent single-grader lock, then grade a bounded batch."""
    lock_dir = db_side_dir(conn, "locks")
    if lock_dir is None:
        return _grade_examples_unlocked(
            conn,
            cfg=cfg,
            datasets=datasets,
            limit=limit,
            endpoint=endpoint,
            model=model,
            force=force,
            now=now,
            transport=transport,
        )
    with file_lock(lock_dir / "dataset-grade.lock") as acquired:
        if not acquired:
            return {
                "action": "dataset-grade",
                "changed": 0,
                "graded": 0,
                "errors": 0,
                "status": "locked",
                "skipped": "grader_lock",
                "local_only": True,
            }
        return _grade_examples_unlocked(
            conn,
            cfg=cfg,
            datasets=datasets,
            limit=limit,
            endpoint=endpoint,
            model=model,
            force=force,
            now=now,
            transport=transport,
        )
