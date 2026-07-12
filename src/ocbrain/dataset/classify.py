"""Deterministic weights-versus-retrieval classification for training examples."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

from ocbrain.events import canonical_json

TRAIN_CLASSES = {
    "train_voice",
    "train_judgment",
    "train_skill",
    "retrieval_only",
    "exclude",
}


def classify_record(row: sqlite3.Row | dict[str, Any]) -> tuple[str, str]:
    label = str(row["quality_label"] or "")
    reasons = _json_list(row["quality_reasons"])
    if label in {"bad", "excluded"}:
        return "exclude", f"quality_label:{label or 'missing'}"
    if any(
        reason in {"secret_residue", "managed_block_leak", "envelope_residue"} for reason in reasons
    ):
        return "exclude", "hard_quality_contamination"
    if "injection_flagged" in reasons:
        return "exclude", "injection_flagged_training_boundary"

    try:
        record = json.loads(row["example_json"])
    except (TypeError, json.JSONDecodeError):
        return "exclude", "invalid_example_json"
    metadata = record.get("metadata") if isinstance(record, dict) else None
    metadata = metadata if isinstance(metadata, dict) else {}
    dataset = str(row["dataset"])

    if dataset == "persona":
        if metadata.get("sender_verified") is not True:
            return "exclude", "persona_author_not_verified"
        if row["source_kind"] == "git_commit":
            return "train_skill", "verified_human_commit_skill"
        return "train_voice", "verified_human_voice"
    if dataset == "dpo":
        return "train_judgment", "accepted_preference_pair"
    if dataset == "sft":
        if label != "good":
            return "retrieval_only", "sft_not_good"
        return "train_skill", "successful_instruction_exchange"
    return "exclude", "unknown_dataset"


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def classify_examples(
    conn: sqlite3.Connection,
    *,
    force: bool = False,
    limit: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    clauses = [] if force else ["train_class IS NULL"]
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params: list[Any] = []
    limit_sql = ""
    if limit is not None:
        limit_sql = " LIMIT ?"
        params.append(max(0, int(limit)))
    rows = conn.execute(
        "SELECT id, dataset, source_kind, privacy_scope, quality_label, "
        "quality_reasons, example_json FROM dataset_examples "
        f"{where} ORDER BY dataset, id{limit_sql}",
        params,
    ).fetchall()
    timestamp = (now or datetime.now(UTC)).isoformat(timespec="microseconds")
    counts = {name: 0 for name in sorted(TRAIN_CLASSES)}
    for row in rows:
        train_class, reason = classify_record(row)
        if train_class not in TRAIN_CLASSES:  # pragma: no cover - internal invariant
            raise ValueError(f"invalid train class: {train_class}")
        conn.execute(
            """
            UPDATE dataset_examples
            SET train_class = ?, train_class_reason = ?, train_classified_at = ?,
                train_selected = CASE WHEN train_class = ? THEN train_selected ELSE 0 END,
                train_selection_rank = CASE
                    WHEN train_class = ? THEN train_selection_rank ELSE NULL END,
                train_selection_reason = CASE
                    WHEN train_class = ? THEN train_selection_reason ELSE NULL END,
                train_selected_at = CASE WHEN train_class = ? THEN train_selected_at ELSE NULL END,
                updated_at = ?
            WHERE id = ?
            """,
            (
                train_class,
                reason,
                timestamp,
                train_class,
                train_class,
                train_class,
                train_class,
                timestamp,
                row["id"],
            ),
        )
        counts[train_class] += 1
    return {
        "action": "dataset-classify",
        "changed": len(rows),
        "counts": counts,
        "classes": sorted(TRAIN_CLASSES),
        "classifier": "deterministic-v1",
        "local_only": True,
        "selection_hash": canonical_json(counts),
    }
