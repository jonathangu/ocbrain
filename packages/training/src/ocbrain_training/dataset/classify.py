"""Deterministic weights-versus-retrieval classification for training examples."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from typing import Any

from ocbrain.events import canonical_json
from ocbrain.text import claim_key

from ocbrain_training.dataset.quality import sanitize_sft_text, scrub_reasons

TRAIN_CLASSES = {
    "train_voice",
    "train_judgment",
    "train_skill",
    "retrieval_only",
    "exclude",
}

DPO_CONTRAST_GATE_VERSION = 2

_RUNTIME_WRAPPER_RE = re.compile(
    r"(?is)(?:"
    r"\[\[\s*reply_to_current\s*\]\]|"
    r"(?:Conversation info|Sender|Replied message)\s*"
    r"\(untrusted(?: metadata)?(?:,\s*for context)?\)|"
    r"System\s*\(untrusted\)\s*:|"
    r"<goal_context\b|</goal_context\s*>|"
    r"<<<\s*(?:BEGIN|END)_OPENCLAW_INTERNAL_CONTEXT\s*>>>|"
    r"<openclawbrain_context\b|</openclawbrain_context\s*>|"
    r"Pre-compaction memory flush\b|"
    r"Read\s+HEARTBEAT\.md\s+if\s+it\s+exists\b|"
    r"\[cron:[^\]]+\]|"
    r"Warning:\s*(?:apply_patch was requested|"
    r"The maximum number of unified exec processes)|"
    r"A new session was started via /(?:new|reset)\b"
    r")"
)


def _content_strings(value: Any) -> list[str]:
    """Return model-visible content strings, excluding provenance metadata."""
    if isinstance(value, dict):
        strings: list[str] = []
        for key, item in value.items():
            if key == "metadata":
                continue
            if key == "content" and isinstance(item, str):
                strings.append(item)
            else:
                strings.extend(_content_strings(item))
        return strings
    if isinstance(value, list):
        strings: list[str] = []
        for item in value:
            strings.extend(_content_strings(item))
        return strings
    return []


def _assistant_content(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    for message in reversed(value):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None


def _training_target(dataset: str, record: dict[str, Any]) -> str | None:
    if dataset == "dpo":
        return _assistant_content(record.get("preferred_output"))
    return _assistant_content(record.get("messages"))


def training_record_issues(dataset: str, record: dict[str, Any]) -> list[str]:
    """Re-evaluate a stored row against the current training boundary.

    Stored examples are derived data and can predate today's parser/sanitizer.
    Selection and pilot preparation therefore inspect their model-visible body
    again instead of trusting historical labels or a high local-model grade.
    """
    if dataset not in {"sft", "dpo", "persona"} or not isinstance(record, dict):
        return ["invalid_training_record"]

    body = {key: value for key, value in record.items() if key != "metadata"}
    target = _training_target(dataset, record)
    if target is None:
        return ["missing_training_target"]

    issues: list[str] = []
    contents = _content_strings(body)
    if any(_RUNTIME_WRAPPER_RE.search(content) for content in contents):
        issues.append("runtime_wrapper_residue")
    if any(sanitize_sft_text(content) != content.strip() for content in contents):
        issues.append("legacy_unsanitized_content")

    # Reuse the current quality policy so a newly added hard scrub immediately
    # protects already-derived rows too. Injection is a hard training boundary
    # at classification/selection even though storage records it as advisory.
    for reason in scrub_reasons(
        target,
        canonical_json(body),
        dataset=dataset,
    ):
        if reason not in issues:
            issues.append(reason)

    if dataset == "dpo":
        metadata = record.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        try:
            gate_version = int(metadata.get("contrast_gate_version") or 0)
        except (TypeError, ValueError):
            gate_version = 0
        if gate_version < DPO_CONTRAST_GATE_VERSION:
            issues.append("legacy_dpo_contrast_gate")

        rejected = _assistant_content(record.get("non_preferred_output"))
        input_value = record.get("input")
        prompt = input_value.get("messages") if isinstance(input_value, dict) else None
        if rejected is None or not isinstance(prompt, list) or not prompt:
            issues.append("invalid_dpo_shape")
        elif claim_key(target) == claim_key(rejected):
            issues.append("dpo_no_meaningful_contrast")

    return list(dict.fromkeys(issues))


def classify_record(row: sqlite3.Row | dict[str, Any]) -> tuple[str, str]:
    label = str(row["quality_label"] or "")
    reasons = _json_list(row["quality_reasons"])
    if label in {"bad", "excluded"}:
        return "exclude", f"quality_label:{label or 'missing'}"
    if any(
        reason
        in {
            "secret_residue",
            "managed_block_leak",
            "envelope_residue",
            "process_chatter",
            "privacy_residue",
            "runtime_wrapper_residue",
            "dpo_contrast_ambiguous",
        }
        for reason in reasons
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
    issues = training_record_issues(dataset, record)
    if issues:
        return "exclude", f"current_training_boundary:{issues[0]}"

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
        "classifier": "deterministic-v2",
        "local_only": True,
        "selection_hash": canonical_json(counts),
    }
