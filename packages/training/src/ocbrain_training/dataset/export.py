"""Deterministic, byte-idempotent dataset export (spec §7.5-7.6, lane 5).

Reads the scrubbed JSONL records already stored on ``dataset_examples`` (the
mining lane produced them via ``quality.store_example``) and writes one stable
file per dataset — ``data/datasets/{sft,dpo,persona}.jsonl`` — plus a
``manifest.json``. Because every stored ``example_json`` is canonical JSON and
rows are emitted in a fixed ``(occurred_at, id)`` order, an unchanged corpus
produces byte-identical output; when the new ``payload_hash`` matches the last
``dataset_exports`` row we skip the write entirely.

Every export writes a ``dataset_exports`` ledger row and an ``egress_audits``
row (target ``local_model`` — there is no hosted export path; the dataset never
leaves the machine, spec R6/§7.6). Filters: ``min_label`` (default ``good``),
``min_scope`` (default ``workspace``; ``private`` rows NEVER export regardless
of the flag), optional ``min_grade`` (ungraded rows fail when enabled), and
``--verified-only`` for persona.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ocbrain.config import OcbrainConfig, load_config
from ocbrain.egress import record_egress_audit
from ocbrain.events import canonical_json, sha256_text
from ocbrain.ids import stable_id
from ocbrain.scope import ScopeContext

DATASETS: tuple[str, ...] = ("sft", "dpo", "persona")
DATASET_FORMATS: dict[str, str] = {
    "sft": "chat",
    "persona": "chat",
    "dpo": "openai-preference",
}

# Ratchet ranks. Higher = more shareable. ``private`` is hard-excluded below.
_SCOPE_RANK: dict[str, int] = {"private": 0, "workspace": 1, "project": 2, "public": 3}
# Quality ranks. ``good`` is the strictest default; ``excluded`` never exports.
_LABEL_RANK: dict[str, int] = {"bad": 0, "neutral": 1, "good": 2}


def _config_hash(cfg: OcbrainConfig) -> str:
    return sha256_text(canonical_json(asdict(cfg)))


def _export_context() -> ScopeContext:
    return ScopeContext(runtime="ocbrain-autopilot", task="dataset-export")


def _passes_scope(scope: str, min_scope: str) -> bool:
    if scope == "private":
        return False  # never exports regardless of flags (spec §7.6)
    return _SCOPE_RANK.get(scope, 1) >= _SCOPE_RANK.get(min_scope, 1)


def _passes_label(label: str, min_label: str) -> bool:
    if label not in _LABEL_RANK:  # 'excluded' or unknown
        return False
    return _LABEL_RANK[label] >= _LABEL_RANK.get(min_label, 2)


def _passes_grade(score: float | None, min_grade: float | None) -> bool:
    if min_grade is None:
        return True
    return score is not None and float(score) >= min_grade


def _selected_rows(
    conn: sqlite3.Connection,
    dataset: str,
    *,
    min_scope: str,
    min_label: str,
    min_grade: float | None = None,
    verified_only: bool,
) -> list[str]:
    """Return the ordered list of ``example_json`` blobs that pass the filters."""
    rows = conn.execute(
        """
        SELECT example_json, quality_label, privacy_scope, grade_score
        FROM dataset_examples
        WHERE dataset = ?
        ORDER BY COALESCE(occurred_at, ''), id
        """,
        (dataset,),
    ).fetchall()
    selected: list[str] = []
    for row in rows:
        if not _passes_label(row["quality_label"], min_label):
            continue
        if not _passes_scope(row["privacy_scope"], min_scope):
            continue
        if not _passes_grade(row["grade_score"], min_grade):
            continue
        if verified_only and dataset == "persona":
            try:
                record = json.loads(row["example_json"])
            except (TypeError, ValueError):
                continue
            meta = record.get("metadata") if isinstance(record, dict) else None
            if not (isinstance(meta, dict) and meta.get("sender_verified") is True):
                continue
        selected.append(row["example_json"])
    return selected


def _corpus_stats(conn: sqlite3.Connection, dataset: str) -> dict[str, Any]:
    label_counts: dict[str, int] = {}
    scope_counts: dict[str, int] = {}
    # Advisory injection tally (spec R2): flagged examples STAY in the dataset —
    # quarantine is the enforcement layer — but the manifest reports how many
    # carry the injection_flagged advisory reason, per stream.
    injection_flags = 0
    graded_count = 0
    for row in conn.execute(
        "SELECT quality_label, privacy_scope, quality_reasons, grade_score "
        "FROM dataset_examples WHERE dataset = ?",
        (dataset,),
    ):
        label_counts[row["quality_label"]] = label_counts.get(row["quality_label"], 0) + 1
        scope_counts[row["privacy_scope"]] = scope_counts.get(row["privacy_scope"], 0) + 1
        reasons = row["quality_reasons"]
        if reasons and "injection_flagged" in reasons:
            injection_flags += 1
        if row["grade_score"] is not None:
            graded_count += 1
    return {
        "label_counts": dict(sorted(label_counts.items())),
        "scope_counts": dict(sorted(scope_counts.items())),
        "excluded_count": label_counts.get("excluded", 0),
        "injection_flags": injection_flags,
        "graded_count": graded_count,
        "ungraded_count": sum(label_counts.values()) - graded_count,
    }


def _last_payload_hash(conn: sqlite3.Connection, dataset: str) -> str | None:
    row = conn.execute(
        "SELECT payload_hash FROM dataset_exports WHERE dataset = ? "
        "ORDER BY ts DESC, id DESC LIMIT 1",
        (dataset,),
    ).fetchone()
    return row["payload_hash"] if row else None


def export_dataset(
    conn: sqlite3.Connection,
    dataset: str,
    *,
    cfg: OcbrainConfig,
    export_dir: Path,
    min_scope: str,
    min_label: str,
    min_grade: float | None = None,
    verified_only: bool,
    ts: str,
) -> dict[str, Any]:
    """Export one dataset. Deterministic bytes; skip-if-unchanged by payload_hash."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    lines = _selected_rows(
        conn,
        dataset,
        min_scope=min_scope,
        min_label=min_label,
        min_grade=min_grade,
        verified_only=verified_only,
    )
    # Trailing newline keeps the file POSIX-clean and the byte layout stable.
    payload = ("\n".join(lines) + "\n") if lines else ""
    payload_bytes = payload.encode("utf-8")
    payload_hash = sha256_text(payload)
    path = export_dir / f"{dataset}.jsonl"

    stats = _corpus_stats(conn, dataset)
    fmt = DATASET_FORMATS[dataset]
    unchanged = _last_payload_hash(conn, dataset) == payload_hash and path.exists()

    result: dict[str, Any] = {
        "dataset": dataset,
        "path": str(path),
        "format": fmt,
        "count": len(lines),
        "bytes": len(payload_bytes),
        "sha256": payload_hash,
        "skipped": unchanged,
        "min_grade": min_grade,
        **stats,
    }
    if unchanged:
        return result

    export_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload_bytes)

    audit = {
        "target": "local_model",
        "context": _export_context().to_dict(),
        "query": None,
        "included": [{"dataset": dataset, "count": len(lines)}],
        "rejected": [{"dataset": dataset, "excluded": stats["excluded_count"]}],
        "payload_hash": payload_hash,
    }
    egress_audit_id = record_egress_audit(conn, audit)

    export_id = stable_id("dsexp", dataset, payload_hash, ts)
    conn.execute(
        """
        INSERT OR IGNORE INTO dataset_exports (
          id, ts, dataset, path, min_scope, min_label, min_grade, format,
          example_count, excluded_count, bytes, payload_hash, manifest_json,
          egress_audit_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            export_id,
            ts,
            dataset,
            str(path),
            min_scope,
            min_label,
            min_grade,
            fmt,
            len(lines),
            stats["excluded_count"],
            len(payload_bytes),
            payload_hash,
            canonical_json(result),
            egress_audit_id,
        ),
    )
    result["export_id"] = export_id
    result["egress_audit_id"] = egress_audit_id
    return result


def export_all(
    conn: sqlite3.Connection,
    *,
    cfg: OcbrainConfig | None = None,
    datasets: list[str] | None = None,
    min_scope: str | None = None,
    min_label: str | None = None,
    min_grade: float | None = None,
    verified_only: bool = False,
    export_dir: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Export the requested datasets and (re)write ``manifest.json`` (spec §7.6)."""
    cfg = cfg or load_config()
    wanted = datasets if datasets is not None else list(DATASETS)
    min_scope = min_scope or cfg.dataset.export_min_scope
    min_label = min_label or cfg.dataset.export_min_label
    if min_grade is None:
        min_grade = cfg.dataset.export_min_grade
    if min_grade is not None and not 0.0 <= min_grade <= 1.0:
        raise ValueError("min_grade must be between 0 and 1")
    out_dir = Path(export_dir or cfg.dataset.export_dir).expanduser()
    ts = (now or datetime.now(UTC)).isoformat(timespec="microseconds")

    per_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        if dataset not in wanted:
            continue
        per_dataset[dataset] = export_dataset(
            conn,
            dataset,
            cfg=cfg,
            export_dir=out_dir,
            min_scope=min_scope,
            min_label=min_label,
            min_grade=min_grade,
            verified_only=verified_only,
            ts=ts,
        )

    manifest = {
        "generated_at": ts,
        "config_hash": _config_hash(cfg),
        "min_scope": min_scope,
        "min_label": min_label,
        "min_grade": min_grade,
        "verified_only": verified_only,
        "datasets": {
            name: {
                "path": d["path"],
                "count": d["count"],
                "bytes": d["bytes"],
                "sha256": d["sha256"],
                "format": d["format"],
                "label_counts": d["label_counts"],
                "scope_counts": d["scope_counts"],
                "excluded_count": d["excluded_count"],
                "injection_flags": d["injection_flags"],
                "graded_count": d["graded_count"],
                "ungraded_count": d["ungraded_count"],
            }
            for name, d in per_dataset.items()
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    changed = sum(0 if d["skipped"] else 1 for d in per_dataset.values())
    return {
        "action": "dataset-export",
        "changed": changed,
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "datasets": per_dataset,
    }
