"""Deterministic, bounded selection of the local v0.4 training pack."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from typing import Any

from ocbrain.events import canonical_json

PACK_CLASSES = {
    "sft": "train_skill",
    "dpo": "train_judgment",
    "persona": "train_voice",
}
DEFAULT_TARGETS = {"sft": 2000, "dpo": 300, "persona": 500}
FINAL_TARGETS = {"sft": 1000, "dpo": 200, "persona": 300}


def _rank(seed: str, dataset: str, row: sqlite3.Row) -> tuple[int, str]:
    priority = {
        "correction_event": 0,
        "authored_doc": 1,
        "git_commit": 1,
        "openclaw_session": 2,
        "claude_session": 2,
        "codex_session": 2,
    }.get(str(row["source_kind"]), 3)
    digest = hashlib.sha256(f"{seed}:{dataset}:{row['id']}".encode()).hexdigest()
    return priority, digest


def select_training_pack(
    conn: sqlite3.Connection,
    *,
    targets: dict[str, int] | None = None,
    seed: str = "ocbrain-v04-selected-pack-v1",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Replace the selected pack with a deterministic, contamination-safe sample."""
    wanted = dict(DEFAULT_TARGETS)
    if targets:
        unknown = set(targets) - set(PACK_CLASSES)
        if unknown:
            raise ValueError(f"unknown pack datasets: {', '.join(sorted(unknown))}")
        wanted.update({key: max(0, int(value)) for key, value in targets.items()})
    timestamp = (now or datetime.now(UTC)).isoformat(timespec="microseconds")
    selected: dict[str, list[sqlite3.Row]] = {}
    for dataset, train_class in PACK_CLASSES.items():
        rows = conn.execute(
            """
            SELECT id, source_kind
            FROM dataset_examples
            WHERE dataset = ? AND train_class = ? AND quality_label = 'good'
              AND privacy_scope != 'private'
              AND instr(COALESCE(quality_reasons, ''), 'injection_flagged') = 0
            """,
            (dataset, train_class),
        ).fetchall()
        selected[dataset] = sorted(rows, key=lambda row: _rank(seed, dataset, row))[
            : wanted[dataset]
        ]

    conn.execute(
        """
        UPDATE dataset_examples
        SET train_selected = 0, train_selection_rank = NULL,
            train_selection_reason = NULL, train_selected_at = NULL
        WHERE train_selected != 0
        """
    )
    for _dataset, rows in selected.items():
        for rank, row in enumerate(rows, 1):
            conn.execute(
                """
                UPDATE dataset_examples
                SET train_selected = 1, train_selection_rank = ?,
                    train_selection_reason = 'deterministic_quality_pack_v1',
                    train_selected_at = ?
                WHERE id = ?
                """,
                (rank, timestamp, row["id"]),
            )
    counts = {dataset: len(rows) for dataset, rows in selected.items()}
    return {
        "action": "dataset-pack-select",
        "selected": counts,
        "targets": wanted,
        "seed": seed,
        "selection_hash": hashlib.sha256(
            canonical_json(
                {dataset: [str(row["id"]) for row in rows] for dataset, rows in selected.items()}
            ).encode()
        ).hexdigest(),
        "local_only": True,
    }


def selected_pack_stats(conn: sqlite3.Connection, *, min_grade: float = 0.8) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT dataset, COUNT(*) AS selected,
               SUM(CASE WHEN grade_score IS NOT NULL THEN 1 ELSE 0 END) AS graded,
               SUM(CASE WHEN grade_score >= ? THEN 1 ELSE 0 END) AS passing
        FROM dataset_examples WHERE train_selected = 1 GROUP BY dataset
        """,
        (min_grade,),
    ).fetchall()
    values = {
        str(row["dataset"]): {
            "selected": int(row["selected"]),
            "graded": int(row["graded"] or 0),
            "passing": int(row["passing"] or 0),
        }
        for row in rows
    }
    for dataset in PACK_CLASSES:
        values.setdefault(dataset, {"selected": 0, "graded": 0, "passing": 0})
    selected_total = sum(item["selected"] for item in values.values())
    graded_total = sum(item["graded"] for item in values.values())
    return {
        "action": "dataset-pack-stats",
        "datasets": values,
        "selected": selected_total,
        "graded": graded_total,
        "grade_coverage": round(graded_total / selected_total, 4) if selected_total else 0.0,
        "min_grade": min_grade,
        "local_only": True,
    }


def finalize_training_pack(
    conn: sqlite3.Connection,
    *,
    targets: dict[str, int] | None = None,
    min_grade: float = 0.8,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Narrow a graded candidate pack to deterministic passing training rows."""
    wanted = dict(FINAL_TARGETS)
    if targets:
        unknown = set(targets) - set(PACK_CLASSES)
        if unknown:
            raise ValueError(f"unknown pack datasets: {', '.join(sorted(unknown))}")
        wanted.update({key: max(0, int(value)) for key, value in targets.items()})
    if not 0.0 <= min_grade <= 1.0:
        raise ValueError("min_grade must be between 0 and 1")

    chosen: dict[str, list[sqlite3.Row]] = {}
    missing: dict[str, dict[str, int]] = {}
    for dataset, train_class in PACK_CLASSES.items():
        rows = conn.execute(
            """
            SELECT id, train_selection_rank
            FROM dataset_examples
            WHERE dataset = ? AND train_class = ? AND train_selected = 1
              AND grade_score >= ?
            ORDER BY COALESCE(train_selection_rank, 2147483647), id
            """,
            (dataset, train_class, min_grade),
        ).fetchall()
        if len(rows) < wanted[dataset]:
            missing[dataset] = {"required": wanted[dataset], "found": len(rows)}
        chosen[dataset] = rows[: wanted[dataset]]
    if missing:
        raise RuntimeError("v0.4 final training pack gate failed: " + canonical_json(missing))

    timestamp = (now or datetime.now(UTC)).isoformat(timespec="microseconds")
    conn.execute(
        """
        UPDATE dataset_examples
        SET train_selected = 0, train_selection_rank = NULL,
            train_selection_reason = NULL, train_selected_at = NULL
        WHERE train_selected = 1
        """
    )
    for _dataset, rows in chosen.items():
        for rank, row in enumerate(rows, 1):
            conn.execute(
                """
                UPDATE dataset_examples
                SET train_selected = 1, train_selection_rank = ?,
                    train_selection_reason = 'deterministic_passing_pack_v1',
                    train_selected_at = ?
                WHERE id = ?
                """,
                (rank, timestamp, row["id"]),
            )
    counts = {dataset: len(rows) for dataset, rows in chosen.items()}
    return {
        "action": "dataset-pack-finalize",
        "selected": counts,
        "targets": wanted,
        "min_grade": min_grade,
        "selection_hash": hashlib.sha256(
            canonical_json(
                {dataset: [str(row["id"]) for row in rows] for dataset, rows in chosen.items()}
            ).encode()
        ).hexdigest(),
        "grade_coverage": 1.0,
        "local_only": True,
    }
