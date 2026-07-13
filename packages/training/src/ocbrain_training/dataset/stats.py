"""Dataset growth reporting (spec §7.6, lane 5).

``dataset_stats`` summarizes the factory's progress toward "the ultimate
dataset": per-dataset counts by quality label, privacy scope, source kind, and
ISO week of ``occurred_at``, plus exclusion/dedup rates and recent export
history. Pure reads — no writes, no egress.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from ocbrain_training.dataset.export import DATASETS


def _group_counts(conn: sqlite3.Connection, dataset: str, column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in conn.execute(
        f"SELECT {column} AS k, COUNT(*) AS n FROM dataset_examples "  # noqa: S608 - fixed identifier
        "WHERE dataset = ? GROUP BY k ORDER BY k",
        (dataset,),
    ):
        key = row["k"] if row["k"] is not None else "(none)"
        counts[str(key)] = int(row["n"])
    return counts


def _by_iso_week(conn: sqlite3.Connection, dataset: str) -> dict[str, int]:
    weeks: dict[str, int] = {}
    for row in conn.execute(
        "SELECT occurred_at FROM dataset_examples WHERE dataset = ? AND occurred_at IS NOT NULL",
        (dataset,),
    ):
        stamp = str(row["occurred_at"])
        # ISO week from the date prefix; robust to trailing time/zone noise.
        try:
            from datetime import date

            year, month, day = (int(part) for part in stamp[:10].split("-"))
            iso_year, iso_week, _ = date(year, month, day).isocalendar()
            key = f"{iso_year:04d}-W{iso_week:02d}"
        except (ValueError, TypeError):
            key = "(unparsed)"
        weeks[key] = weeks.get(key, 0) + 1
    return dict(sorted(weeks.items()))


def _export_history(
    conn: sqlite3.Connection, dataset: str, *, limit: int = 5
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT ts, example_count, excluded_count, bytes, payload_hash "
        "FROM dataset_exports WHERE dataset = ? ORDER BY ts DESC, id DESC LIMIT ?",
        (dataset, limit),
    ).fetchall()
    return [
        {
            "ts": row["ts"],
            "example_count": row["example_count"],
            "excluded_count": row["excluded_count"],
            "bytes": row["bytes"],
            "payload_hash": row["payload_hash"],
        }
        for row in rows
    ]


def dataset_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return a growth report over all datasets (spec §7.6)."""
    datasets: dict[str, Any] = {}
    totals = {"total": 0, "exported_eligible": 0, "excluded": 0}
    for dataset in DATASETS:
        label_counts = _group_counts(conn, dataset, "quality_label")
        scope_counts = _group_counts(conn, dataset, "privacy_scope")
        source_counts = _group_counts(conn, dataset, "source_kind")
        train_class_counts = _group_counts(conn, dataset, "train_class")
        total = sum(label_counts.values())
        excluded = label_counts.get("excluded", 0)
        good = label_counts.get("good", 0)
        graded = conn.execute(
            "SELECT COUNT(*) AS n FROM dataset_examples "
            "WHERE dataset = ? AND grade_score IS NOT NULL",
            (dataset,),
        ).fetchone()["n"]
        grade_row = conn.execute(
            "SELECT AVG(grade_score) AS avg_score, MIN(grade_score) AS min_score, "
            "MAX(grade_score) AS max_score FROM dataset_examples "
            "WHERE dataset = ? AND grade_score IS NOT NULL",
            (dataset,),
        ).fetchone()
        selected = sum(
            count for name, count in train_class_counts.items() if name.startswith("train_")
        )
        selected_graded = conn.execute(
            "SELECT COUNT(*) AS n FROM dataset_examples WHERE dataset = ? "
            "AND train_class LIKE 'train_%' AND grade_score IS NOT NULL",
            (dataset,),
        ).fetchone()["n"]
        selected_passing = conn.execute(
            "SELECT COUNT(*) AS n FROM dataset_examples WHERE dataset = ? "
            "AND train_class LIKE 'train_%' AND grade_score >= 0.8",
            (dataset,),
        ).fetchone()["n"]
        datasets[dataset] = {
            "total": total,
            "by_label": label_counts,
            "by_scope": scope_counts,
            "by_source_kind": source_counts,
            "by_train_class": train_class_counts,
            "by_iso_week": _by_iso_week(conn, dataset),
            "excluded": excluded,
            "exclusion_rate": round(excluded / total, 4) if total else 0.0,
            "good": good,
            "good_rate": round(good / total, 4) if total else 0.0,
            "graded": int(graded),
            "grade_coverage": round(int(graded) / total, 4) if total else 0.0,
            "selected_for_weights": selected,
            "selected_graded": int(selected_graded),
            "selected_grade_coverage": (
                round(int(selected_graded) / selected, 4) if selected else 0.0
            ),
            "selected_passing_0_8": int(selected_passing),
            "grade_score": {
                "average": round(float(grade_row["avg_score"]), 4)
                if grade_row["avg_score"] is not None
                else None,
                "minimum": grade_row["min_score"],
                "maximum": grade_row["max_score"],
            },
            "export_history": _export_history(conn, dataset),
        }
        totals["total"] += total
        totals["excluded"] += excluded
        totals["exported_eligible"] += good
    return {"action": "dataset-stats", "datasets": datasets, "totals": totals}
