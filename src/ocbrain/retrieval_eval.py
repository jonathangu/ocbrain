"""Freeze judged retrievals into a golden file and score today's ranker against it.

The verdicts already filed on live retrievals are a corpus-specific test set; the
golden file lives outside the repository because it holds real queries and belief
ids. Scoring counts only items that still serve, so corpus evolution is not
charged to the ranker.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ocbrain.core_v1 import (
    RELEVANCE_OUTCOMES,
    SERVED_OUTCOME,
    now_iso,
    search_core_v1,
)
from ocbrain.scope import HOSTED_MODEL_TARGET, LOCAL_MODEL_TARGET, ScopeContext

GOLDEN_SCHEMA = "ocbrain.retrieval-golden.v1"
EVAL_SCHEMA = "ocbrain.retrieval-eval.v1"
COMPARE_SCHEMA = "ocbrain.retrieval-eval-compare.v1"
POSITIVE_OUTCOMES = ("helpful", "used")
NEGATIVE_OUTCOMES = ("irrelevant", "harmful")
QUERY_BUCKETS = ("<40", "40-120", "120-300", "300+")
_METRIC_KEYS = (
    "positives_recalled_at_k",
    "scorable_positive_rows",
    "negatives_in_top_k",
    "scorable_negative_rows",
    "mrr_positive",
    "harmful_still_served",
    "empty_packets",
    "rows",
)
_BREAKDOWN_SECTIONS = ("by_delivery_target", "by_query_length")


def _json_dict(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _delivery_target(context: dict[str, Any], provenance: dict[str, Any], runtime: Any) -> str:
    for source in (provenance, context):
        declared = source.get("delivery_target")
        if declared in {LOCAL_MODEL_TARGET, HOSTED_MODEL_TARGET}:
            return str(declared)
        if declared in {"hosted_teacher", "hosted"}:
            return HOSTED_MODEL_TARGET
    for candidate in (
        provenance.get("runtime"),
        provenance.get("client"),
        context.get("runtime"),
        context.get("client"),
        runtime,
    ):
        if candidate and "hosted" in str(candidate).lower():
            return HOSTED_MODEL_TARGET
    return LOCAL_MODEL_TARGET


def _served_ids(conn, retrieval_use_id: str) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(
            "SELECT object_id FROM retrieval_items "
            "WHERE retrieval_use_id=? AND object_kind='belief' ORDER BY rank",
            (retrieval_use_id,),
        )
    ]


def build_golden(conn, *, since: str | None = None, out_path: Path) -> dict[str, Any]:
    sql = [
        "SELECT id, outcome, query_text, context_json, provenance_json, "
        "served_at, served_to_runtime FROM retrieval_uses WHERE outcome <> ?"
    ]
    params: list[Any] = [SERVED_OUTCOME]
    if since:
        sql.append("AND served_at >= ?")
        params.append(since)
    sql.append("ORDER BY served_at, id")
    counts: dict[str, int] = defaultdict(int)
    skipped: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for row in conn.execute(" ".join(sql), params):
        outcome = str(row["outcome"] or "")
        if outcome not in RELEVANCE_OUTCOMES:
            skipped["unjudged"] += 1
            continue
        query = str(row["query_text"] or "").strip()
        if not query:
            skipped["empty_query"] += 1
            continue
        served = _served_ids(conn, str(row["id"]))
        if not served:
            skipped["no_served_items"] += 1
            continue
        context = _json_dict(row["context_json"])
        provenance = _json_dict(row["provenance_json"])
        rows.append(
            {
                "retrieval_use_id": str(row["id"]),
                "served_at": str(row["served_at"]),
                "query": query,
                "context": context,
                "delivery_target": _delivery_target(
                    context, provenance, row["served_to_runtime"]
                ),
                "outcome": outcome,
                "served_ids": served,
            }
        )
        counts[outcome] += 1
    golden = {
        "schema_version": GOLDEN_SCHEMA,
        "built_at": now_iso(),
        "since": since,
        "rows": rows,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    return {
        "schema_version": GOLDEN_SCHEMA,
        "rows": len(rows),
        "by_outcome": dict(sorted(counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "out_path": str(out_path),
    }


def _bucket(query: str) -> str:
    size = len(query)
    if size < 40:
        return "<40"
    if size <= 120:
        return "40-120"
    if size <= 300:
        return "120-300"
    return "300+"


class _Tally:
    def __init__(self) -> None:
        self.rows = 0
        self.positives = 0
        self.positives_recalled = 0.0
        self.negatives = 0
        self.negatives_hit = 0.0
        self.mrr = 0.0
        self.harmful_still_served = 0
        self.empty_packets = 0

    def add(self, other: _Tally) -> None:
        self.rows += other.rows
        self.positives += other.positives
        self.positives_recalled += other.positives_recalled
        self.negatives += other.negatives
        self.negatives_hit += other.negatives_hit
        self.mrr += other.mrr
        self.harmful_still_served += other.harmful_still_served
        self.empty_packets += other.empty_packets

    def metrics(self) -> dict[str, Any]:
        return {
            "positives_recalled_at_k": _mean(self.positives_recalled, self.positives),
            "scorable_positive_rows": self.positives,
            "negatives_in_top_k": _mean(self.negatives_hit, self.negatives),
            "scorable_negative_rows": self.negatives,
            "mrr_positive": _mean(self.mrr, self.positives),
            "harmful_still_served": self.harmful_still_served,
            "empty_packets": self.empty_packets,
            "rows": self.rows,
        }


def _mean(total: float, count: int) -> float:
    if not count:
        return 0.0
    return round(total / count, 6)


def evaluate(conn, golden: dict[str, Any], *, k: int = 12) -> dict[str, Any]:
    serving = {
        str(row[0])
        for row in conn.execute(
            "SELECT belief_id FROM current_beliefs WHERE status='current' AND serve=1"
        )
    }
    overall = _Tally()
    by_target: dict[str, _Tally] = defaultdict(_Tally)
    by_bucket: dict[str, _Tally] = defaultdict(_Tally)
    skipped: dict[str, int] = defaultdict(int)
    for row in golden.get("rows") or []:
        outcome = str(row.get("outcome") or "")
        query = str(row.get("query") or "")
        target = str(row.get("delivery_target") or LOCAL_MODEL_TARGET)
        served = [str(item) for item in (row.get("served_ids") or [])]
        still_serving = [item for item in served if item in serving]
        result = search_core_v1(
            conn,
            query,
            context=ScopeContext.from_dict(row.get("context")),
            limit=k,
            delivery_target=target,
        )
        top = [str(item.get("belief_id")) for item in result.get("items") or []]
        tally = _Tally()
        tally.rows = 1
        if not top:
            tally.empty_packets = 1
        if outcome in POSITIVE_OUTCOMES:
            if still_serving:
                retrieved = set(top) & set(still_serving)
                tally.positives = 1
                tally.positives_recalled = len(retrieved) / len(still_serving)
                for rank, belief_id in enumerate(top, 1):
                    if belief_id in still_serving:
                        tally.mrr = 1.0 / rank
                        break
            else:
                skipped["positive_without_still_serving"] += 1
        elif outcome in NEGATIVE_OUTCOMES:
            if still_serving:
                retrieved = set(top) & set(still_serving)
                tally.negatives = 1
                tally.negatives_hit = len(retrieved) / len(still_serving)
                if outcome == "harmful" and retrieved:
                    tally.harmful_still_served = 1
            else:
                skipped["negative_without_still_serving"] += 1
        elif outcome == "ignored":
            skipped["ignored"] += 1
        else:
            skipped["unknown_outcome"] += 1
        overall.add(tally)
        by_target[target].add(tally)
        by_bucket[_bucket(query)].add(tally)
    return {
        "schema_version": EVAL_SCHEMA,
        "k": k,
        "metrics": overall.metrics(),
        "by_delivery_target": {
            key: tally.metrics() for key, tally in sorted(by_target.items())
        },
        "by_query_length": {
            key: tally.metrics() for key, tally in sorted(by_bucket.items())
        },
        "coverage": {
            "rows": overall.rows,
            "scorable_rows": overall.positives + overall.negatives,
            "scorable_positive_rows": overall.positives,
            "scorable_negative_rows": overall.negatives,
            "skipped": dict(sorted(skipped.items())),
        },
    }


def _section_metrics(payload: dict[str, Any], section: str) -> dict[str, dict[str, Any]]:
    value = payload.get(section)
    return value if isinstance(value, dict) else {}


def _compare_metrics(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    deltas: dict[str, dict[str, Any]] = {}
    for key in _METRIC_KEYS:
        before = baseline.get(key, 0)
        after = candidate.get(key, 0)
        delta = after - before
        deltas[key] = {
            "baseline": before,
            "candidate": after,
            "delta": round(delta, 6),
            "sign": "+" if delta > 0 else "-" if delta < 0 else "0",
        }
    return deltas


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": COMPARE_SCHEMA,
        "metrics": _compare_metrics(
            baseline.get("metrics") or {}, candidate.get("metrics") or {}
        ),
    }
    for section in _BREAKDOWN_SECTIONS:
        before = _section_metrics(baseline, section)
        after = _section_metrics(candidate, section)
        result[section] = {
            key: _compare_metrics(before.get(key) or {}, after.get(key) or {})
            for key in sorted(set(before) | set(after))
        }
    return result
