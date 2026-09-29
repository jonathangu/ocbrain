"""Freeze judged retrievals into a golden file and score today's ranker against it.

The verdicts already filed on live retrievals are a corpus-specific test set; the
golden file lives outside the repository because it holds real queries and belief
ids. Scoring counts only items that still serve, so corpus evolution is not
charged to the ranker.
"""

from __future__ import annotations

import hashlib
import json
import random
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
_BREAKDOWN_SECTIONS = ("by_delivery_target", "by_query_length", "by_split")
SPLIT_SALT = "ocbrain-retrieval-eval-split-v1"
TEST_FRACTION = 0.3
TRAIN = "train"
TEST = "test"
HEADROOM_SATURATION = 0.95
HEADROOM_METRIC = "positives_recalled_at_k"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260929
_PER_QUERY_FIELDS = {
    "positives_recalled_at_k": "positive_recall",
    "mrr_positive": "mrr",
    "negatives_in_top_k": "negative_hit",
}


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
                "delivery_target": _delivery_target(context, provenance, row["served_to_runtime"]),
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


def split_of(query_id: str, *, salt: str = SPLIT_SALT, test_fraction: float = TEST_FRACTION) -> str:
    digest = hashlib.sha256(f"{salt}:{query_id}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    return TEST if position < test_fraction else TRAIN


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("retrieval_use_id") or row.get("query") or "")


def split_rows(
    golden: dict[str, Any],
    *,
    salt: str = SPLIT_SALT,
    test_fraction: float = TEST_FRACTION,
) -> dict[str, list[dict[str, Any]]]:
    parts: dict[str, list[dict[str, Any]]] = {TRAIN: [], TEST: []}
    for row in golden.get("rows") or []:
        parts[split_of(_row_id(row), salt=salt, test_fraction=test_fraction)].append(row)
    return parts


def _score(
    conn, golden: dict[str, Any], *, k: int, salt: str, test_fraction: float
) -> dict[str, Any]:
    serving = {
        str(row[0])
        for row in conn.execute(
            "SELECT belief_id FROM current_beliefs WHERE status='current' AND serve=1"
        )
    }
    overall = _Tally()
    by_target: dict[str, _Tally] = defaultdict(_Tally)
    by_bucket: dict[str, _Tally] = defaultdict(_Tally)
    by_split: dict[str, _Tally] = defaultdict(_Tally)
    skipped: dict[str, int] = defaultdict(int)
    per_query: list[dict[str, Any]] = []
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
        entry: dict[str, Any] = {
            "id": _row_id(row),
            "split": split_of(_row_id(row), salt=salt, test_fraction=test_fraction),
            "positive_recall": None,
            "mrr": None,
            "negative_hit": None,
            "top_digest": hashlib.sha256("\n".join(top).encode()).hexdigest()[:12],
        }
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
                entry["positive_recall"] = tally.positives_recalled
                entry["mrr"] = tally.mrr
            else:
                skipped["positive_without_still_serving"] += 1
        elif outcome in NEGATIVE_OUTCOMES:
            if still_serving:
                retrieved = set(top) & set(still_serving)
                tally.negatives = 1
                tally.negatives_hit = len(retrieved) / len(still_serving)
                entry["negative_hit"] = tally.negatives_hit
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
        by_split[entry["split"]].add(tally)
        per_query.append(entry)
    return {
        "metrics": overall.metrics(),
        "by_delivery_target": {key: tally.metrics() for key, tally in sorted(by_target.items())},
        "by_query_length": {key: tally.metrics() for key, tally in sorted(by_bucket.items())},
        "by_split": {name: by_split[name].metrics() for name in (TRAIN, TEST)},
        "coverage": {
            "rows": overall.rows,
            "scorable_rows": overall.positives + overall.negatives,
            "scorable_positive_rows": overall.positives,
            "scorable_negative_rows": overall.negatives,
            "skipped": dict(sorted(skipped.items())),
        },
        "per_query": per_query,
    }


def _noise(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> dict[str, Any]:
    differing: list[str] = []
    wobble = 0.0
    for before, after in zip(first, second, strict=True):
        changed = before["top_digest"] != after["top_digest"]
        for field in _PER_QUERY_FIELDS.values():
            left, right = before[field], after[field]
            if left is None or right is None:
                changed = changed or left != right
            else:
                wobble = max(wobble, abs(left - right))
                changed = changed or left != right
        if changed:
            differing.append(before["id"])
    return {
        "replicates": 2,
        "deterministic": not differing,
        "nondeterministic_queries": differing,
        "max_score_wobble": round(wobble, 6),
    }


def headroom(report: dict[str, Any]) -> dict[str, Any]:
    test = (report.get("by_split") or {}).get(TEST) or {}
    scorable = test.get("scorable_positive_rows", 0)
    score = test.get(HEADROOM_METRIC) if scorable else None
    if score is None:
        advice = "no scorable positive test queries; add cases before climbing anything"
        saturated = False
    elif score >= HEADROOM_SATURATION:
        advice = "climb cost/latency, not quality: test score is saturated"
        saturated = True
    else:
        advice = "quality has headroom on the test split"
        saturated = False
    return {
        "metric": HEADROOM_METRIC,
        "baseline_test_score": score,
        "threshold": HEADROOM_SATURATION,
        "saturated": saturated,
        "advice": advice,
    }


def evaluate(
    conn,
    golden: dict[str, Any],
    *,
    k: int = 12,
    salt: str = SPLIT_SALT,
    test_fraction: float = TEST_FRACTION,
    verify_determinism: bool = True,
) -> dict[str, Any]:
    scored = _score(conn, golden, k=k, salt=salt, test_fraction=test_fraction)
    report: dict[str, Any] = {
        "schema_version": EVAL_SCHEMA,
        "k": k,
        "split": {"salt": salt, "test_fraction": test_fraction},
        **scored,
    }
    if verify_determinism:
        again = _score(conn, golden, k=k, salt=salt, test_fraction=test_fraction)
        report["noise"] = _noise(scored["per_query"], again["per_query"])
    report["headroom"] = headroom(report)
    return report


def train_failures(golden: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    train_ids = {
        entry["id"]
        for entry in report.get("per_query") or []
        if entry.get("split") == TRAIN
        and (
            (entry.get("positive_recall") is not None and entry["positive_recall"] < 1.0)
            or (entry.get("negative_hit") or 0.0) > 0.0
        )
    }
    return [row for row in golden.get("rows") or [] if _row_id(row) in train_ids]


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


def _bootstrap_delta(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "delta": None, "low": None, "high": None, "significant": False}
    diffs = [after - before for before, after in pairs]
    n = len(diffs)
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_RESAMPLES)
    )
    low = means[int(0.025 * BOOTSTRAP_RESAMPLES)]
    high = means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]
    return {
        "n": n,
        "delta": round(sum(diffs) / n, 6),
        "low": round(low, 6),
        "high": round(high, 6),
        "significant": low > 0 or high < 0,
    }


def _confidence_intervals(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    before_rows = {str(e["id"]): e for e in baseline.get("per_query") or []}
    after_rows = {str(e["id"]): e for e in candidate.get("per_query") or []}
    shared = sorted(set(before_rows) & set(after_rows))
    groups: dict[str, list[str]] = {
        "overall": shared,
        TRAIN: [i for i in shared if before_rows[i].get("split") == TRAIN],
        TEST: [i for i in shared if before_rows[i].get("split") == TEST],
    }
    intervals: dict[str, dict[str, dict[str, Any]]] = {}
    for group, ids in groups.items():
        intervals[group] = {}
        for metric, field in _PER_QUERY_FIELDS.items():
            pairs = [
                (before_rows[i][field], after_rows[i][field])
                for i in ids
                if before_rows[i].get(field) is not None and after_rows[i].get(field) is not None
            ]
            intervals[group][metric] = _bootstrap_delta(pairs)
    return intervals


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": COMPARE_SCHEMA,
        "metrics": _compare_metrics(baseline.get("metrics") or {}, candidate.get("metrics") or {}),
    }
    for section in _BREAKDOWN_SECTIONS:
        before = _section_metrics(baseline, section)
        after = _section_metrics(candidate, section)
        result[section] = {
            key: _compare_metrics(before.get(key) or {}, after.get(key) or {})
            for key in sorted(set(before) | set(after))
        }
    if baseline.get("per_query") and candidate.get("per_query"):
        result["confidence_intervals"] = _confidence_intervals(baseline, candidate)
        result["headline"] = TEST
    result["split_mismatch"] = baseline.get("split") != candidate.get("split")
    result["noise"] = {
        "baseline": baseline.get("noise"),
        "candidate": candidate.get("noise"),
    }
    result["headroom"] = headroom(baseline)
    return result
