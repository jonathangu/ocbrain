from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocbrain.db import counts, now_iso
from ocbrain.temporal import temporal_supersession_groups
from ocbrain.text import find_probable_secret_leaks

GENERIC_BODY_MARKERS = (
    "artifact appears to",
    "route to",
    "create a patch suggestion only",
    "extract only concise",
    "no strong memory/wiki/skill/policy",
)


@dataclass(frozen=True)
class SampleSpec:
    sample_size: int = 100
    seed: int = 20260621
    targets: tuple[str, ...] = ()
    per_target: int | None = None


def evaluate(
    conn,
    spec: SampleSpec,
    *,
    db_label: str,
    sample_output_limit: int = 200,
) -> dict[str, Any]:
    rows = sample_candidates(conn, spec)
    item_scores = [score_candidate(row) for row in rows]
    duplicate_report = duplicate_summary(conn)
    temporal_report = temporal_summary(conn)
    search_report = search_index_summary(conn)
    leakage = leakage_summary(item_scores)
    dimension_scores = aggregate_dimensions(
        item_scores,
        duplicate_report,
        temporal_report,
        search_report,
    )
    overall = sum(dimension_scores.values()) / len(dimension_scores) if dimension_scores else 0.0
    findings = collect_findings(item_scores, duplicate_report, temporal_report, search_report)
    return {
        "run": {
            "created_at": now_iso(),
            "db": db_label,
            "sample": {
                "sample_size": spec.sample_size,
                "seed": spec.seed,
                "targets": list(spec.targets),
                "per_target": spec.per_target,
            },
        },
        "baseline": counts(conn),
        "summary": {
            "overall_score": round(overall, 3),
            "verdict": verdict(overall, leakage["probable_secret_count"]),
            "dimension_scores": {key: round(value, 3) for key, value in dimension_scores.items()},
        },
        "findings": findings[:50],
        "sampled_candidates": item_scores[:sample_output_limit],
        "leakage": leakage,
        "duplicates": duplicate_report,
        "temporal": temporal_report,
        "search_index": search_report,
    }


def sample_candidates(conn, spec: SampleSpec) -> list[Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if spec.targets:
        placeholders = ",".join("?" for _ in spec.targets)
        clauses.append(f"target IN ({placeholders})")
        params.extend(spec.targets)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    clauses.append("status != 'stale'")
    where = f"WHERE {' AND '.join(clauses)}"
    rows = list(conn.execute(f"SELECT * FROM candidates {where} ORDER BY id", params))
    rng = random.Random(spec.seed)
    if spec.per_target is not None:
        grouped: dict[str, list[Any]] = defaultdict(list)
        for row in rows:
            grouped[row["target"]].append(row)
        selected: list[Any] = []
        for target in sorted(grouped):
            target_rows = grouped[target]
            rng.shuffle(target_rows)
            selected.extend(target_rows[: spec.per_target])
        return selected
    rng.shuffle(rows)
    return rows[: spec.sample_size]


def score_candidate(row) -> dict[str, Any]:
    evidence = json.loads(row["evidence_json"] or "[]")
    hints = json.loads(row["hints_json"] or "[]")
    text = "\n".join(
        [
            row["title"],
            row["body"],
            row["evidence_json"],
        ]
    )
    leaks = find_probable_secret_leaks(text)
    dimensions = {
        "target_correctness": target_score(row, hints),
        "evidence_grounding": evidence_score(evidence, row),
        "confidence_calibration": confidence_score(row),
        "redaction_leakage": 0.0 if leaks else 1.0,
        "review_actionability": actionability_score(row),
        "scope_safety": scope_score(row),
    }
    score = sum(dimensions.values()) / len(dimensions)
    notes = notes_for(row, evidence, hints, leaks, dimensions)
    return {
        "candidate_id": row["id"],
        "event_id": row["event_id"],
        "target": row["target"],
        "status": row["status"],
        "risk": row["risk"],
        "confidence": row["confidence"],
        "score": round(score, 3),
        "verdict": verdict(score, len(leaks)),
        "dimensions": {key: round(value, 3) for key, value in dimensions.items()},
        "notes": notes,
    }


def target_score(row, hints: list[str]) -> float:
    if row["target"] == "policy":
        return 1.0 if row["risk"] == "high" and "patch-suggestion-only" in hints else 0.2
    if row["target"] == "skill":
        return 1.0 if row["risk"] in {"medium", "high"} and "proposal-first" in hints else 0.4
    if row["target"] == "ignore":
        return 0.7
    return 0.8


def evidence_score(evidence: list[dict[str, Any]], row) -> float:
    if not evidence:
        return 0.0
    best = 0.0
    title_terms = meaningful_terms(row["title"])
    body_terms = meaningful_terms(claim_text_from_body(row["body"]))
    for item in evidence:
        excerpt = item.get("excerpt", "")
        uri = item.get("uri", "")
        if not excerpt or not uri:
            best = max(best, 0.2)
            continue
        excerpt_terms = meaningful_terms(excerpt)
        title_overlap = len(title_terms & excerpt_terms)
        body_overlap = len(body_terms & excerpt_terms)
        score = 0.5
        if title_overlap:
            score += min(title_overlap, 3) * 0.12
        if body_overlap:
            score += min(body_overlap, 4) * 0.035
        best = max(best, score)
    return min(best, 1.0)


def confidence_score(row) -> float:
    confidence = float(row["confidence"])
    if confidence < 0.0 or confidence > 1.0:
        return 0.0
    target = row["target"]
    risk = row["risk"]
    status = row["status"]

    if target == "ignore":
        low, high = 0.35, 0.6
    elif risk == "high":
        low, high = 0.45, 0.65
    elif target == "skill":
        low, high = 0.55, 0.72
    else:
        low, high = 0.62, 0.82

    if status in {"approved", "applied"}:
        high = min(0.95, high + 0.12)
        low = min(low + 0.08, high)

    if low <= confidence <= high:
        return 1.0

    distance = low - confidence if confidence < low else confidence - high
    if distance <= 0.1:
        return 0.7
    if distance <= 0.25:
        return 0.4
    return 0.1


def actionability_score(row) -> float:
    body = row["body"].lower()
    if any(marker in body for marker in GENERIC_BODY_MARKERS):
        return 0.4
    if len(row["body"].split()) < 6:
        return 0.3
    return 0.8


def scope_score(row) -> float:
    if row["scope"] == "private" and row["status"] not in {"approved", "applied"}:
        return 0.3
    if row["risk"] == "high" and row["status"] not in {"approved", "applied"}:
        return 0.6
    return 1.0


def notes_for(
    row,
    evidence: list[dict[str, Any]],
    hints: list[str],
    leaks: list[str],
    dimensions: dict[str, float],
) -> list[str]:
    notes: list[str] = []
    if leaks:
        notes.append(f"probable secret leakage: {', '.join(leaks)}")
    if not evidence:
        notes.append("missing evidence")
    if dimensions["review_actionability"] < 0.7:
        notes.append("candidate body is generic or not directly approvable")
    if dimensions["confidence_calibration"] < 0.7:
        notes.append("candidate confidence is outside calibrated target/risk/status band")
    if dimensions["evidence_grounding"] < 0.7:
        notes.append("candidate title/body weakly align with evidence excerpt")
    if row["target"] == "policy" and "patch-suggestion-only" not in hints:
        notes.append("policy candidate lacks patch-suggestion-only hint")
    if row["scope"] == "private" and row["status"] not in {"approved", "applied"}:
        notes.append("private scope is not approved for serving")
    return notes


def duplicate_summary(conn) -> dict[str, Any]:
    rows = list(
        conn.execute(
            """
            SELECT
              target,
              CASE
                WHEN claim_key IS NOT NULL AND claim_key != '' THEN claim_key
                ELSE body
              END AS duplicate_key,
              MIN(body) AS body,
              COUNT(*) AS count
            FROM candidates
            WHERE target != 'ignore'
              AND status != 'stale'
            GROUP BY target, duplicate_key
            HAVING COUNT(*) > 1
            ORDER BY count DESC
            LIMIT 20
            """
        )
    )
    clusters = [
        {
            "target": row["target"],
            "count": row["count"],
            "body_template": row["body"][:160],
            "severity": "fail" if row["count"] > 100 else "warn",
        }
        for row in rows
    ]
    return {
        "cluster_count_sample": len(clusters),
        "largest_cluster_size": clusters[0]["count"] if clusters else 0,
        "top_clusters": clusters,
    }


def temporal_summary(conn) -> dict[str, Any]:
    groups = temporal_supersession_groups(conn, limit=50)
    return {
        "stale_risk_count_sample": sum(len(group["stale_candidate_ids"]) for group in groups),
        "supersession_group_count_sample": len(groups),
        "sample": groups[:10],
    }


def search_index_summary(conn) -> dict[str, Any]:
    event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    index_count = conn.execute("SELECT COUNT(*) FROM search_index").fetchone()[0]
    orphan_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM search_index s
        LEFT JOIN events e ON e.id = s.doc_id
        WHERE e.id IS NULL
        """
    ).fetchone()[0]
    return {
        "events": event_count,
        "search_index_rows": index_count,
        "orphan_rows": orphan_count,
        "consistent": event_count == index_count and orphan_count == 0,
    }


def leakage_summary(scores: list[dict[str, Any]]) -> dict[str, Any]:
    leak_notes = [
        note
        for score in scores
        for note in score["notes"]
        if note.startswith("probable secret leakage:")
    ]
    return {
        "probable_secret_count": len(leak_notes),
        "sample_notes": leak_notes[:10],
    }


def aggregate_dimensions(
    scores: list[dict[str, Any]],
    duplicate_report: dict[str, Any],
    temporal_report: dict[str, Any],
    search_report: dict[str, Any],
) -> dict[str, float]:
    aggregate: dict[str, float] = {}
    dimension_names = sorted({name for score in scores for name in score["dimensions"]})
    for name in dimension_names:
        values = [score["dimensions"][name] for score in scores]
        aggregate[name] = sum(values) / len(values) if values else 0.0
    aggregate["duplicate_detection"] = (
        0.2 if duplicate_report["largest_cluster_size"] > 100 else 0.8
    )
    aggregate["temporal_invalidation"] = (
        0.5 if temporal_report["stale_risk_count_sample"] else 0.9
    )
    aggregate["search_index_consistency"] = 1.0 if search_report["consistent"] else 0.0
    return aggregate


def collect_findings(
    scores: list[dict[str, Any]],
    duplicate_report: dict[str, Any],
    temporal_report: dict[str, Any],
    search_report: dict[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for score in scores:
        for note in score["notes"]:
            findings.append(
                {
                    "severity": "fail" if "secret leakage" in note else "warn",
                    "candidate_id": score["candidate_id"],
                    "target": score["target"],
                    "message": note,
                }
            )
    if duplicate_report["largest_cluster_size"] > 100:
        findings.append(
            {
                "severity": "fail",
                "dimension": "duplicate_detection",
                "message": (
                    "largest duplicate body cluster has "
                    f"{duplicate_report['largest_cluster_size']} candidates"
                ),
            }
        )
    if temporal_report["stale_risk_count_sample"]:
        findings.append(
            {
                "severity": "warn",
                "dimension": "temporal_invalidation",
                "message": (
                    f"{temporal_report['stale_risk_count_sample']} sampled operational "
                    "facts need stale review"
                ),
            }
        )
    if not search_report["consistent"]:
        findings.append(
            {
                "severity": "fail",
                "dimension": "search_index_consistency",
                "message": "search_index rows do not match events",
            }
        )
    severity_rank = {"fail": 0, "warn": 1, "info": 2}
    return sorted(findings, key=lambda item: severity_rank.get(item["severity"], 9))


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ocbrain Eval Report",
        "",
        f"- Created: `{report['run']['created_at']}`",
        f"- DB: `{report['run']['db']}`",
        f"- Verdict: `{report['summary']['verdict']}`",
        f"- Overall score: `{report['summary']['overall_score']}`",
        "",
        "## Baseline",
        "",
        f"- Events: `{report['baseline']['events']}`",
        f"- Candidates: `{report['baseline']['candidates']}`",
        f"- By target: `{json.dumps(report['baseline']['by_target'], sort_keys=True)}`",
        "",
        "## Dimension Scores",
        "",
    ]
    for name, score in sorted(report["summary"]["dimension_scores"].items()):
        lines.append(f"- `{name}`: `{score}`")
    lines += ["", "## Top Findings", ""]
    for finding in report["findings"][:20]:
        candidate = f" `{finding['candidate_id']}`" if "candidate_id" in finding else ""
        lines.append(f"- `{finding['severity']}`{candidate}: {finding['message']}")
    if not report["findings"]:
        lines.append("- No findings.")
    lines += ["", "## Duplicate Clusters", ""]
    for cluster in report["duplicates"]["top_clusters"][:10]:
        line = (
            f"- `{cluster['severity']}` {cluster['target']} x{cluster['count']}: "
            f"{cluster['body_template']}"
        )
        lines.append(line)
    if not report["duplicates"]["top_clusters"]:
        lines.append("- No duplicate body clusters in sample.")
    lines += ["", "## Sampled Candidates", ""]
    for item in report["sampled_candidates"][:30]:
        note_text = ", ".join(item["notes"]) or "ok"
        lines.append(
            f"- `{item['candidate_id']}` `{item['target']}` "
            f"score `{item['score']}`: {note_text}"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_reports(
    report: dict[str, Any],
    *,
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> None:
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(report), encoding="utf-8")


def meaningful_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]{4,}", text.lower())
        if term not in {"candidate", "artifact", "appears", "route", "source", "backed"}
    }


def claim_text_from_body(body: str) -> str:
    return re.sub(
        r"(?i)^(draft wiki synthesis|stage operational fact|draft repeatable workflow|"
        r"patch-suggestion constraint) from source:\s*",
        "",
        body,
    )


def verdict(score: float, probable_secret_count: int = 0) -> str:
    if probable_secret_count:
        return "fail"
    if score >= 0.8:
        return "pass"
    if score >= 0.5:
        return "warn"
    return "fail"
