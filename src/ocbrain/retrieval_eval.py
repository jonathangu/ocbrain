"""Frozen, privacy-safe retrieval benchmark runner for the shared brain."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from ocbrain.retrieve import retrieve
from ocbrain.scope import ScopeContext, ScopeTag, scope_match
from ocbrain.text import find_probable_injection

RUNTIMES = {"codex", "chatgpt", "claude", "openclaw"}
CASE_KINDS = {"positive", "negative", "injection"}


def expand_runtime_matrix(base_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Expand 25+ runtime-neutral cases across all four supported runtimes."""
    base = Path(base_path).expanduser()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(base.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("id") or not value.get("query"):
            raise ValueError(f"benchmark base line {line_number} is invalid")
        rows.append(value)
    if len(rows) < 25:
        raise ValueError("benchmark base requires at least 25 cases")
    expanded = []
    for runtime in sorted(RUNTIMES):
        for row in rows:
            expanded.append({**row, "id": f"{row['id']}-{runtime}", "runtime": runtime})
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in expanded
    )
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)
    return {
        "action": "retrieval-benchmark-expand",
        "base_cases": len(rows),
        "cases": len(expanded),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "path": str(destination),
        "contains_corpus_text": False,
    }


def load_benchmark(
    path: str | Path, *, require_cases: int = 100
) -> tuple[list[dict[str, Any]], str]:
    source = Path(path).expanduser()
    payload = source.read_bytes()
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"benchmark line {line_number} is not an object")
        case_id = str(value.get("id") or "")
        if not case_id or case_id in seen:
            raise ValueError(f"benchmark line {line_number} has a missing/duplicate id")
        seen.add(case_id)
        if value.get("runtime") not in RUNTIMES:
            raise ValueError(f"benchmark {case_id} has an invalid runtime")
        if value.get("kind", "positive") not in CASE_KINDS:
            raise ValueError(f"benchmark {case_id} has an invalid kind")
        if not isinstance(value.get("query"), str) or not value["query"].strip():
            raise ValueError(f"benchmark {case_id} has no query")
        if not isinstance(value.get("context", {}), dict):
            raise ValueError(f"benchmark {case_id} context is not an object")
        cases.append(value)
    if len(cases) < require_cases:
        raise ValueError(f"frozen benchmark requires at least {require_cases} cases")
    return cases, hashlib.sha256(payload).hexdigest()


def _case_result(conn: sqlite3.Connection, case: dict[str, Any]) -> dict[str, Any]:
    context_data = dict(case.get("context") or {})
    context_data["runtime"] = case["runtime"]
    context = ScopeContext.from_dict(context_data)
    began = time.monotonic()
    payload = retrieve(
        conn,
        case["query"],
        context=context,
        limit=max(1, min(int(case.get("top_k", 5)), 20)),
        cross_scope=bool(case.get("cross_scope", False)),
    )
    latency_ms = round((time.monotonic() - began) * 1000, 3)
    items = payload["items"]
    combined = "\n".join(str(item["body"]).lower() for item in items)
    expected = [str(value).lower() for value in case.get("expected_any", [])]
    forbidden = [str(value).lower() for value in case.get("forbidden", [])]
    kind = case.get("kind", "positive")

    top5_pass = True
    if kind == "positive":
        top5_pass = bool(expected) and any(value in combined for value in expected)
    elif kind == "negative":
        top5_pass = not items or max(float(item["score"]) for item in items) < 0.2
    elif kind == "injection":
        top5_pass = not any(find_probable_injection(str(item["body"])) for item in items)
    forbidden_pass = not any(value in combined for value in forbidden)

    scope_pass = True
    for item in items:
        try:
            tag = ScopeTag.from_dict(item["scope"])
        except (TypeError, ValueError):
            scope_pass = False
            break
        if scope_match(tag, context, cross_scope=bool(case.get("cross_scope", False))) == 0:
            scope_pass = False
            break
    citation_pass = all(
        bool(item.get("evidence_ids")) or bool(item.get("artifact_refs")) for item in items
    )
    latency_pass = latency_ms <= float(case.get("max_latency_ms", 1500))
    passed = top5_pass and forbidden_pass and scope_pass and citation_pass and latency_pass
    return {
        "id": case["id"],
        "runtime": case["runtime"],
        "kind": kind,
        "passed": passed,
        "top5_pass": top5_pass,
        "forbidden_pass": forbidden_pass,
        "scope_pass": scope_pass,
        "citation_pass": citation_pass,
        "latency_pass": latency_pass,
        "latency_ms": latency_ms,
        "result_ids": [str(item["belief_id"]) for item in items],
    }


def run_benchmark(
    conn: sqlite3.Connection,
    path: str | Path,
    *,
    require_cases: int = 100,
) -> dict[str, Any]:
    cases, benchmark_hash = load_benchmark(path, require_cases=require_cases)
    results = [_case_result(conn, case) for case in cases]
    passed = sum(int(item["passed"]) for item in results)
    top5 = sum(int(item["top5_pass"]) for item in results)
    citations = sum(int(item["citation_pass"]) for item in results)
    scopes = sum(int(item["scope_pass"]) for item in results)
    return {
        "action": "retrieval-benchmark",
        "benchmark_sha256": benchmark_hash,
        "cases": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 4),
        "top5_rate": round(top5 / len(results), 4),
        "citation_rate": round(citations / len(results), 4),
        "scope_rate": round(scopes / len(results), 4),
        "latency_ms": {
            "mean": round(sum(item["latency_ms"] for item in results) / len(results), 3),
            "max": max(item["latency_ms"] for item in results),
        },
        "by_runtime": {
            runtime: {
                "cases": len(items),
                "passed": sum(int(item["passed"]) for item in items),
            }
            for runtime in sorted(RUNTIMES)
            if (items := [item for item in results if item["runtime"] == runtime])
        },
        "failures": [item for item in results if not item["passed"]],
        "contains_corpus_text": False,
    }
