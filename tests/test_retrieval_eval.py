from __future__ import annotations

import json

from ocbrain.db import connect, init_db
from ocbrain.retrieval_eval import expand_runtime_matrix, run_benchmark


def _belief(conn, belief_id: str, body: str, evidence_id: str):
    conn.execute(
        """
        INSERT INTO current_beliefs (
          belief_id, body, scope_type, scope_id, visibility, egress_policy,
          confidence, confidence_band, evidence_ids, status, pinned,
          approved_event_id, last_event_id, last_compiled_at
        ) VALUES (?, ?, 'project', 'project:ocbrain', 'internal', 'local_only',
                  0.9, 'strong', ?, 'current', 0, 'evt_a', 'evt_a',
                  '2026-07-10T00:00:00+00:00')
        """,
        (belief_id, body, json.dumps([evidence_id])),
    )


def test_benchmark_scores_positive_negative_scope_and_citations(tmp_path):
    conn = connect(tmp_path / "eval.sqlite")
    init_db(conn)
    _belief(conn, "belief_contract", "Learning quality requires frozen evaluation.", "evd_1")
    conn.commit()
    cases = [
        {
            "id": "positive",
            "runtime": "codex",
            "kind": "positive",
            "query": "learning quality evaluation",
            "context": {"project": "ocbrain"},
            "expected_any": ["frozen evaluation"],
        },
        {
            "id": "negative",
            "runtime": "claude",
            "kind": "negative",
            "query": "nonexistent lunar gardening fact",
            "context": {"project": "ocbrain"},
        },
    ]
    path = tmp_path / "benchmark.jsonl"
    path.write_text("\n".join(json.dumps(case) for case in cases) + "\n")

    result = run_benchmark(conn, path, require_cases=2)
    assert result["cases"] == 2
    assert result["top5_rate"] == 1.0
    assert result["citation_rate"] == 1.0
    assert result["scope_rate"] == 1.0
    assert result["contains_corpus_text"] is False
    assert all("body" not in failure for failure in result["failures"])


def test_runtime_matrix_expands_25_cases_to_100(tmp_path):
    base = tmp_path / "base.jsonl"
    base.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"case-{index}",
                    "query": f"question {index}",
                    "context": {"project": "ocbrain"},
                    "kind": "negative",
                }
            )
            + "\n"
            for index in range(25)
        )
    )
    output = tmp_path / "expanded.jsonl"
    result = expand_runtime_matrix(base, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["cases"] == 100
    assert len(rows) == 100
    assert {row["runtime"] for row in rows} == {"codex", "chatgpt", "claude", "openclaw"}
