from __future__ import annotations

import json
from pathlib import Path

from ocbrain.cli import main
from ocbrain.core_v1 import append_core_event, init_core_v1
from ocbrain.db import connect
from ocbrain.mcp_v1 import correct_v1, decide_proposal_v1
from ocbrain.retrieval_eval import build_golden, compare, evaluate
from ocbrain.scope import ScopeTag

MATCHING = "curated:bountiful:alpha-runbook"
UNRELATED = "curated:bountiful:beta-lake"
UNJUDGED = "curated:bountiful:gamma-notes"
QUERY = "alpha migration runbook staged backfills"
OTHER_QUERY = "quarterly planning notes for the team"


def _seed_belief(conn, *, belief_id: str, body: str) -> None:
    scope = ScopeTag(
        "project",
        "project:bountiful",
        visibility="internal",
        egress_policy="hosted_ok",
        provenance="test",
    )
    proposal = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": belief_id,
            "belief_type": "curated_fact",
            "body": body,
            "evidence_ids": [],
            "scope": scope.to_dict(),
            "confidence": 0.9,
            "attributes": {"source_quality": 0.95},
        },
        writer="test",
    )
    decide_proposal_v1(
        conn,
        proposal_event_id=proposal,
        decision="approve",
        actor="test",
        edited_body=None,
        reason="test seed",
    )


def _seed_retrieval(
    conn,
    *,
    use_id: str,
    outcome: str,
    query: str,
    served_ids: list[str],
) -> None:
    conn.execute(
        "INSERT INTO retrieval_uses "
        "(id, served_to_runtime, outcome, query_text, context_json, served_at) "
        "VALUES (?,?,?,?,?,?)",
        (
            use_id,
            "claude-code:test",
            outcome,
            query,
            json.dumps({"project": "bountiful"}),
            "2026-08-04T00:00:00+00:00",
        ),
    )
    for rank, object_id in enumerate(served_ids):
        conn.execute(
            "INSERT INTO retrieval_items VALUES (?,?,?,?,?)",
            (use_id, object_id, "belief", rank, 0.5),
        )


def _healthy_dense_arm(monkeypatch) -> None:
    def fake_neighbors(_conn, _query, *, candidate_ids=None, **_kwargs):
        rows = [
            {"belief_id": belief_id, "similarity": 0.9 if belief_id == MATCHING else 0.0}
            for belief_id in sorted(candidate_ids or [])
        ]
        return rows, None, {}

    monkeypatch.setattr("ocbrain.core_v1.semantic_neighbors", fake_neighbors)


def _seeded_core(tmp_path: Path, monkeypatch):
    _healthy_dense_arm(monkeypatch)
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_belief(
        conn,
        belief_id=MATCHING,
        body="The alpha migration runbook uses staged backfills for the analytics export.",
    )
    _seed_belief(conn, belief_id=UNRELATED, body="The data lake is rooted on local disk.")
    _seed_belief(conn, belief_id=UNJUDGED, body="Quarterly planning notes for the team.")
    _seed_retrieval(
        conn, use_id="ret_helpful", outcome="helpful", query=QUERY, served_ids=[MATCHING]
    )
    _seed_retrieval(
        conn,
        use_id="ret_irrelevant",
        outcome="irrelevant",
        query=OTHER_QUERY,
        served_ids=[UNRELATED],
    )
    _seed_retrieval(conn, use_id="ret_served", outcome="served", query=QUERY, served_ids=[UNJUDGED])
    conn.commit()
    return conn


def test_build_golden_freezes_judged_rows_and_skips_unjudged(tmp_path, monkeypatch) -> None:
    conn = _seeded_core(tmp_path, monkeypatch)
    golden_path = tmp_path / "golden.json"
    result = build_golden(conn, since=None, out_path=golden_path)

    assert result["rows"] == 2
    assert result["by_outcome"] == {"helpful": 1, "irrelevant": 1}
    assert result["skipped"] == {}

    golden = json.loads(golden_path.read_text())
    assert golden["schema_version"] == "ocbrain.retrieval-golden.v1"
    by_id = {row["retrieval_use_id"]: row for row in golden["rows"]}
    assert set(by_id) == {"ret_helpful", "ret_irrelevant"}
    assert "ret_served" not in by_id
    assert by_id["ret_helpful"]["query"] == QUERY
    assert by_id["ret_helpful"]["served_ids"] == [MATCHING]
    assert by_id["ret_helpful"]["context"] == {"project": "bountiful"}
    assert by_id["ret_helpful"]["delivery_target"] == "local_model"
    conn.close()


def test_evaluate_scores_recall_and_negatives_then_drops_retired_ids(
    tmp_path, monkeypatch
) -> None:
    conn = _seeded_core(tmp_path, monkeypatch)
    golden_path = tmp_path / "golden.json"
    build_golden(conn, since=None, out_path=golden_path)
    golden = json.loads(golden_path.read_text())

    report = evaluate(conn, golden, k=12)
    assert report["metrics"]["scorable_positive_rows"] == 1
    assert report["metrics"]["positives_recalled_at_k"] == 1.0
    assert report["metrics"]["mrr_positive"] == 1.0
    assert report["metrics"]["scorable_negative_rows"] == 1
    assert report["metrics"]["negatives_in_top_k"] == 0.0
    assert report["metrics"]["harmful_still_served"] == 0
    assert report["metrics"]["empty_packets"] == 0
    assert report["coverage"]["rows"] == 2

    correct_v1(
        conn,
        layer="belief",
        target=UNRELATED,
        op="retract",
        body="withdrawn",
        actor="human:test",
        hard=False,
    )
    retired = evaluate(conn, golden, k=12)
    assert retired["metrics"]["scorable_negative_rows"] == 0
    assert retired["coverage"]["skipped"]["negative_without_still_serving"] == 1
    assert retired["metrics"]["scorable_positive_rows"] == 1
    conn.close()


def test_compare_reports_signed_deltas(tmp_path, monkeypatch) -> None:
    conn = _seeded_core(tmp_path, monkeypatch)
    golden_path = tmp_path / "golden.json"
    build_golden(conn, since=None, out_path=golden_path)
    golden = json.loads(golden_path.read_text())
    baseline = evaluate(conn, golden, k=12)
    candidate = json.loads(json.dumps(baseline))
    candidate["metrics"]["positives_recalled_at_k"] = 0.5

    deltas = compare(baseline, candidate)
    assert deltas["metrics"]["positives_recalled_at_k"] == {
        "baseline": 1.0,
        "candidate": 0.5,
        "delta": -0.5,
        "sign": "-",
    }
    assert deltas["metrics"]["scorable_positive_rows"]["sign"] == "0"
    conn.close()


def test_cli_build_run_compare_round_trip(tmp_path, monkeypatch, capsys) -> None:
    conn = _seeded_core(tmp_path, monkeypatch)
    conn.close()
    db = tmp_path / "core.sqlite"
    golden_path = tmp_path / "golden.json"
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"

    assert main(["--db", str(db), "retrieval-eval", "build", "--out", str(golden_path)]) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["rows"] == 2

    assert (
        main(
            [
                "--db",
                str(db),
                "retrieval-eval",
                "run",
                "--golden",
                str(golden_path),
                "--out",
                str(baseline_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "--db",
                str(db),
                "retrieval-eval",
                "run",
                "--golden",
                str(golden_path),
                "--out",
                str(candidate_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "--db",
                str(db),
                "retrieval-eval",
                "compare",
                str(baseline_path),
                str(candidate_path),
            ]
        )
        == 0
    )
    compared = json.loads(capsys.readouterr().out)
    assert compared["metrics"]["positives_recalled_at_k"]["delta"] == 0
    assert main(["--db", str(db), "retrieval-eval"]) == 2
