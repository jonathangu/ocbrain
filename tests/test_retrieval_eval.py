from __future__ import annotations

import json
import random
from pathlib import Path

from ocbrain import retrieval_eval
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


def test_evaluate_scores_recall_and_negatives_then_drops_retired_ids(tmp_path, monkeypatch) -> None:
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


TOPICS = 24


def _topic_belief(index: int) -> str:
    return f"curated:bountiful:topic-{index}"


def _topic_query(index: int) -> str:
    return f"zebra{index} procedure for quokka{index} rollout"


def _topic_core(tmp_path: Path, monkeypatch):
    def fake_neighbors(_conn, query, *, candidate_ids=None, **_kwargs):
        rows = [
            {
                "belief_id": b,
                "similarity": 0.9 if query.startswith(f"zebra{b.rsplit('-', 1)[1]} ") else 0.0,
            }
            for b in sorted(candidate_ids or [])
        ]
        return rows, None, {}

    monkeypatch.setattr("ocbrain.core_v1.semantic_neighbors", fake_neighbors)
    conn = connect(tmp_path / "topics.sqlite")
    init_core_v1(conn)
    for index in range(TOPICS):
        _seed_belief(
            conn,
            belief_id=_topic_belief(index),
            body=f"The zebra{index} procedure stages the quokka{index} rollout in batches.",
        )
        _seed_retrieval(
            conn,
            use_id=f"ret_topic_{index}",
            outcome="helpful",
            query=_topic_query(index),
            served_ids=[_topic_belief(index)],
        )
    conn.commit()
    golden_path = tmp_path / "topics-golden.json"
    build_golden(conn, since=None, out_path=golden_path)
    return conn, json.loads(golden_path.read_text())


def test_split_is_deterministic_salted_and_disjoint() -> None:
    ids = [f"ret_{index}" for index in range(400)]
    first = [retrieval_eval.split_of(i) for i in ids]
    assert first == [retrieval_eval.split_of(i) for i in ids]
    assert set(first) == {"train", "test"}
    share = first.count("test") / len(ids)
    assert 0.2 < share < 0.4
    other = [retrieval_eval.split_of(i, salt="another-salt") for i in ids]
    assert other != first

    golden = {"rows": [{"retrieval_use_id": i, "query": i} for i in ids]}
    parts = retrieval_eval.split_rows(golden)
    assert len(parts["train"]) + len(parts["test"]) == len(ids)
    assert not {r["retrieval_use_id"] for r in parts["train"]} & {
        r["retrieval_use_id"] for r in parts["test"]
    }


def test_evaluate_reports_train_and_test_separately(tmp_path, monkeypatch) -> None:
    conn, golden = _topic_core(tmp_path, monkeypatch)
    report = evaluate(conn, golden, k=12)

    train = report["by_split"]["train"]
    test = report["by_split"]["test"]
    assert train["rows"] + test["rows"] == report["metrics"]["rows"] == TOPICS
    assert train["rows"] > 0
    assert test["rows"] > 0
    assert {entry["split"] for entry in report["per_query"]} == {"train", "test"}
    assert report["split"]["salt"] == retrieval_eval.SPLIT_SALT

    deltas = compare(report, report)
    assert set(deltas["by_split"]) == {"train", "test"}
    assert deltas["headline"] == "test"
    conn.close()


def test_train_failures_never_expose_test_rows(tmp_path, monkeypatch) -> None:
    conn, golden = _topic_core(tmp_path, monkeypatch)
    monkeypatch.setattr(retrieval_eval, "search_core_v1", lambda *a, **k: {"items": []})
    broken = evaluate(conn, golden, k=12)

    failures = retrieval_eval.train_failures(golden, broken)
    test_ids = {e["id"] for e in broken["per_query"] if e["split"] == "test"}
    assert failures
    assert len(failures) == broken["by_split"]["train"]["rows"]
    assert not {row["retrieval_use_id"] for row in failures} & test_ids
    conn.close()


def test_compare_reports_bootstrap_ci_for_test_delta(tmp_path, monkeypatch) -> None:
    conn, golden = _topic_core(tmp_path, monkeypatch)
    baseline = evaluate(conn, golden, k=12)

    same = compare(baseline, baseline)
    interval = same["confidence_intervals"]["test"]["positives_recalled_at_k"]
    assert interval["delta"] == 0
    assert interval["low"] == interval["high"] == 0
    assert interval["significant"] is False

    monkeypatch.setattr(retrieval_eval, "search_core_v1", lambda *a, **k: {"items": []})
    worse = evaluate(conn, golden, k=12)
    drop = compare(baseline, worse)["confidence_intervals"]
    for group in ("overall", "train", "test"):
        entry = drop[group]["positives_recalled_at_k"]
        assert entry["delta"] < 0
        assert entry["high"] < 0
        assert entry["significant"] is True
    assert compare(baseline, worse) == compare(baseline, worse)
    conn.close()


def test_determinism_check_is_clean_then_flags_a_flaky_retriever(tmp_path, monkeypatch) -> None:
    conn, golden = _topic_core(tmp_path, monkeypatch)
    clean = evaluate(conn, golden, k=12)
    assert clean["noise"]["deterministic"] is True
    assert clean["noise"]["nondeterministic_queries"] == []
    assert clean["noise"]["max_score_wobble"] == 0

    real = retrieval_eval.search_core_v1
    calls = {"n": 0}

    def flaky(conn_, query, **kwargs):
        calls["n"] += 1
        result = real(conn_, query, **kwargs)
        if calls["n"] > TOPICS:
            return {"items": []}
        return result

    monkeypatch.setattr(retrieval_eval, "search_core_v1", flaky)
    noisy = evaluate(conn, golden, k=12)
    assert noisy["noise"]["deterministic"] is False
    assert len(noisy["noise"]["nondeterministic_queries"]) == TOPICS
    assert noisy["noise"]["max_score_wobble"] == 1.0
    assert compare(clean, noisy)["noise"]["candidate"]["deterministic"] is False
    conn.close()


def test_negative_control_random_and_empty_retrievers_score_clearly_worse(
    tmp_path, monkeypatch
) -> None:
    conn, golden = _topic_core(tmp_path, monkeypatch)
    real = evaluate(conn, golden, k=3)
    assert real["metrics"]["positives_recalled_at_k"] == 1.0

    pool = [_topic_belief(index) for index in range(TOPICS)]
    rng = random.Random(7)
    monkeypatch.setattr(
        retrieval_eval,
        "search_core_v1",
        lambda *a, **k: {"items": [{"belief_id": b} for b in rng.sample(pool, 3)]},
    )
    shuffled = evaluate(conn, golden, k=3, verify_determinism=False)
    assert shuffled["metrics"]["positives_recalled_at_k"] < 0.5

    monkeypatch.setattr(retrieval_eval, "search_core_v1", lambda *a, **k: {"items": []})
    empty = evaluate(conn, golden, k=3)
    assert empty["metrics"]["positives_recalled_at_k"] == 0.0
    assert empty["metrics"]["empty_packets"] == TOPICS

    for broken in (shuffled, empty):
        verdict = compare(real, broken)["confidence_intervals"]["test"]
        assert verdict["mrr_positive"]["significant"] is True
        assert verdict["mrr_positive"]["high"] < 0
    conn.close()


def test_headroom_says_climb_cost_when_baseline_test_score_is_saturated(
    tmp_path, monkeypatch
) -> None:
    conn, golden = _topic_core(tmp_path, monkeypatch)
    saturated = evaluate(conn, golden, k=12)
    assert saturated["headroom"]["saturated"] is True
    assert "cost/latency" in saturated["headroom"]["advice"]
    assert compare(saturated, saturated)["headroom"]["saturated"] is True

    monkeypatch.setattr(retrieval_eval, "search_core_v1", lambda *a, **k: {"items": []})
    weak = evaluate(conn, golden, k=12)
    assert weak["headroom"]["saturated"] is False
    assert weak["headroom"]["baseline_test_score"] == 0.0
    assert "cost/latency" not in weak["headroom"]["advice"]

    empty_golden = {"rows": []}
    assert retrieval_eval.headroom(evaluate(conn, empty_golden))["baseline_test_score"] is None
    conn.close()
