"""Direct tests for the duplicate gate's own reader, `document_neighbors`.

This file exists because the instrument the duplicate gate rests on shipped with
no test of its own: every test of the gate's positive path monkeypatches
`document_neighbors` out, so reverting it to exactly the blindness it was written
to fix left the whole suite green. What is pinned here is the difference between
this reader and `semantic_neighbors`, on a real sidecar rather than a stub:

* it answers while the whole-corpus fingerprint is stale (the state a curation
  cycle is in from its first write onward);
* it refuses a candidate's stored vector when that candidate's own body no
  longer hashes to it, and re-embeds instead of scoring against a dead vector;
* it embeds at most `embed_budget` such candidates and reports the rest as
  `uncovered` rather than dropping them quietly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ocbrain.core_v1 import append_core_event, init_core_v1
from ocbrain.db import connect
from ocbrain.hybrid import (
    build_vector_index,
    document_neighbors,
    semantic_neighbors,
)
from ocbrain.mcp_v1 import decide_proposal_v1
from ocbrain.scope import ScopeTag

CITRUS = "Meyer lemons are ready in the citrus grove."
TOMATO = "Tomatoes are available."
PEAR = "Pears are available."


def _seed(conn, *, belief_id: str, body: str) -> None:
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
            "belief_type": "wiki_fact",
            "body": body,
            "evidence_ids": [],
            "scope": scope.to_dict(),
            "confidence": 0.9,
            "attributes": {"key": belief_id.split(":")[-1], "source_quality": 0.95},
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


def _fake_embed(texts, **_kwargs):
    """Two-dimensional stand-in: citrus points one way, everything else the other."""
    out = []
    for text in texts:
        lowered = str(text).lower()
        out.append([1.0, 0.0] if ("citrus" in lowered or "lemon" in lowered) else [0.0, 1.0])
    return out


@pytest.fixture()
def local_embedder(monkeypatch):
    monkeypatch.setenv("OCBRAIN_EMBED_MODEL", "test-local")
    monkeypatch.setenv("OCBRAIN_EMBED_DIMENSIONS", "2")
    monkeypatch.setattr("ocbrain.hybrid.embed_texts", _fake_embed)
    monkeypatch.setattr(
        "ocbrain.hybrid._ollama_model_metadata",
        lambda *_args, **_kwargs: {"digest": "sha256:test-model-v1"},
    )


def _built_core(tmp_path: Path, bodies: dict[str, str]):
    path = tmp_path / "core.sqlite"
    conn = connect(path)
    init_core_v1(conn)
    for belief_id, body in bodies.items():
        _seed(conn, belief_id=belief_id, body=body)
    conn.commit()
    built = build_vector_index(path, model="test-local")
    assert built["rows"] == len(bodies)
    return conn, path


def _rewrite_body(conn, belief_id: str, body: str) -> None:
    """Make one belief's body stop matching its stored vector.

    Straight to the table on purpose: what is being reproduced is the *state*
    the sidecar is in mid-cycle -- a stored vector whose belief has moved on --
    not any particular route into that state.
    """
    conn.execute("UPDATE current_beliefs SET body=? WHERE belief_id=?", (body, belief_id))
    conn.commit()


def test_document_reader_answers_while_the_corpus_fingerprint_is_stale(
    tmp_path: Path, local_embedder
) -> None:
    """The gate must keep working from the first write of a cycle onward.

    `semantic_neighbors` requires a fresh whole-corpus fingerprint and correctly
    refuses here. If `document_neighbors` required it too, the duplicate gate
    would switch itself off for the rest of every curation run -- which is the
    defect this reader was added to fix.
    """
    conn, _path = _built_core(
        tmp_path,
        {
            "curated:bountiful:citrus": CITRUS,
            "curated:bountiful:tomato": TOMATO,
            "curated:bountiful:pear": PEAR,
        },
    )
    ids = ["curated:bountiful:citrus", "curated:bountiful:tomato", "curated:bountiful:pear"]
    _rewrite_body(conn, "curated:bountiful:tomato", "Tomatoes are available in the shed.")

    # Control: the query-side reader is now blind, and says so.
    _, query_side_reason = semantic_neighbors(conn, "citrus", candidate_ids=ids)
    assert query_side_reason == "vector_sidecar_stale"

    neighbors, unavailable, coverage = document_neighbors(conn, CITRUS, candidate_ids=ids)
    assert unavailable is None
    assert coverage == {"candidates": 3, "reused": 2, "embedded": 1, "uncovered": 0}
    assert [row["belief_id"] for row in neighbors][0] == "curated:bountiful:citrus"


def test_a_candidate_whose_body_moved_is_rescored_not_read_from_its_dead_vector(
    tmp_path: Path, local_embedder
) -> None:
    """Per-candidate `content_hash` verification, measured by the score itself.

    `curated:bountiful:tomato` was embedded away from citrus and is then
    rewritten to be a near-copy of the citrus fact. Trusting its stored vector
    would score it 0.0 against a citrus claim -- the gate would wave through a
    restatement of a belief it had just been handed.
    """
    conn, _path = _built_core(
        tmp_path,
        {
            "curated:bountiful:citrus": CITRUS,
            "curated:bountiful:tomato": TOMATO,
        },
    )
    ids = ["curated:bountiful:citrus", "curated:bountiful:tomato"]
    _rewrite_body(conn, "curated:bountiful:tomato", "Citrus lemons are ready in the grove.")

    neighbors, unavailable, coverage = document_neighbors(conn, CITRUS, candidate_ids=ids)
    assert unavailable is None
    assert coverage == {"candidates": 2, "reused": 1, "embedded": 1, "uncovered": 0}
    scores = {row["belief_id"]: row["similarity"] for row in neighbors}
    assert scores["curated:bountiful:tomato"] == pytest.approx(1.0)


def test_named_but_not_serving_candidates_leave_the_count_honest(
    tmp_path: Path, local_embedder
) -> None:
    """A candidate id that names nothing serving is not an uncovered comparison."""
    conn, _path = _built_core(tmp_path, {"curated:bountiful:citrus": CITRUS})
    _neighbors, unavailable, coverage = document_neighbors(
        conn, CITRUS, candidate_ids=["curated:bountiful:citrus", "curated:bountiful:ghost"]
    )
    assert unavailable is None
    assert coverage == {"candidates": 1, "reused": 1, "embedded": 0, "uncovered": 0}


def test_embed_budget_bounds_the_work_and_reports_what_it_could_not_compare(
    tmp_path: Path, local_embedder
) -> None:
    """Past the budget, candidates are reported as `uncovered`, never dropped.

    This is the availability cliff: an incomplete comparison must be visible to
    the caller, because `near_duplicate_neighbor` turns it into unavailability
    and the fail-closed fallback pends the claim.
    """
    bodies = {
        f"curated:bountiful:b{index}": f"Fact number {index} about produce."
        for index in range(5)
    }
    conn, _path = _built_core(tmp_path, bodies)
    ids = sorted(bodies)
    for belief_id in ids[:3]:
        _rewrite_body(conn, belief_id, f"{bodies[belief_id]} Revised.")

    _neighbors, unavailable, coverage = document_neighbors(
        conn, "Fact number 1 about produce. Revised.", candidate_ids=ids, embed_budget=2
    )
    assert unavailable is None
    assert coverage == {"candidates": 5, "reused": 2, "embedded": 2, "uncovered": 1}

    _neighbors, unavailable, coverage = document_neighbors(
        conn, "Fact number 1 about produce. Revised.", candidate_ids=ids, embed_budget=0
    )
    assert coverage == {"candidates": 5, "reused": 2, "embedded": 0, "uncovered": 3}


def test_the_default_embed_budget_is_large_enough_to_cover_a_cycles_writes(
    tmp_path: Path, local_embedder
) -> None:
    """The shipped default must actually embed. A default of 0 fails closed on everything.

    Called with no `embed_budget` argument on purpose: this is the only test
    that can see the value of `DEFAULT_DOCUMENT_EMBED_BUDGET`.
    """
    bodies = {
        f"curated:bountiful:c{index:02d}": f"Cycle fact {index:02d}." for index in range(24)
    }
    conn, _path = _built_core(tmp_path, bodies)
    ids = sorted(bodies)
    for belief_id in ids:
        _rewrite_body(conn, belief_id, f"{bodies[belief_id]} Revised.")

    _neighbors, unavailable, coverage = document_neighbors(
        conn, "Cycle fact 00. Revised.", candidate_ids=ids
    )
    assert unavailable is None
    assert coverage == {"candidates": 24, "reused": 0, "embedded": 24, "uncovered": 0}


def test_a_missing_sidecar_is_reported_with_coverage_not_silence(tmp_path: Path) -> None:
    path = tmp_path / "core.sqlite"
    conn = connect(path)
    init_core_v1(conn)
    _seed(conn, belief_id="curated:bountiful:citrus", body=CITRUS)
    conn.commit()
    neighbors, unavailable, coverage = document_neighbors(
        conn, CITRUS, candidate_ids=["curated:bountiful:citrus"]
    )
    assert neighbors == []
    assert unavailable == "vector_sidecar_missing"
    assert coverage == {"candidates": 1, "reused": 0, "embedded": 0, "uncovered": 0}
