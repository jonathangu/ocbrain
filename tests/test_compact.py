"""Compaction: collapsing historical near-duplicates without losing a fact.

The guards here are not decoration. A compactor that merges one belief too many
destroys knowledge that nothing in the ledger can reconstruct, so every rule that
stops a merge is pinned twice: once by asserting the merge does not happen, and
once by mutating the guard away and asserting the same case then *would* have
merged. A guard nobody has watched fail is a guard nobody knows is wired up.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocbrain.compact import (
    DEFAULT_COSINE_FLOOR,
    apply_compaction,
    build_adjudication_prompt,
    choose_survivor,
    cluster_guard,
    find_clusters,
    load_vectors,
    mechanical_verdict,
    plan_compaction,
    resolve_selection,
    serving_beliefs,
    undo_merge,
)
from ocbrain.core_v1 import append_core_event, get_core_v1_belief, init_core_v1
from ocbrain.db import connect
from ocbrain.mcp_v1 import decide_proposal_v1, get_v1
from ocbrain.scope import ScopeContext, ScopeTag
from ocbrain.vector import encode_embedding

PROJECT_SCOPE = ScopeTag(
    "project",
    "project:test",
    visibility="internal",
    egress_policy="hosted_ok",
    provenance="test",
)
OTHER_SCOPE = ScopeTag(
    "project",
    "project:other",
    visibility="internal",
    egress_policy="hosted_ok",
    provenance="test",
)
DOCTRINE_SCOPE = ScopeTag(
    "global",
    "global:doctrine",
    visibility="internal",
    egress_policy="hosted_ok",
    provenance="test",
)
CONTEXT = ScopeContext(project="test")


def _core(tmp_path: Path):
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    return conn


def _seed(
    conn,
    *,
    belief_id: str,
    body: str,
    scope: ScopeTag = PROJECT_SCOPE,
    confidence: float = 0.8,
    attributes: dict | None = None,
    evidence_ids: list[str] | None = None,
    pinned: bool = False,
) -> str:
    proposal = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": belief_id,
            "belief_type": "wiki_fact",
            "body": body,
            "evidence_ids": evidence_ids or [],
            "scope": scope.to_dict(),
            "confidence": confidence,
            "attributes": attributes or {},
        },
        writer="test",
    )
    decide_proposal_v1(
        conn,
        proposal_event_id=proposal,
        decision="approve",
        actor="test",
        edited_body=None,
        reason="seed",
    )
    if pinned:
        append_core_event(
            conn,
            "correction_recorded",
            {
                "subject": {"kind": "belief", "id": belief_id},
                "target_id": belief_id,
                "target_layer": "belief",
                "op": "pin",
                "author": "test",
                "hard": False,
            },
            writer="test",
            project=True,
        )
    conn.commit()
    return belief_id


def _sidecar(tmp_path: Path, vectors: dict[str, list[float]]) -> None:
    """Write a schema-valid vector sidecar beside the core."""
    import sqlite3

    path = tmp_path / "core-vectors.sqlite"
    side = sqlite3.connect(path)
    side.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE belief_vectors (
          belief_id TEXT PRIMARY KEY,
          content_hash TEXT,
          model TEXT,
          dimensions INTEGER,
          vector BLOB NOT NULL,
          scope_type TEXT,
          scope_id TEXT,
          visibility TEXT,
          egress_policy TEXT,
          last_compiled_at TEXT
        );
        """
    )
    side.execute(
        "INSERT INTO meta (key, value) VALUES ('schema_version', 'ocbrain.vectors.v2')"
    )
    for belief_id, vector in vectors.items():
        side.execute(
            "INSERT INTO belief_vectors (belief_id, vector, dimensions) VALUES (?, ?, ?)",
            (belief_id, encode_embedding(vector), len(vector)),
        )
    side.commit()
    side.close()


# Three unit vectors in the plane: NEAR_A and NEAR_B sit ~0.995 apart, FAR is
# orthogonal to both. Exact numbers keep the cosine arithmetic obvious.
NEAR_A = [1.0, 0.0]
NEAR_B = [0.99, 0.1]
FAR = [0.0, 1.0]


# --------------------------------------------------------------------------
# stage 1: candidates
# --------------------------------------------------------------------------


def test_clusters_never_cross_a_scope_boundary(tmp_path):
    """Two identical sentences in two projects are two facts, not one."""
    conn = _core(tmp_path)
    _seed(conn, belief_id="belief_here", body="the gateway runs on port 9140")
    _seed(
        conn,
        belief_id="belief_there",
        body="the gateway runs on port 9140",
        scope=OTHER_SCOPE,
    )
    beliefs = serving_beliefs(conn)
    vectors = {"belief_here": NEAR_A, "belief_there": NEAR_A}

    assert find_clusters(beliefs, vectors) == []

    # Mutation: same corpus, same vectors, scope collapsed to one value. The
    # cluster appears, which proves the scope split is what suppressed it.
    for belief in beliefs.values():
        belief["scope_id"] = "project:test"
    assert find_clusters(beliefs, vectors) == [["belief_here", "belief_there"]]


def test_below_the_floor_is_not_a_candidate(tmp_path):
    conn = _core(tmp_path)
    _seed(conn, belief_id="belief_a", body="one thing")
    _seed(conn, belief_id="belief_b", body="another thing")
    beliefs = serving_beliefs(conn)
    vectors = {"belief_a": NEAR_A, "belief_b": FAR}
    assert find_clusters(beliefs, vectors, cosine_floor=DEFAULT_COSINE_FLOOR) == []
    assert find_clusters(beliefs, vectors, cosine_floor=0.0) == [
        ["belief_a", "belief_b"]
    ]


def test_union_find_chains_transitively(tmp_path):
    """A~B and B~C cluster A with C even where A~C is below the floor.

    Pinned deliberately: it is how one fact reworded four times is caught, and
    it is also why a cluster is not on its own evidence of anything. The plan
    reports the weakest internal pair for exactly this reason.
    """
    conn = _core(tmp_path)
    for index, body in enumerate(("first", "second", "third")):
        _seed(conn, belief_id=f"belief_{index}", body=body)
    beliefs = serving_beliefs(conn)
    vectors = {
        "belief_0": [1.0, 0.0],
        "belief_1": [0.9, 0.436],
        "belief_2": [0.66, 0.75],
    }
    clusters = find_clusters(beliefs, vectors, cosine_floor=0.88)
    assert clusters == [["belief_0", "belief_1", "belief_2"]]
    # The end pair is genuinely below the floor; only chaining joined them.
    end_pair = zip(vectors["belief_0"], vectors["belief_2"], strict=True)
    assert sum(a * b for a, b in end_pair) < 0.88


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------


def test_a_pinned_belief_is_never_merged(tmp_path):
    conn = _core(tmp_path)
    _seed(conn, belief_id="belief_plain", body="the fleet has five profiles")
    _seed(
        conn,
        belief_id="belief_pinned",
        body="the fleet runs five profiles",
        pinned=True,
    )
    beliefs = serving_beliefs(conn)
    members = ["belief_pinned", "belief_plain"]
    assert beliefs["belief_pinned"]["pinned"] is True

    blocked = cluster_guard(members, beliefs)
    assert blocked is not None and "pinned" in blocked

    # Mutation: clear only the pin. The same cluster now passes, so the pin was
    # the whole reason it did not.
    beliefs["belief_pinned"]["pinned"] = False
    assert cluster_guard(members, beliefs) is None


def test_a_pin_set_only_as_an_attribute_still_guards(tmp_path):
    conn = _core(tmp_path)
    _seed(conn, belief_id="belief_a", body="a fact")
    _seed(
        conn,
        belief_id="belief_b",
        body="a fact restated",
        attributes={"pinned": True},
    )
    beliefs = serving_beliefs(conn)
    assert beliefs["belief_b"]["pinned"] is True
    assert cluster_guard(["belief_a", "belief_b"], beliefs) is not None


def test_a_contradicts_marked_pair_is_never_merged(tmp_path):
    """Somebody already decided these two coexist. That outranks similarity."""
    conn = _core(tmp_path)
    _seed(conn, belief_id="belief_left", body="staging uses the new path")
    _seed(
        conn,
        belief_id="belief_right",
        body="production uses the old path",
        attributes={"contradicts": ["belief_left"]},
    )
    beliefs = serving_beliefs(conn)
    members = ["belief_left", "belief_right"]

    blocked = cluster_guard(members, beliefs)
    assert blocked is not None and "contradicting" in blocked

    # Mutation: drop the annotation only.
    beliefs["belief_right"]["contradicts"] = []
    assert cluster_guard(members, beliefs) is None


def test_a_contradicts_pointing_outside_the_cluster_does_not_guard(tmp_path):
    conn = _core(tmp_path)
    _seed(conn, belief_id="belief_left", body="a fact")
    _seed(
        conn,
        belief_id="belief_right",
        body="a fact restated",
        attributes={"contradicts": ["belief_elsewhere"]},
    )
    beliefs = serving_beliefs(conn)
    assert cluster_guard(["belief_left", "belief_right"], beliefs) is None


def test_doctrine_is_never_compacted(tmp_path):
    conn = _core(tmp_path)
    _seed(conn, belief_id="belief_one", body="doctrine says a thing", scope=DOCTRINE_SCOPE)
    _seed(conn, belief_id="belief_two", body="doctrine states a thing", scope=DOCTRINE_SCOPE)
    beliefs = serving_beliefs(conn)
    members = ["belief_one", "belief_two"]

    assert cluster_guard(members, beliefs) == "doctrine is never compacted unattended"

    # Mutation: same bodies, same similarity, project scope instead.
    for belief in beliefs.values():
        belief["scope_type"] = "project"
        belief["scope_id"] = "project:test"
    assert cluster_guard(members, beliefs) is None


def test_an_oversized_cluster_is_not_adjudicated(tmp_path):
    conn = _core(tmp_path)
    members = []
    for index in range(9):
        members.append(_seed(conn, belief_id=f"belief_{index}", body=f"body {index}"))
    beliefs = serving_beliefs(conn)
    blocked = cluster_guard(sorted(members), beliefs)
    assert blocked is not None and "review cap" in blocked


# --------------------------------------------------------------------------
# stage 2: mechanical
# --------------------------------------------------------------------------


def test_identical_bodies_resolve_with_no_model_call(tmp_path):
    conn = _core(tmp_path)
    body = "the watchdog runs hourly and pauses on a sentinel file"
    _seed(conn, belief_id="belief_a", body=body)
    _seed(conn, belief_id="belief_b", body=body, confidence=0.7)
    _sidecar(tmp_path, {"belief_a": NEAR_A, "belief_b": NEAR_A})

    def explode(members, beliefs):  # pragma: no cover - must never run
        raise AssertionError("identical bodies must not reach a model")

    plan = plan_compaction(conn, adjudicator=explode)
    assert plan["stages"]["resolved_identical_bodies"] == 1
    assert plan["stages"]["escalated"] == 0
    assert plan["hosted_calls"] == 0
    assert len(plan["merges"]) == 1
    assert plan["merges"][0]["stage"] == "identical_bodies"
    # Highest confidence survives.
    assert plan["merges"][0]["survivor"] == "belief_a"


def test_a_restatement_cluster_resolves_mechanically():
    beliefs = {
        "belief_a": {"body": "the fleet upgraded the shared runtime to version four"},
        "belief_b": {"body": "the shared runtime the fleet uses upgraded to version four"},
    }
    verdict = mechanical_verdict(["belief_a", "belief_b"], beliefs)
    assert verdict is not None and verdict[0] == "restatement"


def test_a_low_overlap_cluster_does_not_resolve_mechanically():
    beliefs = {
        "belief_a": {"body": "use the mac-sleep command for intentional sleep"},
        "belief_b": {"body": "the data lake root is a directory with a size budget"},
    }
    assert mechanical_verdict(["belief_a", "belief_b"], beliefs) is None


def test_every_pair_must_clear_the_restatement_threshold():
    """A chained cluster has pairs nothing ever compared; all of them count."""
    beliefs = {
        "belief_a": {"body": "the runtime upgraded to version four today"},
        "belief_b": {"body": "the runtime upgraded to version four"},
        "belief_c": {"body": "a completely unrelated statement about storage"},
    }
    assert mechanical_verdict(["belief_a", "belief_b"], beliefs) is not None
    assert mechanical_verdict(["belief_a", "belief_b", "belief_c"], beliefs) is None


def test_choose_survivor_prefers_confidence_then_evidence():
    beliefs = {
        "belief_a": {"confidence": 0.7, "evidence_ids": ["e1", "e2"], "body": "aaa"},
        "belief_b": {"confidence": 0.9, "evidence_ids": [], "body": "b"},
        "belief_c": {"confidence": 0.9, "evidence_ids": ["e1"], "body": "c"},
    }
    assert choose_survivor(["belief_a", "belief_b", "belief_c"], beliefs) == "belief_c"


# --------------------------------------------------------------------------
# stage 3: adjudication
# --------------------------------------------------------------------------


def test_an_invented_index_produces_no_action():
    members = ["belief_a", "belief_b"]
    survivor, losers, reason = resolve_selection(
        {"keep": 0, "merge": [7], "reason": "made it up"}, members
    )
    assert survivor is None
    assert losers == []
    assert "no valid merge index" in reason


def test_an_out_of_range_keep_produces_no_action():
    survivor, _, reason = resolve_selection(
        {"keep": 9, "merge": [0], "reason": "nope"}, ["belief_a", "belief_b"]
    )
    assert survivor is None
    assert "outside the supplied list" in reason


def test_a_partly_invented_merge_keeps_only_the_valid_indices():
    members = ["belief_a", "belief_b", "belief_c"]
    survivor, losers, _ = resolve_selection(
        {"keep": 0, "merge": [1, 42, -1], "reason": "one real index"}, members
    )
    assert survivor == "belief_a"
    assert losers == ["belief_b"]


def test_merge_may_be_a_strict_subset_of_the_cluster():
    """The split that saves a real fact from a chained cluster."""
    members = ["belief_a", "belief_b", "belief_c"]
    survivor, losers, _ = resolve_selection(
        {"keep": 0, "merge": [2], "reason": "1 carries an extra fact"}, members
    )
    assert survivor == "belief_a"
    assert losers == ["belief_c"]
    assert "belief_b" not in losers


def test_coexist_produces_no_merge():
    survivor, losers, reason = resolve_selection(
        {"coexist": True, "reason": "different subjects"}, ["belief_a", "belief_b"]
    )
    assert survivor is None
    assert losers == []
    assert reason == "different subjects"


def test_a_boolean_is_not_an_index():
    survivor, _, _ = resolve_selection(
        {"keep": True, "merge": [1], "reason": "x"}, ["belief_a", "belief_b"]
    )
    assert survivor is None


def test_the_prompt_numbers_every_body_and_names_the_valid_range():
    beliefs = {
        "belief_a": {"body": "first statement"},
        "belief_b": {"body": "second statement"},
    }
    prompt = build_adjudication_prompt(["belief_a", "belief_b"], beliefs)
    assert "[0] first statement" in prompt
    assert "[1] second statement" in prompt
    assert "0 through 1" in prompt
    # Ids never reach the model; it selects positions, not names.
    assert "belief_a" not in prompt


def test_an_adjudicated_merge_records_an_egress_audit(tmp_path):
    conn = _core(tmp_path)
    _seed(conn, belief_id="belief_a", body="the first phrasing of a durable fact")
    _seed(conn, belief_id="belief_b", body="a second and quite different phrasing")
    _sidecar(tmp_path, {"belief_a": NEAR_A, "belief_b": NEAR_B})

    from ocbrain.compact import record_compaction_egress

    beliefs = serving_beliefs(conn)
    audit_id = record_compaction_egress(
        conn,
        members=["belief_a", "belief_b"],
        beliefs=beliefs,
        provider="anthropic",
        model="claude-sonnet-5",
        egress_policies=("hosted_ok",),
    )
    row = conn.execute(
        "SELECT target, context_json, included_json FROM egress_audits WHERE id=?",
        (audit_id,),
    ).fetchone()
    assert row["target"] == "anthropic:claude-sonnet-5"
    assert json.loads(row["context_json"])["purpose"] == "belief_compaction"
    assert {item["belief_id"] for item in json.loads(row["included_json"])} == {
        "belief_a",
        "belief_b",
    }


# --------------------------------------------------------------------------
# planning and applying
# --------------------------------------------------------------------------


def test_an_absent_sidecar_degrades_to_not_measured(tmp_path):
    conn = _core(tmp_path)
    _seed(conn, belief_id="belief_a", body="a fact")
    _seed(conn, belief_id="belief_b", body="a fact")
    assert load_vectors(conn, ["belief_a"]) is None
    plan = plan_compaction(conn)
    assert plan["measured"] is False
    assert plan["status"] == "not_measured"
    assert plan["merges"] == []
    assert plan["would_retire"] == 0
    # Still reports what it could see, so the run is informative, not silent.
    assert plan["serving"] == 2


def test_both_plan_branches_have_the_same_envelope(tmp_path):
    """A caller reading a plan never has to ask which branch produced it."""
    measured_dir = tmp_path / "measured"
    measured_dir.mkdir()
    conn = _core(measured_dir)
    _seed(conn, belief_id="belief_a", body="a repeated statement")
    _seed(conn, belief_id="belief_b", body="a repeated statement")
    _sidecar(measured_dir, {"belief_a": NEAR_A, "belief_b": NEAR_A})
    measured = plan_compaction(conn)
    assert measured["measured"] is True

    bare_dir = tmp_path / "bare"
    bare_dir.mkdir()
    bare = plan_compaction(_core(bare_dir))
    assert bare["measured"] is False

    assert set(measured) - set(bare) == set()
    assert set(bare) - set(measured) == {"detail"}


def test_a_stale_sidecar_schema_degrades_to_not_measured(tmp_path):
    import sqlite3

    conn = _core(tmp_path)
    _seed(conn, belief_id="belief_a", body="a fact")
    side = sqlite3.connect(tmp_path / "core-vectors.sqlite")
    side.executescript(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE belief_vectors (belief_id TEXT PRIMARY KEY, vector BLOB);"
    )
    side.execute("INSERT INTO meta VALUES ('schema_version', 'ocbrain.vectors.v1')")
    side.commit()
    side.close()
    assert plan_compaction(conn)["status"] == "not_measured"


def test_the_limit_defers_rather_than_drops(tmp_path):
    conn = _core(tmp_path)
    shared = "the scheduled job runs every hour and writes a receipt"
    for index in range(3):
        _seed(conn, belief_id=f"belief_a{index}", body=shared, confidence=0.9 - index / 100)
    other = "the nightly sweep retires anything past its expiry date"
    for index in range(3):
        _seed(conn, belief_id=f"belief_b{index}", body=other, confidence=0.7 - index / 100)
    _sidecar(
        tmp_path,
        {
            "belief_a0": NEAR_A, "belief_a1": NEAR_A, "belief_a2": NEAR_A,
            "belief_b0": FAR, "belief_b1": FAR, "belief_b2": FAR,
        },
    )
    plan = plan_compaction(conn, limit=2)
    assert len(plan["merges"]) == 1
    assert len(plan["deferred"]) == 1
    assert plan["would_retire"] == 2
    assert "cap of 2" in plan["deferred"][0]["deferred_reason"]
    # Nothing was lost: every cluster is still accounted for somewhere.
    assert plan["stages"]["clusters_found"] == 2


def test_the_cap_keeps_the_safest_merges(tmp_path):
    """A run cut short by the cap keeps what needed a human least."""
    conn = _core(tmp_path)
    identical = "the exporter writes parquet and never pandas"
    _seed(conn, belief_id="belief_x0", body=identical)
    _seed(conn, belief_id="belief_x1", body=identical)
    _seed(conn, belief_id="belief_y0", body="a statement about one subject entirely")
    _seed(conn, belief_id="belief_y1", body="a different sentence covering other ground")
    _sidecar(
        tmp_path,
        {"belief_x0": NEAR_A, "belief_x1": NEAR_A, "belief_y0": FAR, "belief_y1": FAR},
    )

    def adjudicator(members, beliefs):
        return {"survivor": members[0], "losers": members[1:], "reason": "model said so"}

    plan = plan_compaction(conn, limit=1, adjudicator=adjudicator)
    assert len(plan["merges"]) == 1
    assert plan["merges"][0]["stage"] == "identical_bodies"
    assert plan["deferred"][0]["stage"] == "adjudicated"


def test_no_adjudicator_leaves_the_tail_undecided(tmp_path):
    conn = _core(tmp_path)
    _seed(conn, belief_id="belief_a", body="the first phrasing of a durable fact")
    _seed(conn, belief_id="belief_b", body="a second and quite different phrasing")
    _sidecar(tmp_path, {"belief_a": NEAR_A, "belief_b": NEAR_B})
    plan = plan_compaction(conn, adjudicator=None)
    assert plan["stages"]["escalated"] == 1
    assert plan["stages"]["not_adjudicated"] == 1
    assert plan["merges"] == []
    assert plan["hosted_calls"] == 0


def test_a_body_that_may_not_leave_is_never_sent(tmp_path):
    conn = _core(tmp_path)
    local = ScopeTag(
        "project",
        "project:test",
        visibility="internal",
        egress_policy="local_only",
        provenance="test",
    )
    _seed(conn, belief_id="belief_a", body="the first phrasing of a durable fact", scope=local)
    _seed(conn, belief_id="belief_b", body="a second and quite different phrasing", scope=local)
    _sidecar(tmp_path, {"belief_a": NEAR_A, "belief_b": NEAR_B})

    def explode(members, beliefs):  # pragma: no cover - must never run
        raise AssertionError("a local_only body must not be sent")

    plan = plan_compaction(conn, adjudicator=explode, egress_policies=("hosted_ok",))
    assert plan["stages"]["withheld_egress"] == 1
    assert plan["merges"] == []
    assert plan["hosted_calls"] == 0


def test_apply_supersedes_the_loser_behind_the_survivor(tmp_path):
    conn = _core(tmp_path)
    body = "the publisher writes through the experiment api, never raw mongo"
    _seed(conn, belief_id="belief_keep", body=body, confidence=0.9)
    _seed(conn, belief_id="belief_drop", body=body, confidence=0.7)
    _sidecar(tmp_path, {"belief_keep": NEAR_A, "belief_drop": NEAR_A})

    plan = plan_compaction(conn)
    result = apply_compaction(conn, plan)
    assert result["retired"] == 1
    assert result["applied"] == [
        {"belief_id": "belief_drop", "superseded_by": "belief_keep"}
    ]

    survivor = get_core_v1_belief(conn, "belief_keep")
    # A merge is not a new claim: the survivor keeps its id, body and confidence.
    assert survivor["status"] == "current"
    assert bool(survivor["serve"]) is True
    assert survivor["confidence"] == 0.9
    assert survivor["body"] == body

    loser = get_core_v1_belief(conn, "belief_drop")
    assert loser["status"] == "retracted"
    assert loser["attributes"]["superseded_by"] == "belief_keep"
    # The validity window is closed rather than erased.
    assert loser["attributes"]["valid_until"]


def test_a_merged_loser_is_still_reachable_as_stored(tmp_path):
    conn = _core(tmp_path)
    _seed(conn, belief_id="belief_keep", body="one statement about a thing", confidence=0.9)
    _seed(conn, belief_id="belief_drop", body="one statement about a thing", confidence=0.7)
    _sidecar(tmp_path, {"belief_keep": NEAR_A, "belief_drop": NEAR_A})
    apply_compaction(conn, plan_compaction(conn))

    stored = get_v1(conn, "belief_drop", context=CONTEXT, mode="as_stored")
    assert stored["body"] == "one statement about a thing"
    assert stored["invalidated"] is True
    assert stored["superseded_by"] == "belief_keep"

    # And the pointer resolves forward for a caller holding the retired id.
    resolved = get_v1(conn, "belief_drop", context=CONTEXT, mode="resolve")
    assert resolved["resolved_from"] == ["belief_drop"]
    assert resolved["canonical_id"] == "belief_keep"


def test_apply_writes_nothing_for_an_empty_plan(tmp_path):
    conn = _core(tmp_path)
    _seed(conn, belief_id="belief_a", body="a fact")
    before = conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0]
    result = apply_compaction(conn, {"merges": []})
    assert result["retired"] == 0
    assert conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0] == before


def test_undo_puts_a_merged_belief_back(tmp_path):
    """The one command the report promises actually works.

    `hygiene --restore` alone cannot do this: a serving successor blocks the
    restore by design. The undo clears the pointer first, and clears
    `valid_until` with it so the expiry sweep does not re-retire the belief on
    its next pass.
    """
    conn = _core(tmp_path)
    _seed(conn, belief_id="belief_keep", body="a statement worth keeping once", confidence=0.9)
    _seed(conn, belief_id="belief_drop", body="a statement worth keeping once", confidence=0.7)
    _sidecar(tmp_path, {"belief_keep": NEAR_A, "belief_drop": NEAR_A})
    apply_compaction(conn, plan_compaction(conn))

    from ocbrain.hygiene import plan_retirements, restore

    with pytest.raises(PermissionError):
        restore(conn, belief_id="belief_drop")

    result = undo_merge(conn, belief_id="belief_drop")
    assert result["was_superseded_by"] == "belief_keep"
    assert result["changed"] is True

    revived = get_core_v1_belief(conn, "belief_drop")
    assert revived["status"] == "current"
    assert bool(revived["serve"]) is True
    assert revived["body"] == "a statement worth keeping once"
    # And the next hygiene sweep leaves it alone rather than re-retiring it.
    targets = plan_retirements(conn, classes=("expired",))["targets"]
    assert "belief_drop" not in [target["belief_id"] for target in targets]


def test_a_retired_belief_is_not_a_compaction_candidate(tmp_path):
    conn = _core(tmp_path)
    _seed(conn, belief_id="belief_keep", body="a repeated statement", confidence=0.9)
    _seed(conn, belief_id="belief_drop", body="a repeated statement", confidence=0.7)
    _sidecar(tmp_path, {"belief_keep": NEAR_A, "belief_drop": NEAR_A})
    apply_compaction(conn, plan_compaction(conn))
    # Second pass sees one serving belief and nothing to do.
    again = plan_compaction(conn)
    assert again["serving"] == 1
    assert again["merges"] == []


# --------------------------------------------------------------------------
# CLI acceptance
# --------------------------------------------------------------------------


def _cli_core(tmp_path, monkeypatch):
    """A core with one obvious duplicate pair, and no hosted provider reachable.

    The env scrub is not hygiene, it is the test's subject: `cmd_compact` builds
    its adjudicator out of the ambient environment, so a developer running the
    suite with a real key set would otherwise bill live calls from a unit test.
    """
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "KIMI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    conn = _core(tmp_path)
    body = "the nightly job publishes a receipt and then exits"
    _seed(conn, belief_id="belief_keep", body=body, confidence=0.9)
    _seed(conn, belief_id="belief_drop", body=body, confidence=0.7)
    conn.close()
    _sidecar(tmp_path, {"belief_keep": NEAR_A, "belief_drop": NEAR_A})
    return tmp_path / "core.sqlite"


def test_cli_dry_run_is_the_default_and_writes_nothing(tmp_path, capsys, monkeypatch):
    from ocbrain.cli import main

    db = _cli_core(tmp_path, monkeypatch)
    assert main(["--db", str(db), "compact"]) == 0
    report = capsys.readouterr().out
    assert "DRY RUN. Nothing was written." in report
    # The side-by-side is the report's whole purpose: both bodies, both ids.
    assert "belief_keep" in report and "belief_drop" in report
    assert "the nightly job publishes a receipt" in report
    assert "ocbrain compact --apply --yes" in report

    conn = connect(db)
    assert get_core_v1_belief(conn, "belief_drop")["status"] == "current"


def test_cli_local_compaction_supports_active_local_only_policy(
    tmp_path, capsys, monkeypatch
):
    from ocbrain.cli import main

    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "KIMI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    conn = _core(tmp_path)
    local_scope = ScopeTag(
        "project",
        "project:test",
        visibility="internal",
        egress_policy="local_only",
        provenance="test",
    )
    body = "the nightly job publishes a receipt and then exits"
    _seed(conn, belief_id="belief_keep", body=body, confidence=0.9, scope=local_scope)
    _seed(conn, belief_id="belief_drop", body=body, confidence=0.7, scope=local_scope)
    conn.close()
    _sidecar(tmp_path, {"belief_keep": NEAR_A, "belief_drop": NEAR_A})
    db = tmp_path / "core.sqlite"
    config_path = tmp_path / "ocbrain.config.json"
    config_path.write_text(
        json.dumps({"curator": {"egress_policies": ["local_only"]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("OCBRAIN_CONFIG", str(config_path))

    assert main(["--db", str(db), "compact"]) == 0
    report = capsys.readouterr().out
    assert "DRY RUN. Nothing was written." in report
    assert "belief_keep" in report and "belief_drop" in report


def test_hosted_compaction_rejects_local_only_selection_policy(tmp_path):
    conn = _core(tmp_path)
    _seed(
        conn,
        belief_id="belief_a",
        body="the first phrasing of a durable fact",
        scope=PROJECT_SCOPE,
    )
    _seed(
        conn,
        belief_id="belief_b",
        body="a second and quite different phrasing",
        scope=PROJECT_SCOPE,
    )
    _sidecar(tmp_path, {"belief_a": NEAR_A, "belief_b": NEAR_B})

    def explode(members, beliefs):  # pragma: no cover - must never run
        raise AssertionError("local_only policy reached a hosted adjudicator")

    with pytest.raises(ValueError, match="local_only.*hosted curator"):
        plan_compaction(
            conn,
            adjudicator=explode,
            egress_policies=(policy for policy in ("hosted_ok", "local_only")),
        )


def test_cli_requires_explicit_authority_before_building_a_hosted_adjudicator(
    tmp_path, monkeypatch
):
    from ocbrain.cli import _compaction_adjudicator, build_parser

    monkeypatch.setenv("ANTHROPIC_API_KEY", "unit-test-placeholder")
    db = tmp_path / "core.sqlite"
    conn = _core(tmp_path)
    try:
        args = build_parser().parse_args(["--db", str(db), "compact"])
        assert _compaction_adjudicator(conn, args, ("hosted_ok",)) is None

        allowed = build_parser().parse_args(
            ["--db", str(db), "compact", "--allow-hosted-egress"]
        )
        assert callable(_compaction_adjudicator(conn, allowed, ("hosted_ok",)))
    finally:
        conn.close()


def test_dry_run_report_discloses_hosted_egress_audit_writes(tmp_path):
    from ocbrain.cli import render_compaction

    conn = _core(tmp_path)
    body = "the nightly job publishes a receipt and then exits"
    _seed(conn, belief_id="belief_keep", body=body, confidence=0.9)
    _seed(conn, belief_id="belief_drop", body=body, confidence=0.7)
    _sidecar(tmp_path, {"belief_keep": NEAR_A, "belief_drop": NEAR_A})
    plan = plan_compaction(conn)
    plan["hosted_calls"] = 1

    report = render_compaction(plan, applied=None)
    assert "DRY RUN. No beliefs were changed." in report
    assert "recorded 1 egress audit(s)" in report
    assert "Nothing was written" not in report


def test_cli_apply_is_refused_without_yes(tmp_path, capsys, monkeypatch):
    from ocbrain.cli import main

    db = _cli_core(tmp_path, monkeypatch)
    assert main(["--db", str(db), "compact", "--apply"]) == 0
    assert "refusing to apply: --apply also requires --yes" in capsys.readouterr().out

    conn = connect(db)
    assert get_core_v1_belief(conn, "belief_drop")["status"] == "current"


def test_cli_apply_with_yes_merges_takes_a_snapshot_and_prints_the_undo(
    tmp_path, capsys, monkeypatch
):
    from ocbrain.cli import main

    db = _cli_core(tmp_path, monkeypatch)
    assert main(["--db", str(db), "compact", "--apply", "--yes"]) == 0
    report = capsys.readouterr().out
    assert "APPLIED. 1 beliefs retired" in report
    assert "ocbrain compact --undo belief_drop" in report
    # The snapshot precondition is satisfied by taking one, and it is named.
    assert "snapshot covering this run:" in report
    assert list((tmp_path / "backups").glob("pre-compact-*.sqlite"))

    conn = connect(db)
    assert get_core_v1_belief(conn, "belief_drop")["status"] == "retracted"

    # And the printed undo command is real.
    assert main(["--db", str(db), "compact", "--undo", "belief_drop"]) == 0
    conn = connect(db)
    assert get_core_v1_belief(conn, "belief_drop")["status"] == "current"


def test_cli_json_mode_emits_the_plan(tmp_path, capsys, monkeypatch):
    from ocbrain.cli import main

    db = _cli_core(tmp_path, monkeypatch)
    assert main(["--db", str(db), "compact", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "compact"
    assert payload["applied"] is None
    assert payload["plan"]["stages"]["resolved_identical_bodies"] == 1
