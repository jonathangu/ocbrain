"""What feedback is allowed to say, and how long a belief's record survives.

Every corpus figure below is from one snapshot frozen at 2026-08-28T19:28:58Z.
The corpus is live and moved four times over the hour these were written, so an
undated count here would read as a claim that no longer reproduces.

Two defects, one column. ``retrieval_uses.outcome`` carries both "the corpus had
nothing for this query" and "the corpus served the wrong thing", and feedback is
the only ranking signal the brain has. In the snapshot, 1,086 of 2,048
retrievals served zero items and 183 of those carry a relevance verdict anyway,
174 of them ``irrelevant`` -- filed against a written instruction not to file
them. An instruction that 183 rows ignore is not a rule, so the server enforces
it and records the zero-item case itself.

The second defect runs the other way: every curator pass mints a new belief_id,
so the retrieval history stayed behind on an id nothing serves. 390 of 587
ever-retrieved ids are retracted. These tests pin the successor inheriting its
ancestors' record, once, across a three-generation chain.

A third group covers what the two cores are each *told*. The refusal lives in
the v1 path, and the text asserting it was served to every connection; these pin
each string to the core that enforces it.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ocbrain.cli import main as cli_main
from ocbrain.core_v1 import (
    NO_COVERAGE_OUTCOME,
    RELEVANCE_OUTCOMES,
    SERVED_OUTCOME,
    _retrieval_feedback_scores,
    append_core_event,
    init_core_v1,
    reclassify_no_coverage_receipts,
    record_core_v1_evidence,
    record_core_v1_retrieval,
    retrieval_history_by_lineage,
)
from ocbrain.db import connect, init_db, log_retrieval_use
from ocbrain.mcp import call_tool, handle_request, instructions_text, tool_list
from ocbrain.mcp_v1 import MAX_RESOLUTION_HOPS, feedback_v1, supersede_v1
from ocbrain.scope import ScopeContext, ScopeTag

SCOPE = ScopeTag(
    "project",
    "project:bountiful",
    visibility="internal",
    egress_policy="local_only",
    provenance="test",
)
CONTEXT = ScopeContext(project="bountiful")


def _core(tmp_path: Path):
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    return conn


def _seed(conn, *, belief_id: str, body: str) -> str:
    evidence_id, _event = record_core_v1_evidence(
        conn,
        body=f"evidence for {belief_id}",
        kind="observation",
        scope=SCOPE,
        writer="test",
    )
    proposal = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": belief_id,
            "belief_type": "wiki_fact",
            "body": body,
            "evidence_ids": [evidence_id],
            "scope": SCOPE.to_dict(),
            "confidence": 0.9,
            "attributes": {},
        },
        writer="test",
    )
    append_core_event(
        conn,
        "compilation_decided",
        {"proposal_event_id": proposal, "decision": "approve", "actor": "test"},
        writer="test",
        project=True,
    )
    conn.commit()
    return belief_id


def _serve(conn, *belief_ids: str, query: str) -> str:
    """Record one retrieval receipt naming these beliefs, or none of them."""
    retrieval_id = record_core_v1_retrieval(
        conn,
        query=query,
        context=CONTEXT.to_dict(),
        items=[{"belief_id": belief_id, "score": 1.0} for belief_id in belief_ids],
        runtime="test",
        task_ref="test-task",
        session_id="session-1",
    )
    conn.commit()
    return retrieval_id


def _outcome(conn, retrieval_id: str) -> str:
    return str(
        conn.execute(
            "SELECT outcome FROM retrieval_uses WHERE id=?", (retrieval_id,)
        ).fetchone()[0]
    )


# --------------------------------------------------------------------------- #
# Defect 1 -- an empty retrieval is not a bad retrieval
# --------------------------------------------------------------------------- #
def test_the_server_records_a_zero_item_read_as_no_coverage(tmp_path: Path) -> None:
    """The item count is observed where the row is written, not reported later."""
    conn = _core(tmp_path)
    belief = _seed(conn, belief_id="belief:vm", body="The research VM is reached with ssh asa2.")

    served = _serve(conn, belief, query="how do I reach the research vm")
    empty = _serve(conn, query="what is the pager rotation")

    assert _outcome(conn, served) == SERVED_OUTCOME
    assert _outcome(conn, empty) == NO_COVERAGE_OUTCOME


def test_a_relevance_verdict_on_a_zero_item_retrieval_is_refused(tmp_path: Path) -> None:
    conn = _core(tmp_path)
    empty = _serve(conn, query="what is the pager rotation")

    with pytest.raises(ValueError) as raised:
        feedback_v1(conn, empty, outcome="irrelevant", note="nothing came back")

    message = str(raised.value)
    assert "served zero items" in message
    # The error has to say what to do instead, or the caller files it anyway.
    assert "brain.ingest" in message
    assert NO_COVERAGE_OUTCOME in message
    # And the refusal leaves the receipt exactly as the server wrote it.
    assert _outcome(conn, empty) == NO_COVERAGE_OUTCOME
    row = conn.execute(
        "SELECT note, feedback_at FROM retrieval_uses WHERE id=?", (empty,)
    ).fetchone()
    assert row["note"] is None
    assert row["feedback_at"] is None


def test_no_coverage_cannot_be_filed_by_the_caller(tmp_path: Path) -> None:
    """Server-derived, because a caller-supplied count can disagree with the row."""
    conn = _core(tmp_path)
    belief = _seed(conn, belief_id="belief:vm", body="The research VM is reached with ssh asa2.")
    served = _serve(conn, belief, query="how do I reach the research vm")

    for claimed in (NO_COVERAGE_OUTCOME, SERVED_OUTCOME):
        with pytest.raises(ValueError) as raised:
            feedback_v1(conn, served, outcome=claimed, note=None)
        assert "recorded by the server" in str(raised.value)

    assert _outcome(conn, served) == SERVED_OUTCOME


def test_feedback_on_a_served_retrieval_still_records_the_verdict(tmp_path: Path) -> None:
    conn = _core(tmp_path)
    belief = _seed(conn, belief_id="belief:vm", body="The research VM is reached with ssh asa2.")
    served = _serve(conn, belief, query="how do I reach the research vm")

    result = feedback_v1(conn, served, outcome="used", note="used the host name")

    assert result == {"retrieval_use_id": served, "outcome": "used", "served_items": 1}
    assert _outcome(conn, served) == "used"


def test_an_unknown_retrieval_id_is_still_a_not_found_error(tmp_path: Path) -> None:
    conn = _core(tmp_path)
    with pytest.raises(ValueError, match="retrieval use not found"):
        feedback_v1(conn, "ret:nope", outcome="used", note=None)


# --------------------------------------------------------------------------- #
# Defect 1 -- the rows already written
# --------------------------------------------------------------------------- #
def _force_verdict(conn, retrieval_id: str, outcome: str) -> None:
    """Write a verdict the way the old server would have: no zero-item check."""
    conn.execute(
        "UPDATE retrieval_uses SET outcome=?, feedback_source='runtime_explicit' WHERE id=?",
        (outcome, retrieval_id),
    )
    conn.commit()


def test_reclassification_reports_by_default_and_spares_judged_packets(tmp_path: Path) -> None:
    conn = _core(tmp_path)
    belief = _seed(conn, belief_id="belief:vm", body="The research VM is reached with ssh asa2.")
    empty_one = _serve(conn, query="what is the pager rotation")
    empty_two = _serve(conn, query="who owns the staging cluster")
    judged = _serve(conn, belief, query="how do I reach the research vm")
    _force_verdict(conn, empty_one, "irrelevant")
    _force_verdict(conn, empty_two, "ignored")
    _force_verdict(conn, judged, "irrelevant")

    plan = reclassify_no_coverage_receipts(conn, apply=False)
    assert plan["candidates"] == 2
    assert plan["by_outcome"] == {"ignored": 1, "irrelevant": 1}
    assert plan["applied"] == 0
    assert plan["dry_run"] is True
    # A report writes nothing.
    assert _outcome(conn, empty_one) == "irrelevant"

    applied = reclassify_no_coverage_receipts(conn, apply=True)
    conn.commit()
    assert applied["applied"] == 2
    assert _outcome(conn, empty_one) == NO_COVERAGE_OUTCOME
    assert _outcome(conn, empty_two) == NO_COVERAGE_OUTCOME
    # The verdict on the packet that actually served an item is untouched: this
    # command repairs a category error, it does not launder bad reviews.
    assert _outcome(conn, judged) == "irrelevant"
    note = conn.execute(
        "SELECT note FROM retrieval_uses WHERE id=?", (empty_one,)
    ).fetchone()[0]
    assert "reclassified from irrelevant" in str(note)

    # Idempotent: nothing is left to reclassify.
    assert reclassify_no_coverage_receipts(conn, apply=False)["candidates"] == 0


def test_cli_feedback_repair_reports_then_applies(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "core.sqlite"
    conn = connect(db_path)
    init_core_v1(conn)
    empty = _serve(conn, query="what is the pager rotation")
    _force_verdict(conn, empty, "irrelevant")
    conn.close()

    assert cli_main(["--db", str(db_path), "feedback-repair"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["action"] == "feedback-repair"
    assert report["candidates"] == 1
    assert report["applied"] == 0

    assert cli_main(["--db", str(db_path), "feedback-repair", "--apply"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["applied"] == 1
    assert applied["dry_run"] is False

    conn = connect(db_path)
    assert _outcome(conn, empty) == NO_COVERAGE_OUTCOME
    conn.close()


# --------------------------------------------------------------------------- #
# Defect 2 -- history survives the recompile that renames the belief
# --------------------------------------------------------------------------- #
def _chain(conn) -> tuple[str, str, str]:
    """Three generations of one fact, each supersession retiring the last."""
    gen1 = _seed(conn, belief_id="belief:vm", body="The research VM is reached with ssh asa0.")
    second = supersede_v1(
        conn,
        target=gen1,
        body="The research VM is reached with ssh asa1.",
        reason="asa0 was retired in June",
        context=CONTEXT,
        actor="agent:test",
    )
    conn.commit()
    third = supersede_v1(
        conn,
        target=second["successor_id"],
        body="The research VM is reached with ssh asa2; asa1 was terminated.",
        reason="asa1 was terminated on 2026-08-20",
        context=CONTEXT,
        actor="agent:test",
    )
    conn.commit()
    return gen1, str(second["successor_id"]), str(third["successor_id"])


def test_a_three_generation_chain_inherits_its_ancestors_verdicts_once(tmp_path: Path) -> None:
    """Generation three carries generations one and two, counted exactly once.

    The count is the assertion that matters. ``prior_observations`` damping
    means a belief with one verdict barely moves and a belief with four moves
    four times as far, so an inheritance that silently doubled would be a
    ranking change disguised as a bugfix.
    """
    conn = _core(tmp_path)
    gen1, gen2, gen3 = _chain(conn)

    feedback_v1(conn, _serve(conn, gen1, query="reach the vm one"), outcome="used", note=None)
    feedback_v1(conn, _serve(conn, gen1, query="reach the vm two"), outcome="used", note=None)
    feedback_v1(conn, _serve(conn, gen2, query="reach the vm three"), outcome="helpful", note=None)
    conn.commit()

    history = retrieval_history_by_lineage(conn, {gen1, gen2, gen3})

    # gen1 is the oldest id: it owns two verdicts and inherits nothing.
    assert history[gen1] == {"n": 2, "signal": 2.0, "inherited_n": 0}
    # gen2 owns its own verdict and inherits gen1's two.
    assert history[gen2] == {"n": 3, "signal": 4.0, "inherited_n": 2}
    # gen3 has never been retrieved under its own id and inherits all three.
    assert history[gen3] == {"n": 3, "signal": 4.0, "inherited_n": 3}


def test_one_retrieval_serving_two_generations_counts_once(tmp_path: Path) -> None:
    """The lineage is a set of ids, not a sum over hops."""
    conn = _core(tmp_path)
    gen1, gen2, gen3 = _chain(conn)

    both = _serve(conn, gen1, gen2, query="reach the vm")
    feedback_v1(conn, both, outcome="used", note=None)
    conn.commit()

    history = retrieval_history_by_lineage(conn, {gen3})
    assert history[gen3] == {"n": 1, "signal": 1.0, "inherited_n": 1}


def test_a_verdict_the_belief_earned_itself_is_never_called_inherited(tmp_path: Path) -> None:
    """One retrieval serving the successor and an ancestor is the successor's."""
    conn = _core(tmp_path)
    gen1, _gen2, gen3 = _chain(conn)

    feedback_v1(conn, _serve(conn, gen3, gen1, query="reach the vm"), outcome="used", note=None)
    conn.commit()

    assert retrieval_history_by_lineage(conn, {gen3})[gen3] == {
        "n": 1,
        "signal": 1.0,
        "inherited_n": 0,
    }


def test_the_feedback_boost_moves_with_the_inherited_count(tmp_path: Path) -> None:
    """Same weights as ranking uses, asserted as numbers.

    Three ``used`` verdicts on retired ancestors: average 1.0, weight 0.125,
    damping 3/(3+3) = 0.5, so the successor's boost is 0.0625. Before this
    change the successor had no history of its own and scored 0.0 -- ranked as
    if the fact had never been served, on the day it was recompiled.
    """
    conn = _core(tmp_path)
    gen1, gen2, gen3 = _chain(conn)
    for index in range(3):
        feedback_v1(
            conn, _serve(conn, gen1, query=f"reach the vm {index}"), outcome="used", note=None
        )
    conn.commit()

    scores = _retrieval_feedback_scores(
        conn, {gen3}, weight=0.125, clamp=0.25, prior_observations=3.0
    )
    assert scores == {gen3: pytest.approx(0.0625)}


def test_an_inherited_record_of_harm_still_hits_the_clamp(tmp_path: Path) -> None:
    """Inherited history is bounded by the same clamp as first-hand history.

    Six ``harmful`` verdicts on the ancestors: average -4.0, weight 0.125,
    damping 6/(6+3), which is -0.333 before the clamp and -0.25 after it. A
    successor cannot be pushed further down by an ancestor's record than it
    could be by its own.
    """
    conn = _core(tmp_path)
    gen1, _gen2, gen3 = _chain(conn)
    for index in range(6):
        feedback_v1(
            conn, _serve(conn, gen1, query=f"harmful vm {index}"), outcome="harmful", note=None
        )
    conn.commit()

    clamped = _retrieval_feedback_scores(
        conn, {gen3}, weight=0.125, clamp=0.25, prior_observations=3.0
    )
    assert clamped[gen3] == pytest.approx(-0.25)


def test_a_belief_with_no_lineage_scores_exactly_as_before(tmp_path: Path) -> None:
    """The inheritance is additive: an only-generation belief is unchanged."""
    conn = _core(tmp_path)
    belief = _seed(conn, belief_id="belief:vm", body="The research VM is reached with ssh asa2.")
    feedback_v1(conn, _serve(conn, belief, query="reach the vm"), outcome="used", note=None)
    conn.commit()

    history = retrieval_history_by_lineage(conn, {belief})
    assert history[belief] == {"n": 1, "signal": 1.0, "inherited_n": 0}
    scores = _retrieval_feedback_scores(
        conn, {belief}, weight=0.125, clamp=0.25, prior_observations=3.0
    )
    # average 1.0 * 0.125 * 1/(1+3)
    assert scores == {belief: pytest.approx(0.03125)}


def test_an_unjudged_retrieval_is_not_history(tmp_path: Path) -> None:
    """``served`` and ``no_coverage`` are not verdicts and must not damp one."""
    conn = _core(tmp_path)
    gen1, _gen2, gen3 = _chain(conn)
    _serve(conn, gen1, query="reach the vm")
    _serve(conn, query="unrelated question")
    conn.commit()

    assert retrieval_history_by_lineage(conn, {gen1, gen3}) == {}
    assert _retrieval_feedback_scores(conn, {gen3}) == {}


# --------------------------------------------------------------------------- #
# Defect 1 -- what the two cores are each told, and what each one enforces
#
# The refusal lives in `feedback_v1`, which only a v1 core reaches. The
# instruction block and the tool description are served on every connection.
# A legacy v0 core cannot keep either half of the v1 promise: its
# `retrieval_uses.outcome` CHECK has no `no_coverage` value, and its receipts do
# not carry a served-item count on every path. So the text it is served must not
# make the claim. These tests pin the text to the core that is actually enforcing
# it -- in both directions, so narrowing the v1 wording fails too.
# --------------------------------------------------------------------------- #
def _legacy_core(tmp_path: Path):
    conn = connect(tmp_path / "legacy.sqlite")
    init_db(conn)
    return conn


def _feedback_tool(*, core_v1: bool = True) -> dict:
    return next(tool for tool in tool_list(core_v1=core_v1) if tool["name"] == "brain.feedback")


def _feedback_description(*, core_v1: bool) -> str:
    return str(_feedback_tool(core_v1=core_v1)["description"])


def _feedback_schema_enum() -> list[str]:
    return list(_feedback_tool()["inputSchema"]["properties"]["outcome"]["enum"])


def test_the_v1_instructions_state_the_refusal_and_the_recorded_outcome() -> None:
    text = instructions_text(core_v1=True)
    assert "refuses it" in text
    assert "no_coverage" in text


def test_the_legacy_instructions_claim_neither_refusal_nor_no_coverage() -> None:
    """A legacy core does neither, so it must not be described as doing either."""
    text = instructions_text(core_v1=False)
    assert "no_coverage" not in text
    assert "refuses" not in text
    # It still has to say the useful part: do not file, do not re-poll.
    assert "do not file brain.feedback for it" in text
    assert "do not re-poll the same query" in text


def test_neither_instruction_block_names_a_field_its_core_does_not_emit(
    tmp_path: Path,
) -> None:
    """`coverage.feedback_needed` exists only in the v1 envelope.

    Naming it to a legacy client is the same defect one layer down: prose
    pointing at an instrument that is not there. The legacy `coverage` block is
    built in `shared_context` and has no such key, so the legacy wording states
    the condition without it.
    """
    assert "feedback_needed" in instructions_text(core_v1=True)
    assert "feedback_needed" not in instructions_text(core_v1=False)

    legacy = _legacy_core(tmp_path)
    payload = json.loads(
        call_tool(
            legacy,
            {"name": "brain.context", "arguments": {"query": "anything", "context": {}}},
        )["content"][0]["text"]
    )
    assert "feedback_needed" not in payload["coverage"]
    legacy.close()

    core = _core(tmp_path)
    payload = json.loads(
        call_tool(
            core,
            {"name": "brain.context", "arguments": {"query": "anything", "context": {}}},
        )["content"][0]["text"]
    )
    assert payload["coverage"]["feedback_needed"] is False
    core.close()


def test_the_feedback_description_promises_the_refusal_only_on_a_v1_core() -> None:
    assert "no_coverage" in _feedback_description(core_v1=True)
    assert "refused" in _feedback_description(core_v1=True)
    legacy = _feedback_description(core_v1=False)
    assert "no_coverage" not in legacy
    assert "refus" not in legacy


def test_a_legacy_connection_is_served_the_legacy_instructions(tmp_path: Path) -> None:
    """The end-to-end seam: `initialize` picks the text from the open core."""
    legacy = _legacy_core(tmp_path)
    response = handle_request(legacy, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert "no_coverage" not in response["result"]["instructions"]
    legacy.close()

    core = _core(tmp_path)
    response = handle_request(core, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert "no_coverage" in response["result"]["instructions"]
    core.close()


def test_a_legacy_connection_is_listed_the_legacy_feedback_description(tmp_path: Path) -> None:
    legacy = _legacy_core(tmp_path)
    response = handle_request(legacy, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    described = next(
        tool for tool in response["result"]["tools"] if tool["name"] == "brain.feedback"
    )
    assert "no_coverage" not in described["description"]
    legacy.close()


def test_a_legacy_receipt_cannot_prove_a_read_served_nothing(tmp_path: Path) -> None:
    """Why the v1 guard is not simply ported to the legacy path.

    On a legacy core `brain.get` of a belief and `brain.digest` both write
    ``knowledge_id`` NULL with ``served_ids_json`` ``'[]'`` while having served
    an item, so an empty receipt there is not evidence of an empty packet: a
    served-count refusal would refuse feedback on reads that did serve. If this
    ever stops being true, the legacy text can make the claim -- and this test
    is what says so.
    """
    conn = _legacy_core(tmp_path)
    served_a_belief = log_retrieval_use(
        conn, None, runtime="mcp", task_ref="brain.get", note="object=belief", outcome="served"
    )
    conn.commit()
    row = conn.execute(
        "SELECT knowledge_id, served_ids_json FROM retrieval_uses WHERE id=?", (served_a_belief,)
    ).fetchone()
    assert row["knowledge_id"] is None
    assert row["served_ids_json"] == "[]"
    conn.close()


def test_a_legacy_core_cannot_hold_the_no_coverage_value(tmp_path: Path) -> None:
    """The other half of the reason: the legacy CHECK constraint forbids it."""
    conn = _legacy_core(tmp_path)
    retrieval_id = log_retrieval_use(conn, None, runtime="mcp", task_ref="brain.context")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        conn.execute(
            "UPDATE retrieval_uses SET outcome=? WHERE id=?",
            (NO_COVERAGE_OUTCOME, retrieval_id),
        )
    conn.rollback()
    conn.close()


def _file_legacy_feedback(conn, retrieval_id: str, outcome: str) -> None:
    call_tool(
        conn,
        {
            "name": "brain.feedback",
            "arguments": {"retrieval_use_id": retrieval_id, "outcome": outcome},
        },
    )


def test_the_legacy_feedback_path_reads_the_shared_outcome_vocabulary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy allowed-set is the shared tuple, not a second hand-typed copy.

    Two literals spelling one vocabulary is how the two paths drift apart, and
    comparing error text cannot tell them apart while they still agree. So the
    tuple is narrowed under the running server: if the legacy path reads it,
    ``helpful`` stops being accepted; if it carries its own copy, nothing moves.
    """
    conn = _legacy_core(tmp_path)
    retrieval_id = log_retrieval_use(
        conn, None, runtime="mcp", task_ref="brain.context", served_ids=["know:1"]
    )
    conn.commit()

    _file_legacy_feedback(conn, retrieval_id, "helpful")
    assert _outcome(conn, retrieval_id) == "helpful"

    monkeypatch.setattr("ocbrain.mcp.RELEVANCE_OUTCOMES", ("used",))
    with pytest.raises(ValueError, match="used"):
        _file_legacy_feedback(conn, retrieval_id, "helpful")
    assert _outcome(conn, retrieval_id) == "helpful"
    _file_legacy_feedback(conn, retrieval_id, "used")
    assert _outcome(conn, retrieval_id) == "used"
    conn.close()


def test_the_advertised_outcome_enum_is_the_same_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What clients are told they may send is what the server actually takes.

    The schema `enum` was a third hand-typed copy of the five outcomes, beside
    the v1 validator and the legacy one. Comparing it to the tuple would pass
    while they merely happen to agree, so the tuple is narrowed underneath and
    the published schema has to follow it.
    """
    assert _feedback_schema_enum() == list(RELEVANCE_OUTCOMES)
    monkeypatch.setattr("ocbrain.mcp.RELEVANCE_OUTCOMES", ("used", "harmful"))
    assert _feedback_schema_enum() == ["used", "harmful"]


# --------------------------------------------------------------------------- #
# Defect 1 -- each half of the reclassification predicate, on its own
#
# The command selects rows that (a) have no `retrieval_items` and (b) whose
# `served_ids_json` names nothing. Removing both conjuncts at once is caught by
# either half, which proves neither. One test per conjunct, each seeded so that
# only the conjunct under test can exclude the row.
# --------------------------------------------------------------------------- #
def test_a_receipt_naming_items_is_spared_even_with_no_item_rows(tmp_path: Path) -> None:
    """The `served_ids_json` half, alone.

    A core whose `retrieval_items` were never backfilled still has the id list
    in the receipt column. Without this conjunct its real, judged retrievals are
    swept into `no_coverage` under `--apply`, and nothing reports it.
    """
    conn = _core(tmp_path)
    belief = _seed(conn, belief_id="belief:vm", body="The research VM is reached with ssh asa2.")
    judged = _serve(conn, belief, query="how do I reach the research vm")
    conn.execute("DELETE FROM retrieval_items WHERE retrieval_use_id=?", (judged,))
    _force_verdict(conn, judged, "irrelevant")

    assert (
        conn.execute(
            "SELECT COUNT(*) FROM retrieval_items WHERE retrieval_use_id=?", (judged,)
        ).fetchone()[0]
        == 0
    )
    assert json.loads(
        conn.execute(
            "SELECT served_ids_json FROM retrieval_uses WHERE id=?", (judged,)
        ).fetchone()[0]
    ) == [belief]

    assert reclassify_no_coverage_receipts(conn, apply=False)["candidates"] == 0
    reclassify_no_coverage_receipts(conn, apply=True)
    conn.commit()
    assert _outcome(conn, judged) == "irrelevant"


def test_a_receipt_with_item_rows_is_spared_even_with_an_empty_id_list(tmp_path: Path) -> None:
    """The `NOT EXISTS` half, alone: the mirror-image damaged receipt."""
    conn = _core(tmp_path)
    belief = _seed(conn, belief_id="belief:vm", body="The research VM is reached with ssh asa2.")
    judged = _serve(conn, belief, query="how do I reach the research vm")
    conn.execute("UPDATE retrieval_uses SET served_ids_json='[]' WHERE id=?", (judged,))
    _force_verdict(conn, judged, "irrelevant")

    assert reclassify_no_coverage_receipts(conn, apply=False)["candidates"] == 0
    reclassify_no_coverage_receipts(conn, apply=True)
    conn.commit()
    assert _outcome(conn, judged) == "irrelevant"


# --------------------------------------------------------------------------- #
# Defect 2 -- the walk's depth, against the forward walk's bound
# --------------------------------------------------------------------------- #
def test_the_lineage_walk_is_not_bounded_by_the_resolution_hop_limit(
    tmp_path: Path,
) -> None:
    """A chain longer than `MAX_RESOLUTION_HOPS` still carries all its history.

    The forward walk in `mcp_v1._resolve_supersession_chain` stops after ten
    hops because each hop is a separate belief read and a long chain there is a
    corpus problem rather than a read to satisfy. This walk answers a different
    question -- *what is this belief's whole record* -- from one query and an
    in-memory traversal, and the live core's deepest serving lineage was
    measured at 11 generations on 2026-08-28, one past that bound. Bounding it
    at ten would silently drop a generation of verdicts from ranking today.
    """
    conn = _core(tmp_path)
    generations = [_seed(conn, belief_id="belief:vm", body="The VM is reached with ssh asa0.")]
    for step in range(MAX_RESOLUTION_HOPS + 2):
        successor = supersede_v1(
            conn,
            target=generations[-1],
            body=f"The VM is reached with ssh asa{step + 1}.",
            reason=f"asa{step} was retired",
            context=CONTEXT,
            # A new actor per generation: the supersede rate cap is per caller,
            # and a chain this deep on the live core is many callers over weeks,
            # not one agent in an afternoon.
            actor=f"agent:test-{step}",
        )
        conn.commit()
        assert successor.get("mode") == "direct", successor.get("pending_reason")
        generations.append(str(successor["successor_id"]))
    assert len(generations) == MAX_RESOLUTION_HOPS + 3 > MAX_RESOLUTION_HOPS

    for index, ancestor in enumerate(generations[:-1]):
        feedback_v1(conn, _serve(conn, ancestor, query=f"vm {index}"), outcome="used", note=None)
    conn.commit()

    newest = generations[-1]
    history = retrieval_history_by_lineage(conn, {newest})
    assert history[newest]["n"] == len(generations) - 1
    assert history[newest]["inherited_n"] == len(generations) - 1


# --------------------------------------------------------------------------- #
# Defect 2 -- history recorded under a collapsed alias
# --------------------------------------------------------------------------- #
def test_history_recorded_under_an_alias_is_attributed_to_the_belief(tmp_path: Path) -> None:
    """Retrieval rows written before an alias was collapsed still count.

    Order is the whole test. `record_core_v1_retrieval` resolves the id at write
    time, so a retrieval served *after* the alias exists is already stored under
    the canonical id and proves nothing about this walk. The rows that need the
    walk are the ones written **before** the collapse, which keep the old id
    forever. `object_aliases` has 0 rows on the live core, so nothing in the
    corpus exercises this and the whole alias block could be deleted with the
    rest of the suite green -- it is the mechanism the SQL this replaced named in
    its own docstring, so it gets a falsifier rather than a comment.
    """
    conn = _core(tmp_path)
    belief = _seed(conn, belief_id="belief:vm", body="The research VM is reached with ssh asa2.")
    alias_id = "belief:vm-old-id"

    under_the_alias = _serve(conn, alias_id, query="reach the vm")
    feedback_v1(conn, under_the_alias, outcome="helpful", note=None)
    conn.commit()
    assert (
        conn.execute(
            "SELECT object_id FROM retrieval_items WHERE retrieval_use_id=?", (under_the_alias,)
        ).fetchone()[0]
        == alias_id
    )

    source_event = str(conn.execute("SELECT id FROM brain_events LIMIT 1").fetchone()[0])
    conn.execute(
        "INSERT INTO object_aliases(alias_id, canonical_id, object_kind, source, source_event_id) "
        "VALUES (?, ?, 'belief', 'test', ?)",
        (alias_id, belief, source_event),
    )
    conn.commit()

    history = retrieval_history_by_lineage(conn, {belief})
    assert history[belief] == {"n": 1, "signal": 2.0, "inherited_n": 0}

    # And an heir inherits it, since the alias hangs off the ancestor it walks to.
    successor = supersede_v1(
        conn,
        target=belief,
        body="The research VM is reached with ssh asa2; asa1 was terminated.",
        reason="rewritten",
        context=CONTEXT,
        actor="agent:test",
    )
    conn.commit()
    heir = str(successor["successor_id"])
    assert retrieval_history_by_lineage(conn, {heir})[heir] == {
        "n": 1,
        "signal": 2.0,
        "inherited_n": 1,
    }
