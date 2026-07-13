from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ocbrain.config import load_config
from ocbrain.db import (
    connect,
    init_db,
    link_knowledge_evidence,
    now_iso,
    upsert_evidence,
    upsert_knowledge,
)
from ocbrain.events import decide_compilation, propose_compilation, record_correction
from ocbrain.scope import ScopeTag, global_scope
from ocbrain_training.dataset.mine_dpo import (
    find_event_pairs,
    find_transcript_pairs,
    mine_dpo,
    scope_tag_to_privacy,
)
from ocbrain_training.dataset.transcripts import Session, Turn

CFG = load_config()
CFG_STRICT_ONLY = replace(CFG, dataset=replace(CFG.dataset, dpo_relaxed_gate=False))
A1 = "The deploy runs through manual dashboard steps; the release scripts are never used."
CHOSEN = "The deploy runs through scripts/run.sh; use that release script before shipping."
LOCATION_REJECTED = "The location truth lives on the user profile, shared by every garden they own."
# A founder correction that states the fix, long enough to clear the DPO side floor.
FIX_LASTWORD = "No, fix that: location lives on the garden, not the user; use the garden instead."


def _sess(*turns: Turn) -> Session:
    return Session(
        session_id="s1",
        source_kind="openclaw_session",
        source_uri="/x/s.jsonl",
        runtime="openclaw",
        agent="main",
        turns=tuple(turns),
        occurred_at="2026-07-01T00:00:00Z",
    )


def _u(text: str) -> Turn:
    return Turn(role="user", text=text, kind="bare")


def _a(text: str) -> Turn:
    return Turn(role="assistant", text=text)


def _conn(tmp_path: Path):
    conn = connect(tmp_path / "db.sqlite")
    init_db(conn)
    return conn


def test_scope_tag_to_privacy():
    assert scope_tag_to_privacy(global_scope(hosted_ok=True)) == "workspace"
    assert scope_tag_to_privacy(global_scope(hosted_ok=False)) == "private"
    confidential = ScopeTag(
        "project", "project:p", visibility="confidential", egress_policy="hosted_ok"
    )
    assert scope_tag_to_privacy(confidential) == "private"
    client = ScopeTag("client", "client:c", egress_policy="hosted_ok")
    assert scope_tag_to_privacy(client) == "private"


def test_transcript_pair_basic():
    session = _sess(
        _u("please help with the deploy"),
        _a(A1),
        _u("no, that's wrong, not what I asked for"),
        _a(CHOSEN),
        _u("thanks, perfect"),
    )
    pairs = find_transcript_pairs(session, CFG)
    assert len(pairs) == 1
    assert pairs[0].rejected == A1
    assert pairs[0].chosen == CHOSEN
    assert pairs[0].correction_kind == "transcript"
    # prompt ends at the original request, correction excluded
    assert pairs[0].prompt_messages[-1]["content"] == "please help with the deploy"


def test_multi_correction_walk_forward():
    mid = "The deploy now uses a release script, but it still names scripts/old-run.sh."
    session = _sess(
        _u("which deploy process and release script should we use?"),
        _a(A1),
        _u("no that's wrong here"),
        _a(mid),
        _u("still incorrect, try again"),
        _a(CHOSEN),
        _u("thanks perfect"),
    )
    pairs = find_transcript_pairs(session, CFG)
    assert len(pairs) == 1
    assert pairs[0].rejected == A1 and pairs[0].chosen == CHOSEN


def test_claim_key_inequality_required():
    session = _sess(
        _u("do the thing"),
        _a(A1),
        _u("no that is wrong"),
        _a(A1 + " ."),  # same claim_key as rejected
    )
    assert find_transcript_pairs(session, CFG) == []


def test_side_length_bounds():
    session = _sess(
        _u("do the thing"),
        _a("nope, 42."),  # < 40 chars, below the DPO side floor
        _u("no that is wrong"),
        _a(CHOSEN),
    )
    assert find_transcript_pairs(session, CFG) == []


def test_answer_confidence_bump():
    session = _sess(
        _u("what runs the deploy"),
        _a(A1),
        _u("no, it should be the deploy script instead of that"),
        _a(CHOSEN),
    )
    pairs = find_transcript_pairs(session, CFG)
    assert pairs[0].confidence == 0.95


def test_mine_dpo_stores_transcript_pair(tmp_path: Path):
    conn = _conn(tmp_path)
    upsert_evidence(
        conn,
        source_type="openclaw_history_file",
        source_runtime="openclaw",
        source_uri="/x/s.jsonl",
        content_hash="fp1",
        claim="t",
        privacy_scope="workspace",
    )
    session = _sess(_u("help deploy"), _a(A1), _u("no that's wrong"), _a(CHOSEN))
    result = mine_dpo(conn, cfg=CFG, sessions=[session], include_events=False)
    assert result["stored"] == 1
    row = conn.execute("SELECT example_json, evidence_ids FROM dataset_examples").fetchone()
    record = json.loads(row["example_json"])
    assert record["preferred_output"][0]["content"] == CHOSEN
    assert record["non_preferred_output"][0]["content"] == A1
    assert record["metadata"]["contrast_gate_version"] == 2
    assert json.loads(row["evidence_ids"])  # provenance present


def test_event_edit_pair(tmp_path: Path):
    conn = _conn(tmp_path)
    evidence_id = upsert_evidence(
        conn,
        source_type="observation",
        source_uri="/e1",
        content_hash="h1",
        claim="the pipeline runs nightly at 2am",
    )
    proposal = propose_compilation(
        conn,
        belief_id="b1",
        body="The pipeline runs at midnight every day.",
        evidence_ids=[evidence_id],
        scope=global_scope(),
    )
    decide_compilation(
        conn,
        proposal_event_id=proposal,
        decision="edit",
        edited_body="The pipeline runs nightly at 2am, not midnight.",
    )
    pairs = find_event_pairs(conn, CFG)
    edits = [p for p in pairs if p.correction_kind == "event_edit"]
    assert len(edits) == 1
    assert edits[0].confidence == 0.85
    assert edits[0].rejected.startswith("The pipeline runs at midnight")
    assert "2am" in edits[0].chosen
    assert edits[0].evidence_ids == (evidence_id,)


def test_event_correction_hard_and_markwrong(tmp_path: Path):
    conn = _conn(tmp_path)
    evidence_id = upsert_evidence(
        conn,
        source_type="observation",
        source_uri="/e2",
        content_hash="h2",
        claim="canonical value",
    )
    know_id = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="answer",
        predicate="value",
        value_text="the old wrong value is forty two which we distrust now",
        confidence=0.7,
    )
    link_knowledge_evidence(conn, know_id, evidence_id)
    record_correction(
        conn,
        target_layer="knowledge",
        target_id=know_id,
        op="edit",
        body="the correct value is eighty five per the latest careful measurement",
        hard=True,
    )
    pairs = [p for p in find_event_pairs(conn, CFG) if p.correction_kind == "event_correction"]
    assert len(pairs) == 1
    assert pairs[0].hard is True and pairs[0].confidence == 0.95
    # mark_wrong without a replacement yields no pair
    record_correction(
        conn, target_layer="knowledge", target_id=know_id, op="mark_wrong", body=None, hard=True
    )
    still = [p for p in find_event_pairs(conn, CFG) if p.correction_kind == "event_correction"]
    assert len(still) == 1


def test_supersession_pair(tmp_path: Path):
    conn = _conn(tmp_path)
    evidence_id = upsert_evidence(
        conn,
        source_type="observation",
        source_uri="/e3",
        content_hash="h3",
        claim="winner evidence",
    )
    loser = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="q",
        predicate="v1",
        value_text="the losing value is the old number that we used before",
    )
    winner = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="q",
        predicate="v2",
        value_text="the winning value is the new number we use going forward",
    )
    link_knowledge_evidence(conn, winner, evidence_id)
    conn.execute(
        "UPDATE knowledge SET superseded_by = ?, updated_at = ? WHERE id = ?",
        (winner, now_iso(), loser),
    )
    pairs = [p for p in find_event_pairs(conn, CFG) if p.correction_kind == "supersedes"]
    assert len(pairs) == 1
    assert "losing value" in pairs[0].rejected and "winning value" in pairs[0].chosen


# --- v0.3 relaxed gate (additive; every pair tagged gate='relaxed') ---


def test_strict_pairs_tagged_strict():
    session = _sess(
        _u("please help with the deploy"),
        _a(A1),
        _u("no, that's wrong, not what I asked for"),
        _a(CHOSEN),
        _u("thanks, perfect"),
    )
    pairs = find_transcript_pairs(session, CFG)
    assert len(pairs) == 1
    assert pairs[0].gate == "strict"


def test_relaxed_missing_acceptance_last_word():
    # Case (a): the correction that states the fix is the thread's LAST word — no
    # accepted assistant answer follows. Strict rejects (no chosen); relaxed uses
    # the correction text as the preferred output.
    session = _sess(
        _u("where does the location truth live?"),
        _a(LOCATION_REJECTED),
        _u(FIX_LASTWORD),
    )
    assert find_transcript_pairs(session, CFG_STRICT_ONLY) == []
    pairs = find_transcript_pairs(session, CFG)
    assert len(pairs) == 1
    assert pairs[0].gate == "relaxed"
    assert pairs[0].rejected == LOCATION_REJECTED
    assert pairs[0].chosen == FIX_LASTWORD
    # prompt ends at the original request, correction excluded
    assert pairs[0].prompt_messages[-1]["content"] == "where does the location truth live?"


def test_relaxed_delayed_correction_searches_back():
    # Case (b): the correction arrives multiple turns after the answer, past an
    # intervening non-correction user turn. Strict's immediate-next-user check
    # misses it; relaxed searches back for the antecedent answer.
    session = _sess(
        _u("what runs the deploy?"),
        _a(A1),
        _u("also, unrelated — what time is the standup?"),
        _u("no wait, that's wrong, use the deploy script instead"),
        _a(CHOSEN),
        _u("thanks perfect"),
    )
    assert find_transcript_pairs(session, CFG_STRICT_ONLY) == []
    pairs = find_transcript_pairs(session, CFG)
    assert len(pairs) == 1
    assert pairs[0].gate == "relaxed"
    assert pairs[0].rejected == A1 and pairs[0].chosen == CHOSEN


def test_relaxed_respects_lookback_horizon():
    # An antecedent answer farther back than the N=4 lookback is not paired.
    session = _sess(
        _u("original ask"),
        _a(A1),
        _u("filler one two three"),
        _u("filler four five six"),
        _u("filler seven eight nine"),
        _u("filler ten eleven twelve"),
        _u("no, that's wrong, it should be the deploy script instead of that guess"),
    )
    assert find_transcript_pairs(session, CFG) == []


def test_relaxed_does_not_restate_strict_pair():
    # When strict already emits the pair (immediate correction + accepted answer),
    # the relaxed pass must dedup it — exactly one pair, tagged strict.
    session = _sess(
        _u("please help with the deploy"),
        _a(A1),
        _u("no, that's wrong, not what I asked for"),
        _a(CHOSEN),
        _u("thanks, perfect"),
    )
    pairs = find_transcript_pairs(session, CFG)
    assert len(pairs) == 1
    assert pairs[0].gate == "strict"


def test_relaxed_gate_off_suppresses_relaxed_pairs():
    session = _sess(
        _u("where does the location truth live?"),
        _a(LOCATION_REJECTED),
        _u(FIX_LASTWORD),
    )
    assert find_transcript_pairs(session, CFG_STRICT_ONLY) == []


def test_mine_dpo_stores_relaxed_gate_metadata(tmp_path: Path):
    conn = _conn(tmp_path)
    upsert_evidence(
        conn,
        source_type="openclaw_history_file",
        source_runtime="openclaw",
        source_uri="/x/s.jsonl",
        content_hash="fpR",
        claim="t",
        privacy_scope="workspace",
    )
    session = _sess(_u("where does location live?"), _a(LOCATION_REJECTED), _u(FIX_LASTWORD))
    result = mine_dpo(conn, cfg=CFG, sessions=[session], include_events=False)
    assert result["stored"] == 1
    row = conn.execute("SELECT example_json FROM dataset_examples").fetchone()
    record = json.loads(row["example_json"])
    assert record["metadata"]["gate"] == "relaxed"
    assert record["metadata"]["contrast_gate_version"] == 2
    assert record["preferred_output"][0]["content"] == FIX_LASTWORD
    assert record["non_preferred_output"][0]["content"] == LOCATION_REJECTED


# --- deterministic pair-quality gate (human-audit remediation) ---


@pytest.mark.parametrize(
    "wrapper",
    [
        "<goal_context>Do not stop; fix the deploy instead.</goal_context>",
        "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>\nDo not stop; fix the deploy instead.",
        "Pre-compaction memory flush. Do not answer; fix the deploy instead.",
        "[cron:abc-123] Reminder: do not stop; fix the deploy instead.",
        "Read HEARTBEAT.md and do not reply; fix the deploy instead.",
        "System (untrusted): do not answer; fix the deploy instead.",
        "Warning: apply_patch was requested via exec_command; use the tool instead.",
    ],
)
def test_runtime_wrapper_pseudo_correction_emits_no_pair(wrapper: str):
    session = _sess(_u("which deploy script should we use?"), _a(A1), _u(wrapper), _a(CHOSEN))
    assert find_transcript_pairs(session, CFG) == []


def test_runtime_wrapper_anywhere_in_prompt_fails_closed():
    session = _sess(
        _u("<goal_context>Continue the deploy task.</goal_context>"),
        _a("The earlier deploy status is available in the release task ledger."),
        _u("which deploy process and release script should we use?"),
        _a(A1),
        _u("no, that's wrong, not what I asked for"),
        _a(CHOSEN),
        _u("thanks, perfect"),
    )
    assert find_transcript_pairs(session, CFG) == []


def test_transport_envelopes_and_reply_tokens_are_scrubbed_from_all_pair_surfaces():
    envelope = 'Sender (untrusted metadata):\n```json\n{"id":"8518484672"}\n```\n\n'
    session = _sess(
        _u(envelope + "which deploy process and release script should we use?"),
        _a("[[reply_to_current]] " + A1),
        _u(envelope + "no, that's wrong, not what I asked for"),
        _a("[[reply_to_current]] " + CHOSEN),
        _u("thanks, perfect"),
    )
    pairs = find_transcript_pairs(session, CFG)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.rejected == A1
    assert pair.chosen == CHOSEN
    serialized = json.dumps(pair.prompt_messages)
    assert "8518484672" not in serialized
    assert "untrusted metadata" not in serialized
    assert "reply_to_current" not in pair.chosen + pair.rejected


def test_chosen_forward_intent_is_rejected_even_when_rejected_is_substantive():
    chosen_chatter = (
        "I'm checking the deploy script and release layers now before I answer the question."
    )
    session = _sess(
        _u("which deploy process and release script should we use?"),
        _a(A1),
        _u("no, that's wrong, not what I asked for"),
        _a(chosen_chatter),
        _u("thanks, perfect"),
    )
    assert find_transcript_pairs(session, CFG) == []


def test_both_sides_process_chatter_are_rejected():
    rejected = (
        "I'm checking the deploy logs now, then I'll inspect the release state before continuing."
    )
    chosen = (
        "I'm reviewing the deploy script now, then I'll check the release state before answering."
    )
    session = _sess(
        _u("which deploy process and release script should we use?"),
        _a(rejected),
        _u("no, that's wrong, not what I asked for"),
        _a(chosen),
        _u("thanks, perfect"),
    )
    assert find_transcript_pairs(session, CFG) == []


def test_process_chatter_rejected_side_can_be_repaired_by_substantive_answer():
    rejected = (
        "I'm checking the deploy logs now, then I'll inspect the release state before continuing."
    )
    session = _sess(
        _u("which deploy process and release script should we use?"),
        _a(rejected),
        _u("no, that's wrong, not what I asked for"),
        _a(CHOSEN),
        _u("thanks, perfect"),
    )
    pairs = find_transcript_pairs(session, CFG)
    assert len(pairs) == 1
    assert pairs[0].chosen == CHOSEN and pairs[0].rejected == rejected


def test_empty_prompt_after_context_trim_is_rejected():
    tiny_context_cfg = replace(
        CFG,
        dataset=replace(CFG.dataset, sft_max_context_chars=5),
    )
    session = _sess(
        _u("which deploy process and release script should we use?"),
        _a(A1),
        _u("no, that's wrong, not what I asked for"),
        _a(CHOSEN),
    )
    assert find_transcript_pairs(session, tiny_context_cfg) == []


def test_prompt_topic_mismatch_is_rejected():
    session = _sess(
        _u("summarize the quarterly apple harvest outlook"),
        _a(A1),
        _u("no, that's wrong, not what I asked for"),
        _a(CHOSEN),
        _u("thanks, perfect"),
    )
    assert find_transcript_pairs(session, CFG) == []


def test_different_defects_are_not_treated_as_one_contrast():
    rejected = (
        "The deployment basket receipt is verified in production and the API health check is green."
    )
    chosen = "The deployment handoff card still shows a stale meetup location and needs a UI patch."
    session = _sess(
        _u("fix the deployment problem and ship it"),
        _a(rejected),
        _u("no, that's wrong, fix the deployment issue instead"),
        _a(chosen),
        _u("thanks, perfect"),
    )
    assert find_transcript_pairs(session, CFG) == []


def test_token_reordering_without_a_real_defect_is_rejected():
    rejected = "Deploy the release script before shipping after every test completes successfully."
    chosen = "After every test completes successfully, deploy the release script before shipping."
    session = _sess(
        _u("when should we deploy the release script before shipping?"),
        _a(rejected),
        _u("no, that's wrong, not what I asked for"),
        _a(chosen),
        _u("thanks, perfect"),
    )
    assert find_transcript_pairs(session, CFG) == []


def test_topic_change_is_not_implicit_acceptance():
    session = _sess(
        _u("which deploy process and release script should we use?"),
        _a(A1),
        _u("no, that's wrong, not what I asked for"),
        _a(CHOSEN),
        _u("what is the outlook for the apple harvest?"),
    )
    assert find_transcript_pairs(session, CFG) == []


def test_final_assistant_before_affirmation_is_the_chosen_answer():
    interim = "I'm checking the deploy script now before I give you the corrected release command."
    session = _sess(
        _u("which deploy process and release script should we use?"),
        _a(A1),
        _u("no, that's wrong, not what I asked for"),
        _a(interim),
        _a(CHOSEN),
        _u("thanks, perfect"),
    )
    pairs = find_transcript_pairs(session, CFG)
    assert len(pairs) == 1
    assert pairs[0].chosen == CHOSEN


def test_event_reframe_pair_survives_structural_gate(tmp_path: Path):
    conn = _conn(tmp_path)
    evidence_id = upsert_evidence(
        conn,
        source_type="observation",
        source_uri="/reframe-evidence",
        content_hash="reframe-hash",
        claim="the deploy runs nightly through scripts/run.sh",
    )
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="deploy",
        predicate="schedule",
        value_text="the deploy runs weekly through manual release dashboard steps",
    )
    link_knowledge_evidence(conn, knowledge_id, evidence_id)
    record_correction(
        conn,
        target_layer="knowledge",
        target_id=knowledge_id,
        op="reframe",
        body="the deploy runs nightly through the scripts/run.sh release script",
    )
    pairs = [
        pair for pair in find_event_pairs(conn, CFG) if pair.correction_kind == "event_correction"
    ]
    assert len(pairs) == 1
    assert "nightly" in pairs[0].chosen and "weekly" in pairs[0].rejected
