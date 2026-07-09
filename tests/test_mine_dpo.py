from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from ocbrain.config import load_config
from ocbrain.dataset.mine_dpo import (
    find_event_pairs,
    find_transcript_pairs,
    mine_dpo,
    scope_tag_to_privacy,
)
from ocbrain.dataset.transcripts import Session, Turn
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

CFG = load_config()
CFG_STRICT_ONLY = replace(CFG, dataset=replace(CFG.dataset, dpo_relaxed_gate=False))
A1 = "The answer is forty-two according to my first and frankly hasty guess about this."
CHOSEN = "Actually the deploy is driven by scripts/run.sh which you invoke before shipping."
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
    confidential = ScopeTag("project", "project:p", visibility="confidential",
                            egress_policy="hosted_ok")
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
    mid = "A partial second attempt that is still not correct but long enough here to count."
    session = _sess(
        _u("original request please"),
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
    upsert_evidence(conn, source_type="openclaw_history_file", source_runtime="openclaw",
                    source_uri="/x/s.jsonl", content_hash="fp1", claim="t",
                    privacy_scope="workspace")
    session = _sess(_u("help deploy"), _a(A1), _u("no that's wrong"), _a(CHOSEN))
    result = mine_dpo(conn, cfg=CFG, sessions=[session], include_events=False)
    assert result["stored"] == 1
    row = conn.execute("SELECT example_json, evidence_ids FROM dataset_examples").fetchone()
    record = json.loads(row["example_json"])
    assert record["preferred_output"][0]["content"] == CHOSEN
    assert record["non_preferred_output"][0]["content"] == A1
    assert json.loads(row["evidence_ids"])  # provenance present


def test_event_edit_pair(tmp_path: Path):
    conn = _conn(tmp_path)
    evidence_id = upsert_evidence(conn, source_type="observation", source_uri="/e1",
                                  content_hash="h1", claim="the pipeline runs nightly at 2am")
    proposal = propose_compilation(conn, belief_id="b1",
                                   body="The pipeline runs at midnight every day.",
                                   evidence_ids=[evidence_id], scope=global_scope())
    decide_compilation(conn, proposal_event_id=proposal, decision="edit",
                       edited_body="The pipeline runs nightly at 2am, not midnight.")
    pairs = find_event_pairs(conn, CFG)
    edits = [p for p in pairs if p.correction_kind == "event_edit"]
    assert len(edits) == 1
    assert edits[0].confidence == 0.85
    assert edits[0].rejected.startswith("The pipeline runs at midnight")
    assert "2am" in edits[0].chosen
    assert edits[0].evidence_ids == (evidence_id,)


def test_event_correction_hard_and_markwrong(tmp_path: Path):
    conn = _conn(tmp_path)
    evidence_id = upsert_evidence(conn, source_type="observation", source_uri="/e2",
                                  content_hash="h2", claim="canonical value")
    know_id = upsert_knowledge(conn, knowledge_type="value", gate="auto", subject="answer",
                               predicate="value",
                               value_text="the old wrong value is forty two which we distrust now",
                               confidence=0.7)
    link_knowledge_evidence(conn, know_id, evidence_id)
    record_correction(conn, target_layer="knowledge", target_id=know_id, op="edit",
                      body="the correct value is eighty five per the latest careful measurement",
                      hard=True)
    pairs = [p for p in find_event_pairs(conn, CFG) if p.correction_kind == "event_correction"]
    assert len(pairs) == 1
    assert pairs[0].hard is True and pairs[0].confidence == 0.95
    # mark_wrong without a replacement yields no pair
    record_correction(conn, target_layer="knowledge", target_id=know_id, op="mark_wrong",
                      body=None, hard=True)
    still = [p for p in find_event_pairs(conn, CFG) if p.correction_kind == "event_correction"]
    assert len(still) == 1


def test_supersession_pair(tmp_path: Path):
    conn = _conn(tmp_path)
    evidence_id = upsert_evidence(conn, source_type="observation", source_uri="/e3",
                                  content_hash="h3", claim="winner evidence")
    loser = upsert_knowledge(conn, knowledge_type="value", gate="auto", subject="q",
                             predicate="v1",
                             value_text="the losing value is the old number that we used before")
    winner = upsert_knowledge(conn, knowledge_type="value", gate="auto", subject="q",
                              predicate="v2",
                              value_text="the winning value is the new number we use going forward")
    link_knowledge_evidence(conn, winner, evidence_id)
    conn.execute("UPDATE knowledge SET superseded_by = ?, updated_at = ? WHERE id = ?",
                 (winner, now_iso(), loser))
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
        _a(A1),
        _u(FIX_LASTWORD),
    )
    assert find_transcript_pairs(session, CFG_STRICT_ONLY) == []
    pairs = find_transcript_pairs(session, CFG)
    assert len(pairs) == 1
    assert pairs[0].gate == "relaxed"
    assert pairs[0].rejected == A1
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
        _a(A1),
        _u(FIX_LASTWORD),
    )
    assert find_transcript_pairs(session, CFG_STRICT_ONLY) == []


def test_mine_dpo_stores_relaxed_gate_metadata(tmp_path: Path):
    conn = _conn(tmp_path)
    upsert_evidence(conn, source_type="openclaw_history_file", source_runtime="openclaw",
                    source_uri="/x/s.jsonl", content_hash="fpR", claim="t",
                    privacy_scope="workspace")
    session = _sess(_u("where does location live?"), _a(A1), _u(FIX_LASTWORD))
    result = mine_dpo(conn, cfg=CFG, sessions=[session], include_events=False)
    assert result["stored"] == 1
    row = conn.execute("SELECT example_json FROM dataset_examples").fetchone()
    record = json.loads(row["example_json"])
    assert record["metadata"]["gate"] == "relaxed"
    assert record["preferred_output"][0]["content"] == FIX_LASTWORD
    assert record["non_preferred_output"][0]["content"] == A1
