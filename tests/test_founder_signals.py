"""Founder feedback signals: attribution, weighting, DPO provenance, isolation.

Covers the founder-feedback-author feature end to end with SYNTHETIC placeholder
telegram ids only (this repo is public — never a real sender id):

* config helpers ``founder_ids`` / ``founder_weight``,
* the transcript parser stamping ``authored_by`` for a founder without persona
  verifying them, and multi-user groups keeping distinct senders un-collapsed,
* ``review`` scaling corrections/approvals/thanks by the author's founder weight
  and the label fold respecting that weight,
* ``mine_dpo`` tagging a founder-issued correction pair with author provenance,
* PERSONA ISOLATION: a non-persona author (a founder or any identified stranger)
  never enters the persona/voice stream.
"""

from __future__ import annotations

import json
from pathlib import Path

from ocbrain.autolabel import label_from_signals
from ocbrain.config import founder_ids, founder_weight, load_config
from ocbrain.dataset.mine_dpo import find_transcript_pairs, mine_dpo
from ocbrain.dataset.mine_persona import mine_persona, telegram_examples
from ocbrain.dataset.transcripts import Session, Turn, classify_user_text, parse_openclaw_session
from ocbrain.db import connect, init_db, upsert_evidence
from ocbrain.review import review_session

# --- synthetic identities (NOT real telegram ids) ---------------------------- #
FOUNDER = "700000001"  # co-founder: feedback author, weight 2.0, NOT persona
OPERATOR = "700000002"  # operator: persona author AND feedback author, weight 1.5
STRANGER = "700000009"  # generic group member: no founder weight, not persona


def _cfg(tmp_path: Path):
    path = tmp_path / "cfg.json"
    path.write_text(
        json.dumps(
            {
                "dataset": {
                    "persona_author_ids": [OPERATOR],
                    "persona_direct_agents": ["main"],
                    "founder_feedback_authors": [
                        {"id": FOUNDER, "weight": 2.0},
                        {"id": OPERATOR, "weight": 1.5},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    return load_config(path)


def _conn(tmp_path: Path):
    conn = connect(tmp_path / "db.sqlite")
    init_db(conn)
    return conn


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def test_founder_weight_and_ids(tmp_path: Path):
    cfg = _cfg(tmp_path)
    assert set(founder_ids(cfg)) == {FOUNDER, OPERATOR}
    assert founder_weight(cfg, FOUNDER) == 2.0
    assert founder_weight(cfg, OPERATOR) == 1.5
    assert founder_weight(cfg, STRANGER) == 1.0  # generic user
    assert founder_weight(cfg, None) == 1.0
    # Empty default config: everyone is generic.
    empty = load_config(tmp_path / "missing.json")
    assert founder_ids(empty) == []
    assert founder_weight(empty, FOUNDER) == 1.0


# --------------------------------------------------------------------------- #
# Parser attribution + isolation of the verify flag
# --------------------------------------------------------------------------- #
def _envelope(sender_id: str, message: str) -> str:
    env = {"message_id": "1", "sender_id": sender_id, "is_group_chat": True}
    return (
        "Conversation info (untrusted metadata):\n```json\n"
        + json.dumps(env)
        + "\n```\n"
        + message
    )


def test_parser_stamps_founder_without_verifying(tmp_path: Path):
    cfg = _cfg(tmp_path)
    fids = founder_ids(cfg)
    pids = cfg.dataset.persona_author_ids
    # Founder (non-persona): attributed but NOT verified.
    founder = classify_user_text(
        _envelope(FOUNDER, "that copy is wrong, use ready not available"),
        author_ids=pids,
        founder_ids=fids,
    )
    assert founder.authored_by == FOUNDER
    assert founder.sender_verified is False
    # Operator (persona author): attributed AND verified.
    operator = classify_user_text(
        _envelope(OPERATOR, "ship it tonight please"), author_ids=pids, founder_ids=fids
    )
    assert operator.authored_by == OPERATOR
    assert operator.sender_verified is True
    # A stranger with no founder/persona membership stays anonymous+unverified.
    stranger = classify_user_text(
        _envelope(STRANGER, "hello there everyone"), author_ids=pids, founder_ids=fids
    )
    assert stranger.authored_by is None
    assert stranger.sender_verified is False


def test_multiuser_group_keeps_distinct_senders(tmp_path: Path):
    cfg = _cfg(tmp_path)
    lines = [
        {"type": "session", "id": "g1", "version": "1", "timestamp": "2026-07-01T00:00:00Z"},
        {"type": "message", "message": {"role": "assistant", "content": "draft ready"}},
        {"type": "message", "message": {
            "role": "user", "content": _envelope(FOUNDER, "call it ready, not available")}},
        {"type": "message", "message": {
            "role": "user", "content": _envelope(OPERATOR, "agreed, do that")}},
    ]
    path = tmp_path / "g1.jsonl"
    path.write_text("\n".join(json.dumps(o) for o in lines) + "\n", encoding="utf-8")
    session = parse_openclaw_session(
        path, author_ids=cfg.dataset.persona_author_ids, founder_ids=founder_ids(cfg)
    )
    user_turns = [t for t in session.turns if t.role == "user"]
    # Two distinct senders back-to-back must NOT collapse into one turn.
    assert len(user_turns) == 2
    assert user_turns[0].authored_by == FOUNDER
    assert user_turns[1].authored_by == OPERATOR


# --------------------------------------------------------------------------- #
# Review: founder-weighted session signals + fold respecting the weight
# --------------------------------------------------------------------------- #
def _turn(role: str, text: str, *, authored_by: str | None = None) -> Turn:
    return Turn(role=role, text=text, authored_by=authored_by,
                kind="telegram_envelope" if authored_by else "")


def _session(*turns: Turn) -> Session:
    return Session(session_id="s1", source_kind="openclaw_session",
                   source_uri="/x/s.jsonl", runtime="openclaw", agent="main",
                   turns=tuple(turns), occurred_at="2026-07-01T00:00:00Z")


def _correction_row(conn, kind: str = "user_correction"):
    return conn.execute(
        "SELECT polarity, weight, occurred_at, created_at, details "
        "FROM signal_events WHERE kind = ?",
        (kind,),
    ).fetchone()


def test_review_founder_correction_weighted_and_fold(tmp_path: Path):
    conn = _conn(tmp_path)
    cfg = _cfg(tmp_path)
    correction = "no, that's wrong, it should be the ready label instead of that"
    session = _session(
        _turn("assistant", "Here is the label plan you asked for and more."),
        _turn("user", correction, authored_by=FOUNDER),
    )
    review_session(conn, session, cfg)
    row = _correction_row(conn)
    # base 0.8 * founder weight 2.0
    assert abs(row["weight"] - 1.6) < 1e-9
    details = json.loads(row["details"])
    assert details["authored_by"] == FOUNDER
    assert details["author_weight"] == 2.0
    # The fold respects the weight: 1.6 >= hard_bad_weight -> hard bad at conf 0.95.
    label, conf, _s, _m = label_from_signals([row], cfg)
    assert label == "bad"
    assert conf == 0.95


def test_review_generic_correction_unweighted(tmp_path: Path):
    conn = _conn(tmp_path)
    cfg = _cfg(tmp_path)
    correction = "no, that's wrong, it should be the ready label instead of that"
    session = _session(
        _turn("assistant", "Here is the label plan you asked for and more."),
        _turn("user", correction, authored_by=STRANGER),  # generic member
    )
    review_session(conn, session, cfg)
    row = _correction_row(conn)
    assert abs(row["weight"] - 0.8) < 1e-9  # no founder scaling
    details = json.loads(row["details"])
    assert "authored_by" not in details  # generic authors add no provenance
    # A lone weight-0.8 bad signal is NOT hard-bad; its confidence is the ratio form.
    label, conf, _s, _m = label_from_signals([row], cfg)
    assert label == "bad"
    assert conf < 0.95


# --------------------------------------------------------------------------- #
# DPO: founder-issued correction pair carries author provenance
# --------------------------------------------------------------------------- #
A1 = "The answer is forty-two according to my first and frankly hasty guess about this."
CHOSEN = "Actually the deploy is driven by scripts/run.sh which you invoke before shipping."


def _dpo_session(corrector: str | None) -> Session:
    return _session(
        _turn("user", "please help with the deploy"),
        _turn("assistant", A1),
        _turn("user", "no, that's wrong, it should be the deploy script", authored_by=corrector),
        _turn("assistant", CHOSEN),
    )


def test_dpo_founder_correction_provenance(tmp_path: Path):
    cfg = _cfg(tmp_path)
    pairs = find_transcript_pairs(_dpo_session(FOUNDER), cfg)
    assert len(pairs) == 1
    assert pairs[0].chosen == CHOSEN  # corrected answer is the chosen side
    assert pairs[0].corrected_by == FOUNDER
    assert pairs[0].corrector_weight == 2.0
    # Stored example metadata is tagged with the founder provenance.
    conn = _conn(tmp_path)
    upsert_evidence(conn, source_type="openclaw_history_file", source_runtime="openclaw",
                    source_uri="/x/s.jsonl", content_hash="fp", claim="t",
                    privacy_scope="workspace")
    result = mine_dpo(conn, cfg=cfg, sessions=[_dpo_session(FOUNDER)], include_events=False)
    assert result["stored"] == 1
    record = json.loads(
        conn.execute("SELECT example_json FROM dataset_examples").fetchone()["example_json"]
    )
    meta = record["metadata"]
    assert meta["corrected_by"] == FOUNDER
    assert meta["founder_correction"] is True


def test_dpo_generic_correction_no_provenance(tmp_path: Path):
    cfg = _cfg(tmp_path)
    pairs = find_transcript_pairs(_dpo_session(STRANGER), cfg)
    assert len(pairs) == 1
    assert pairs[0].corrector_weight == 1.0


# --------------------------------------------------------------------------- #
# PERSONA ISOLATION (critical negative test)
# --------------------------------------------------------------------------- #
def test_persona_isolation_founder_never_becomes_voice(tmp_path: Path):
    cfg = _cfg(tmp_path)
    prompt = "The agent has finished the task and is awaiting your next instruction now."
    founder_msg = "Rename the button to Share a Bounty, that phrasing is warmer for neighbors."
    # A founder (identified non-persona sender) turn must never enter persona voice.
    founder_session = _session(_turn("assistant", prompt), _turn("user", founder_msg,
                                                                  authored_by=FOUNDER))
    assert telegram_examples(founder_session, cfg) == []
    # A stranger who is merely identified is likewise excluded.
    stranger_session = _session(_turn("assistant", prompt),
                                _turn("user", founder_msg, authored_by=STRANGER))
    assert telegram_examples(stranger_session, cfg) == []
    # The operator (persona author) IS admitted as voice from the identical shape.
    op_turn = Turn(role="user", text=founder_msg, kind="telegram_envelope",
                   authored_by=OPERATOR, sender_verified=True)
    op_session = _session(_turn("assistant", prompt), op_turn)
    assert len(telegram_examples(op_session, cfg)) == 1


def test_persona_isolation_end_to_end(tmp_path: Path):
    conn = _conn(tmp_path)
    cfg = _cfg(tmp_path)
    prompt = "The agent has finished the task and is awaiting your next instruction now."
    founder_msg = "Rename the button to Share a Bounty, that phrasing is warmer for neighbors."
    founder_session = _session(_turn("assistant", prompt),
                               _turn("user", founder_msg, authored_by=FOUNDER))
    # No repos -> only telegram voice mining; the founder turn must store nothing.
    result = mine_persona(conn, cfg=cfg, sessions=[founder_session], repos=[])
    assert result["stored"] == 0
    rows = conn.execute("SELECT COUNT(*) FROM dataset_examples WHERE dataset='persona'").fetchone()
    assert rows[0] == 0
