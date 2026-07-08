from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from ocbrain.autolabel import Signal, record_signal
from ocbrain.config import load_config
from ocbrain.db import connect, get_knowledge, init_db, now_iso, upsert_knowledge
from ocbrain.judge import (
    build_judge_batch,
    eligible_rows,
    judge_ambiguous,
    spent_today,
)

KEY_ENV = {"OPENAI_API_KEY": "sk-test-not-a-real-key-000000000000"}


def _cfg(tmp_path: Path, **judge_overrides):
    base = load_config(tmp_path / "cfg.json")
    if judge_overrides:
        return replace(base, judge=replace(base.judge, **judge_overrides))
    return base


def _db(tmp_path: Path) -> sqlite3.Connection:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    return conn


def _neutral_with_mass(conn: sqlite3.Connection, predicate: str, scope: str = "workspace") -> str:
    kid = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime",
        predicate=predicate,
        value_text=f"claim {predicate}",
        status="current",
        privacy_scope=scope,
    )
    conn.execute("UPDATE knowledge SET quality_label='neutral' WHERE id=?", (kid,))
    now = datetime.now(UTC).isoformat()
    _sig(conn, "good", f"r:{predicate}:1", kid, now)
    _sig(conn, "bad", f"r:{predicate}:2", kid, now)
    return kid


def _sig(conn, polarity, ref, kid, occurred):
    record_signal(
        conn,
        Signal(
            "retrieval_feedback", polarity, 0.5, "retrieval", ref,
            knowledge_id=kid, occurred_at=occurred,
        ),
    )


def _stub_call(verdict_label: str = "good"):
    def call(payload, *, api_key, model):
        items = json.loads(payload["messages"][1]["content"])
        verdicts = [
            {"id": item["id"], "label": verdict_label, "confidence": 0.8, "rationale": "ok"}
            for item in items
        ]
        return {
            "choices": [{"message": {"content": json.dumps(verdicts)}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        }

    return call


def _boom_call(payload, *, api_key, model):  # pragma: no cover - must never run
    raise AssertionError("network call must not happen")


# --------------------------------------------------------------------------- #
def test_eligibility_neutral_mass_and_zero_signal(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    ambiguous = _neutral_with_mass(conn, "ambiguous")
    # Neutral but with too little mass -> excluded.
    weak = upsert_knowledge(
        conn, knowledge_type="value", gate="auto", subject="runtime",
        predicate="weak", value_text="v", status="current",
    )
    conn.execute("UPDATE knowledge SET quality_label='neutral' WHERE id=?", (weak,))
    record_signal(
        conn,
        Signal(
            "retrieval_feedback", "neutral", 0.1, "retrieval", "r:weak",
            knowledge_id=weak, occurred_at=datetime.now(UTC).isoformat(),
        ),
    )
    # Zero-signal candidate -> included.
    zero = upsert_knowledge(
        conn, knowledge_type="value", gate="auto", subject="runtime",
        predicate="zero", value_text="v", status="current",
    )
    ids = {row["id"] for row in eligible_rows(conn, cfg)}
    assert ambiguous in ids
    assert zero in ids
    assert weak not in ids


def test_private_scope_and_egress_drop(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    private = _neutral_with_mass(conn, "priv", scope="private")
    workspace = _neutral_with_mass(conn, "work", scope="workspace")
    rows = [get_knowledge(conn, private), get_knowledge(conn, workspace)]
    included, rejected = build_judge_batch(conn, rows)
    included_ids = {item["id"] for item in included}
    rejected_ids = {item["id"] for item in rejected}
    assert workspace in included_ids
    assert private in rejected_ids
    assert any(r["reason"] == "private_scope" for r in rejected)


def test_redaction_before_dispatch(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    kid = upsert_knowledge(
        conn, knowledge_type="value", gate="auto", subject="runtime",
        predicate="leak", value_text="the token is sk-abcdefghijklmnopqrstuvwxyz012345 keep it",
        status="current",
    )
    included, _rejected = build_judge_batch(conn, [get_knowledge(conn, kid)])
    assert included
    assert "sk-abcdefghijklmnop" not in included[0]["text"]
    assert "[REDACTED]" in included[0]["text"]


def test_daily_budget_skip(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    _neutral_with_mass(conn, "a")
    conn.execute(
        "INSERT INTO judge_runs (id, ts, model, status, cost_usd) VALUES ('j0', ?, ?, 'ok', 0.60)",
        (now_iso(), cfg.judge.model),
    )
    result = judge_ambiguous(conn, cfg, call=_boom_call, env=KEY_ENV)
    assert result["status"] == "skipped_budget"
    assert conn.execute(
        "SELECT COUNT(*) FROM judge_runs WHERE status='skipped_budget'"
    ).fetchone()[0] == 1


def test_inert_without_api_key(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    _neutral_with_mass(conn, "a")
    result = judge_ambiguous(conn, cfg, call=_boom_call, env={})
    assert result["status"] == "skipped"
    assert result["reason"] == "no_api_key"


def test_cost_accounting_and_budget(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(
        tmp_path,
        price_per_mtok={"gpt-5-mini": {"prompt": 0.15, "completion": 0.60}},
    )
    _neutral_with_mass(conn, "a")
    judge_ambiguous(conn, cfg, call=_stub_call(), env=KEY_ENV)
    # 1000 prompt @ 0.15/M + 500 completion @ 0.60/M = 0.00015 + 0.0003 = 0.00045
    spent = spent_today(conn)
    assert abs(spent - 0.00045) < 1e-9


def test_verdict_folds_to_signal(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    kid = _neutral_with_mass(conn, "a")
    result = judge_ambiguous(conn, cfg, call=_stub_call("good"), env=KEY_ENV)
    assert result["changed"] >= 1
    signal = conn.execute(
        "SELECT polarity, weight, knowledge_id FROM signal_events WHERE kind='llm_judge'"
    ).fetchone()
    assert signal is not None
    assert signal["polarity"] == "good"
    assert signal["weight"] == cfg.judge.signal_weight
    assert signal["knowledge_id"] == kid


def test_response_stores_verdicts_only_and_audit(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    _neutral_with_mass(conn, "secretclaim")
    judge_ambiguous(conn, cfg, call=_stub_call(), env=KEY_ENV)
    run = conn.execute(
        "SELECT response_json, egress_audit_id FROM judge_runs WHERE status='ok' AND item_count>0"
    ).fetchone()
    assert run is not None
    stored = json.loads(run["response_json"])
    # Stored payload is verdicts only: id/label/confidence, never the dispatched body text.
    assert all(set(v.keys()) <= {"id", "label", "confidence", "rationale"} for v in stored)
    assert run["egress_audit_id"] is not None
    assert conn.execute("SELECT COUNT(*) FROM egress_audits").fetchone()[0] >= 1
