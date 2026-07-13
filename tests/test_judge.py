from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from ocbrain.config import load_config
from ocbrain.db import (
    connect,
    get_knowledge,
    init_db,
    log_retrieval_use,
    now_iso,
    upsert_knowledge,
)
from ocbrain_ops.autolabel import Signal, record_signal
from ocbrain_ops.judge import (
    build_judge_batch,
    eligible_rows,
    judge_ambiguous,
    spent_today,
)

KEY_ENV = {"OPENAI_API_KEY": "sk-test-not-a-real-key-000000000000"}


def _cfg(tmp_path: Path, **judge_overrides):
    base = load_config(tmp_path / "cfg.json")
    # Functional judge tests opt into the hosted lane explicitly; the product
    # default is fail-closed.
    settings = {"enabled": True, **judge_overrides}
    return replace(base, judge=replace(base.judge, **settings))


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


def _catalog_doc(conn: sqlite3.Connection, slug: str, *, origin: str = "autopilot") -> str:
    """A raw imported doc-kind catalog row, zero-signal, no retrieval."""
    return upsert_knowledge(
        conn, knowledge_type="doc", gate="auto", slug=slug,
        title=f"catalog {slug}", doc_kind="memory", status="current", origin=origin,
    )


def test_targeting_excludes_never_referenced_catalog_docs(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)  # default targeting: exclude_catalog_docs=True
    catalog = _catalog_doc(conn, "catalog-1")
    lesson = upsert_knowledge(
        conn, knowledge_type="value", gate="auto", subject="runtime",
        predicate="lesson", value_text="a distilled lesson", status="current",
        origin="harvest",
    )
    ids = {row["id"] for row in eligible_rows(conn, cfg)}
    assert catalog not in ids  # never-referenced catalog doc is drained no more
    assert lesson in ids  # review/lesson-derived value survives


def test_targeting_keeps_retrieval_touched_catalog_doc(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    touched = _catalog_doc(conn, "catalog-hot")
    log_retrieval_use(conn, touched, outcome="served")
    ids = {row["id"] for row in eligible_rows(conn, cfg)}
    assert touched in ids  # actually-served docs stay judgeable


def test_targeting_disabled_keeps_catalog(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path, targeting={"sources": [], "exclude_catalog_docs": False})
    catalog = _catalog_doc(conn, "catalog-2")
    ids = {row["id"] for row in eligible_rows(conn, cfg)}
    assert catalog in ids  # inert targeting keeps legacy behavior


def test_targeting_loop_origin_is_a_lesson(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    loop_doc = upsert_knowledge(
        conn, knowledge_type="doc", gate="auto", slug="loop-proc",
        title="loop procedure", doc_kind="procedure", status="current", origin="loop",
    )
    ids = {row["id"] for row in eligible_rows(conn, cfg)}
    assert loop_doc in ids  # loop-distilled docs are not catalog backlog


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


def test_hosted_call_never_holds_sqlite_writer_lock(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    _neutral_with_mass(conn, "network-window")

    def observing_call(payload, *, api_key, model):
        observer = sqlite3.connect(tmp_path / "ocbrain.sqlite", timeout=0)
        observer.execute("BEGIN IMMEDIATE")
        assert observer.execute("SELECT COUNT(*) FROM egress_audits").fetchone()[0] == 1
        observer.rollback()
        observer.close()
        return _stub_call()(payload, api_key=api_key, model=model)

    result = judge_ambiguous(conn, cfg, call=observing_call, env=KEY_ENV)
    assert result["changed"] >= 1


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


def test_fold_skips_row_deleted_between_selection_and_write(tmp_path: Path) -> None:
    # Ruling 2: the real light-run FK failure. A row selected for judging is
    # archived/deleted before the verdict is folded; signal_events.knowledge_id
    # FKs to knowledge(id) and INSERT OR IGNORE does NOT suppress FK violations,
    # so the naive fold aborts the run with 'FOREIGN KEY constraint failed'.
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    kid = _neutral_with_mass(conn, "vanishing")

    def deleting_call(payload, *, api_key, model):
        # Simulate a concurrent archive/delete landing between selection and fold.
        conn.execute("DELETE FROM signal_events WHERE knowledge_id=?", (kid,))
        conn.execute("DELETE FROM knowledge WHERE id=?", (kid,))
        items = json.loads(payload["messages"][1]["content"])
        verdicts = [
            {"id": item["id"], "label": "good", "confidence": 0.8, "rationale": "ok"}
            for item in items
        ]
        return {
            "choices": [{"message": {"content": json.dumps(verdicts)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

    result = judge_ambiguous(conn, cfg, call=deleting_call, env=KEY_ENV)
    # No crash; the run completes and records the skip.
    assert result["status"] == "ok"
    assert result["changed"] == 0
    assert result["skipped_missing"] >= 1
    # A judge_runs row with items>0 still lands (the batch WAS dispatched).
    run = conn.execute(
        "SELECT COUNT(*) FROM judge_runs WHERE status='ok' AND item_count>0"
    ).fetchone()[0]
    assert run >= 1
    # No dangling signal for the deleted row.
    assert conn.execute(
        "SELECT COUNT(*) FROM signal_events WHERE kind='llm_judge'"
    ).fetchone()[0] == 0


def test_fold_skips_hallucinated_verdict_id(tmp_path: Path) -> None:
    # The hosted model echoes an id we never sent (or a mangled one).
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    kid = _neutral_with_mass(conn, "real")

    def ghost_call(payload, *, api_key, model):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            [
                                {"id": kid, "label": "good", "confidence": 0.7},
                                {"id": "know-ghost-does-not-exist", "label": "bad",
                                 "confidence": 0.9},
                            ]
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

    result = judge_ambiguous(conn, cfg, call=ghost_call, env=KEY_ENV)
    assert result["status"] == "ok"
    assert result["changed"] == 1  # the real row folded
    assert result["skipped_missing"] == 1  # the ghost skipped, no crash
    folded = conn.execute(
        "SELECT knowledge_id FROM signal_events WHERE kind='llm_judge'"
    ).fetchall()
    assert [r["knowledge_id"] for r in folded] == [kid]


def test_call_retries_once_on_read_timeout(tmp_path: Path) -> None:
    # Ruling 2: heavy-run transient 'read operation timed out' -> one bounded retry.
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    kid = _neutral_with_mass(conn, "flaky")
    calls = {"n": 0}

    def flaky_call(payload, *, api_key, model):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("The read operation timed out")
        items = json.loads(payload["messages"][1]["content"])
        verdicts = [{"id": item["id"], "label": "good", "confidence": 0.8} for item in items]
        return {
            "choices": [{"message": {"content": json.dumps(verdicts)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

    result = judge_ambiguous(
        conn, cfg, call=flaky_call, env=KEY_ENV, sleep=lambda _s: None
    )
    assert calls["n"] == 2  # one retry
    assert result["status"] == "ok"
    assert result["changed"] >= 1
    signal = conn.execute(
        "SELECT knowledge_id FROM signal_events WHERE kind='llm_judge'"
    ).fetchone()
    assert signal["knowledge_id"] == kid


def test_call_reraises_persistent_timeout(tmp_path: Path) -> None:
    # A timeout that outlives the retry budget still propagates (autolabel guards it).
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    _neutral_with_mass(conn, "dead")

    def always_timeout(payload, *, api_key, model):
        raise TimeoutError("The read operation timed out")

    import pytest

    with pytest.raises(TimeoutError):
        judge_ambiguous(conn, cfg, call=always_timeout, env=KEY_ENV, sleep=lambda _s: None)


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
