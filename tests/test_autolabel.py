from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ocbrain.config import load_config
from ocbrain.db import (
    connect,
    get_knowledge,
    init_db,
    link_knowledge_evidence,
    log_retrieval_use,
    update_retrieval_use_feedback,
    upsert_evidence,
    upsert_knowledge,
)
from ocbrain.events import append_event
from ocbrain.ids import content_hash
from ocbrain.write_batch import DatasetWriteBatch
from ocbrain_ops.autolabel import (
    Signal,
    attribute_signals,
    autolabel,
    fold_labels,
    get_watermark,
    label_from_signals,
    mine_commitments,
    mine_cron_state,
    mine_event_signals,
    mine_learning_db,
    mine_retrieval_signals,
    record_signal,
)


def _cfg(tmp_path: Path):
    return load_config(tmp_path / "cfg.json")


def _db(tmp_path: Path) -> sqlite3.Connection:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    return conn


def _kid(conn: sqlite3.Connection, predicate: str = "p", value: str = "v") -> str:
    return upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime",
        predicate=predicate,
        value_text=value,
        status="current",
    )


def _rec(conn, kind, polarity, weight, kid, ref, source="retrieval", occurred=None):
    record_signal(
        conn,
        Signal(kind, polarity, weight, source, ref, knowledge_id=kid, occurred_at=occurred),
    )


def _count_signals(conn: sqlite3.Connection, kind: str | None = None) -> int:
    if kind:
        return conn.execute(
            "SELECT COUNT(*) FROM signal_events WHERE kind = ?", (kind,)
        ).fetchone()[0]
    return conn.execute("SELECT COUNT(*) FROM signal_events").fetchone()[0]


# --------------------------------------------------------------------------- #
# Miners
# --------------------------------------------------------------------------- #
def test_mine_retrieval_signals_maps_outcomes(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    kid = _kid(conn)
    for outcome in ("improved", "harmful", "served"):
        rid = log_retrieval_use(conn, kid, outcome="served")
        update_retrieval_use_feedback(conn, rid, outcome=outcome)
    result = mine_retrieval_signals(conn)
    # 'served' is not a labeling outcome -> only improved + harmful emit.
    assert result["changed"] == 2
    assert _count_signals(conn, "retrieval_feedback") == 2
    polarities = {
        r["polarity"]
        for r in conn.execute("SELECT polarity FROM signal_events WHERE kind='retrieval_feedback'")
    }
    assert polarities == {"good", "bad"}


def test_mine_retrieval_idempotent_and_watermark(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    kid = _kid(conn)
    rid = log_retrieval_use(conn, kid, outcome="served")
    update_retrieval_use_feedback(conn, rid, outcome="helpful")
    first = mine_retrieval_signals(conn)
    second = mine_retrieval_signals(conn)
    assert first["changed"] == 1
    assert second["changed"] == 0
    assert _count_signals(conn, "retrieval_feedback") == 1
    assert get_watermark(conn, "autolabel", "retrieval_uses") is not None


def test_mine_event_signals_hard_correction(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    kid = _kid(conn)
    append_event(
        conn,
        "correction_recorded",
        {
            "target_layer": "knowledge",
            "target_id": kid,
            "op": "mark_wrong",
            "body": None,
            "author": "human:jonathan",
            "hard": True,
        },
        writer="human:jonathan",
    )
    result = mine_event_signals(conn)
    assert result["changed"] == 1
    row = conn.execute(
        "SELECT polarity, weight, knowledge_id FROM signal_events "
        "WHERE kind='hard_correction_event'"
    ).fetchone()
    assert row["polarity"] == "bad"
    assert row["weight"] == 1.0
    assert row["knowledge_id"] == kid
    # Idempotent second pass.
    assert mine_event_signals(conn)["changed"] == 0


def test_mine_event_signals_verifier_result(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    kid = _kid(conn)
    evd = upsert_evidence(
        conn,
        source_type="test",
        claim="tests pass",
        content_hash=content_hash("tests pass"),
        source_uri="file://x",
        verifier_status="passed",
    )
    link_knowledge_evidence(conn, kid, evd)
    result = mine_event_signals(conn)
    assert result["changed"] == 1
    row = conn.execute(
        "SELECT polarity, knowledge_id FROM signal_events WHERE kind='verifier_result'"
    ).fetchone()
    assert row["polarity"] == "good"
    assert row["knowledge_id"] == kid


def test_mine_learning_db(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    learning_path = tmp_path / "learning.db"
    ext = sqlite3.connect(learning_path)
    ext.executescript(
        """
        CREATE TABLE learnings (
          id INTEGER PRIMARY KEY, rule_type TEXT, status TEXT,
          confidence REAL, recurrence INTEGER, prevention_rule TEXT
        );
        CREATE TABLE gate_violations (id INTEGER PRIMARY KEY, content_snippet TEXT);
        INSERT INTO learnings VALUES (1,'GATE','active',0.8,3,'never git push without approval');
        INSERT INTO learnings VALUES (2,'NOTE','active',0.5,1,'informational only');
        INSERT INTO gate_violations VALUES (1,'ran git push on main');
        """
    )
    ext.commit()
    ext.close()

    result = mine_learning_db(conn, learning_db_path=str(learning_path))
    # 1 active GATE learning + 1 gate violation (the NOTE row is not GATE/CORRECTION).
    assert result["changed"] == 2
    assert _count_signals(conn, "learning_gate_rule") == 1
    assert _count_signals(conn, "gate_violation") == 1
    # The prevention rule also becomes a prescriptive harvested candidate.
    prescriptive = conn.execute(
        "SELECT COUNT(*) FROM knowledge WHERE prescriptive=1 AND origin='harvest'"
    ).fetchone()[0]
    assert prescriptive == 1
    # Idempotent.
    assert mine_learning_db(conn, learning_db_path=str(learning_path))["changed"] == 0


def test_mine_learning_db_missing_path_is_noop(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    result = mine_learning_db(conn, learning_db_path=str(tmp_path / "nope.db"))
    assert result["changed"] == 0
    assert _count_signals(conn) == 0


def test_mine_commitments(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    path = tmp_path / "commitments.json"
    path.write_text(
        '[{"id":"c1","status":"completed"},'
        '{"id":"c2","status":"missed"},'
        '{"id":"c3","status":"pending"}]',
        encoding="utf-8",
    )
    result = mine_commitments(conn, commitments_path=str(path))
    assert result["changed"] == 2
    assert _count_signals(conn, "commitment_outcome") == 2


def test_mine_cron_state(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    path = tmp_path / "jobs-state.json"
    path.write_text(
        '{"jobs":{"j1":{"status":"ok","delivered":true,"updatedAtMs":100},'
        '"j2":{"status":"error","updatedAtMs":200}}}',
        encoding="utf-8",
    )
    result = mine_cron_state(conn, cron_state_path=str(path))
    assert result["changed"] == 2
    kinds = {
        (r["polarity"])
        for r in conn.execute("SELECT polarity FROM signal_events WHERE kind='cron_run'")
    }
    assert kinds == {"good", "bad"}


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #
def test_attribute_signals_via_fts(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    kid = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="deployment",
        predicate="rollback",
        value_text="kubernetes deployment rollback procedure documented",
        status="current",
    )
    record_signal(
        conn,
        Signal(
            kind="learning_gate_rule",
            polarity="bad",
            weight=0.6,
            source="learning_db",
            source_ref="learnings:9",
            details={"content": "kubernetes deployment rollback procedure"},
        ),
    )
    attributed = attribute_signals(conn)
    assert kid in attributed
    row = conn.execute(
        "SELECT knowledge_id FROM signal_events WHERE kind='learning_gate_rule'"
    ).fetchone()
    assert row["knowledge_id"] == kid


def test_attribute_signals_skips_session_signals(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    _kid(conn)
    record_signal(
        conn,
        Signal(
            kind="user_correction",
            polarity="bad",
            weight=0.8,
            source="session",
            source_ref="path#1",
            session_key="s1",
            details={"snippet": "runtime fact"},
        ),
    )
    attributed = attribute_signals(conn)
    assert attributed == set()
    row = conn.execute(
        "SELECT knowledge_id FROM signal_events WHERE kind='user_correction'"
    ).fetchone()
    assert row["knowledge_id"] is None


def test_attribute_signals_honors_zero_time_budget(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    _kid(conn)
    record_signal(
        conn,
        Signal(
            kind="learning_gate_rule",
            polarity="bad",
            weight=0.6,
            source="learning_db",
            source_ref="budget:0",
            details={"content": "runtime fact"},
        ),
    )
    assert attribute_signals(conn, time_budget_seconds=0) == set()
    assert conn.execute(
        "SELECT knowledge_id FROM signal_events WHERE source_ref='budget:0'"
    ).fetchone()[0] is None


def test_attribute_signals_releases_writer_before_next_fts_query(
    tmp_path: Path, monkeypatch
) -> None:
    conn = _db(tmp_path)
    kid = _kid(conn, predicate="rollback", value="documented rollback procedure")
    for index in range(2):
        record_signal(
            conn,
            Signal(
                kind="learning_gate_rule",
                polarity="bad",
                weight=0.6,
                source="learning_db",
                source_ref=f"learning:{index}",
                details={"content": "documented rollback procedure"},
            ),
        )
    conn.commit()
    calls = 0

    def observed_search(_conn, text, limit):
        nonlocal calls
        calls += 1
        if calls == 2:
            observer = sqlite3.connect(tmp_path / "ocbrain.sqlite", timeout=0)
            observer.execute("BEGIN IMMEDIATE")
            observer.rollback()
            observer.close()
        return [{"doc_id": kid, "title": text, "snippet": text}]

    monkeypatch.setattr("ocbrain_ops.autolabel.search", observed_search)
    batch = DatasetWriteBatch(conn, max_operations=1, max_seconds=60)
    attributed = attribute_signals(conn, write_batch=batch)
    assert attributed == {kid}
    assert calls == 2
    assert batch.metrics()["batches_committed"] == 2


# --------------------------------------------------------------------------- #
# Label fold
# --------------------------------------------------------------------------- #
def _sig_dict(polarity: str, now: datetime, weight: float = 0.5) -> dict:
    return {
        "polarity": polarity,
        "weight": weight,
        "occurred_at": now.isoformat(),
        "created_at": None,
    }


def _good_signal(now: datetime, weight: float = 0.5) -> dict:
    return _sig_dict("good", now, weight)


def _bad_signal(now: datetime, weight: float = 0.5) -> dict:
    return _sig_dict("bad", now, weight)


def test_label_from_signals_good_bad_neutral(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    now = datetime.now(UTC)
    label, conf, _s, _m = label_from_signals([_good_signal(now), _good_signal(now)], cfg, now=now)
    assert label == "good"
    assert conf > 0
    label, _c, _s, _m = label_from_signals([_good_signal(now), _bad_signal(now)], cfg, now=now)
    assert label == "neutral"
    label, _c, _s, _m = label_from_signals([_bad_signal(now), _bad_signal(now)], cfg, now=now)
    assert label == "bad"


def test_label_mass_gate_forces_neutral(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    now = datetime.now(UTC)
    # Single small-weight good signal: ratio is 1.0 but mass (0.4) < min_mass (0.6).
    label, _c, _s, mass = label_from_signals([_good_signal(now, 0.4)], cfg, now=now)
    assert mass < cfg.labels.min_mass
    assert label == "neutral"


def test_hard_bad_precedence_over_judge(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    now = datetime.now(UTC)
    signals = [
        _good_signal(now),
        _good_signal(now),
        {"polarity": "bad", "weight": 0.9, "occurred_at": now.isoformat(), "created_at": None},
    ]
    label, conf, _s, _m = label_from_signals(signals, cfg, now=now)
    assert label == "bad"
    assert conf == 0.9


def test_label_decay_reduces_old_signal(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    now = datetime.now(UTC)
    old = now - timedelta(days=90)  # 3 half-lives
    fresh = [_good_signal(now)]
    stale = [_sig_dict("good", old)]
    _l, _c, _s_fresh, m_fresh = label_from_signals(fresh, cfg, now=now)
    _l, _c, _s_stale, m_stale = label_from_signals(stale, cfg, now=now)
    assert m_stale < m_fresh


def test_fold_labels_writes_and_advances_watermark(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    now = datetime.now(UTC)
    kid = _kid(conn)
    _rec(conn, "retrieval_feedback", "good", 0.5, kid, "r:1", occurred=now.isoformat())
    _rec(conn, "retrieval_feedback", "good", 0.5, kid, "r:2", occurred=now.isoformat())
    result = fold_labels(conn, cfg, now=now)
    assert result["changed"] == 1
    assert get_knowledge(conn, kid)["quality_label"] == "good"
    # Second fold with no new signals is a no-op.
    assert fold_labels(conn, cfg, now=now)["changed"] == 0


def test_fold_bad_flip_demotes_injected(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    now = datetime.now(UTC)
    kid = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime",
        predicate="fact",
        value_text="use the shared brain",
        status="current",
        inject=True,
    )
    assert get_knowledge(conn, kid)["inject"] == 1
    _rec(conn, "hard_correction_event", "bad", 0.9, kid, "e:1", source="events")
    fold_labels(conn, cfg, now=now)
    row = get_knowledge(conn, kid)
    assert row["quality_label"] == "bad"
    assert row["inject"] == 0


def test_fold_never_demotes_human_row(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    now = datetime.now(UTC)
    kid = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime",
        predicate="fact",
        value_text="human curated fact",
        status="current",
        inject=True,
        origin="human",
        actor="human:jonathan",
    )
    _rec(conn, "hard_correction_event", "bad", 0.9, kid, "e:1", source="events")
    fold_labels(conn, cfg, now=now)
    row = get_knowledge(conn, kid)
    assert row["quality_label"] == "bad"  # labels still inform the dataset
    assert row["inject"] == 1  # but a human row is never auto-demoted


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #
def test_autolabel_orchestration_no_judge(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    base = _cfg(tmp_path)
    cfg = replace(
        base,
        dataset=replace(
            base.dataset,
            learning_db=str(tmp_path / "missing.db"),
            commitments_path=str(tmp_path / "missing.json"),
            cron_state_path=str(tmp_path / "missing-cron.json"),
        ),
    )
    kid = _kid(conn)
    rid = log_retrieval_use(conn, kid, outcome="served")
    update_retrieval_use_feedback(conn, rid, outcome="improved")
    result = autolabel(conn, cfg, run_judge=False)
    assert result["action"] == "autolabel"
    assert "retrieval" in result["stages"]
    assert result["stages"]["retrieval"]["changed"] == 1
