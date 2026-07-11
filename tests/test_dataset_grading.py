from __future__ import annotations

import dataclasses
import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ocbrain.config import DatasetGradingConfig, OcbrainConfig
from ocbrain.dataset.export import export_all
from ocbrain.dataset.grade import (
    DATASET_RUBRICS,
    MAX_GRADE_CONTEXT_CHARS,
    _messages,
    grade_examples,
    normalize_grade,
    require_loopback_endpoint,
)
from ocbrain.dataset.quality import store_example
from ocbrain.db import connect, init_db


def _db(tmp_path: Path):
    conn = connect(tmp_path / "db.sqlite")
    init_db(conn)
    return conn


def _cfg(*, per_run: int = 100, daily: int = 500, parallel: int = 1) -> OcbrainConfig:
    base = OcbrainConfig()
    return dataclasses.replace(
        base,
        dataset_grading=DatasetGradingConfig(
            endpoint="http://127.0.0.1:11434/api/chat",
            model="local-test-model",
            per_run_item_cap=per_run,
            daily_item_cap=daily,
            parallel_requests=parallel,
        ),
    )


def _store(conn, dataset: str, index: int):
    target = (
        f"Example {index} for {dataset} gives a concrete, careful answer with enough "
        "substance to clear every quality length rule without relying on vague filler."
    )
    if dataset == "dpo":
        body = {
            "input": {"messages": [{"role": "user", "content": f"question {index}"}]},
            "preferred_output": [{"role": "assistant", "content": target}],
            "non_preferred_output": [
                {"role": "assistant", "content": "A generic and incorrect rejected answer."}
            ],
        }
    else:
        body = {
            "messages": [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": target},
            ]
        }
    return store_example(
        conn,
        dataset=dataset,
        source_kind="correction_event" if dataset == "dpo" else "codex_session",
        source_uri=f"/local/{dataset}-{index}.jsonl",
        evidence_ids=[f"evd_{dataset}_{index}"],
        privacy_scope="workspace",
        body=body,
        metadata={"session_id": f"s-{index}", "private_marker": "not-for-prompt"},
        target_text=target,
        base_label="good",
        base_confidence=0.9,
        occurred_at=f"2026-07-{index + 1:02d}T00:00:00Z",
    )


def _transport(endpoint, model, messages, timeout):
    assert endpoint.startswith("http://127.0.0.1:")
    assert model == "local-test-model"
    assert timeout > 0
    request = json.loads(messages[1]["content"])
    assert "metadata" not in request["example"]
    dataset = request["dataset"]
    return {
        "overall_score": 0.86,
        "dimensions": {name: 0.84 for name in DATASET_RUBRICS[dataset]},
        "verdict": "pass",
        "flags": [],
        "explanation": "Strong local-only training example.",
    }


def test_loopback_boundary_rejects_remote_endpoints():
    assert require_loopback_endpoint("http://localhost:11434/api/chat")
    assert require_loopback_endpoint("http://[::1]:11434/api/chat")
    with pytest.raises(ValueError, match="loopback-only"):
        require_loopback_endpoint("https://grader.example.com/v1/chat")
    with pytest.raises(ValueError, match="credentials"):
        require_loopback_endpoint("http://user:secret@localhost:11434/api/chat")


def test_normalize_grade_accepts_validated_flat_local_model_dimensions():
    raw = {
        "overall_score": 0.82,
        **{name: 0.8 for name in DATASET_RUBRICS["dpo"]},
        "verdict": "pass",
        "flags": [],
        "explanation": "Valid flat structured response.",
    }
    grade = normalize_grade("dpo", raw)
    assert grade["overall_score"] == 0.82
    assert grade["dimensions"] == {name: 0.8 for name in DATASET_RUBRICS["dpo"]}


def test_grade_view_bounds_old_context_but_preserves_target():
    target = "target response " * 100
    record = {
        "messages": [
            {"role": "user", "content": "old context " * 1000},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": target},
        ],
        "metadata": {"private": "not included"},
    }
    request = json.loads(_messages("sft", record)[1]["content"])
    example = request["example"]
    assert example["messages"][-1]["content"] == target
    assert sum(len(item["content"]) for item in example["messages"][:-1]) <= (
        MAX_GRADE_CONTEXT_CHARS
    )
    assert "metadata" not in example


def test_grade_persists_metadata_and_is_idempotent(tmp_path: Path):
    conn = _db(tmp_path)
    for i, dataset in enumerate(("sft", "dpo", "persona")):
        _store(conn, dataset, i)
    cfg = _cfg()
    now = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)

    result = grade_examples(conn, cfg=cfg, transport=_transport, now=now)
    conn.commit()
    assert result["graded"] == 3 and result["errors"] == 0
    assert result["local_only"] is True
    rows = conn.execute(
        "SELECT grade_score, grade_model, grade_prompt_version, example_json "
        "FROM dataset_examples ORDER BY dataset"
    ).fetchall()
    assert all(row["grade_score"] == 0.86 for row in rows)
    assert all(row["grade_model"] == "local-test-model" for row in rows)
    assert all(
        json.loads(row["example_json"])["metadata"]["llm_grade"]["local_only"] for row in rows
    )

    again = grade_examples(conn, cfg=cfg, transport=_transport, now=now)
    assert again["skipped"] == "no_candidates"


def test_parallel_local_inference_keeps_one_db_writer(tmp_path: Path):
    conn = _db(tmp_path)
    for index in range(4):
        _store(conn, "sft", index)
    conn.commit()
    state = {"active": 0, "maximum": 0}
    lock = threading.Lock()

    def concurrent_transport(endpoint, model, messages, timeout):
        with lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        time.sleep(0.02)
        with lock:
            state["active"] -= 1
        return _transport(endpoint, model, messages, timeout)

    result = grade_examples(
        conn,
        cfg=_cfg(parallel=2),
        transport=concurrent_transport,
    )
    assert result["graded"] == 4
    assert result["parallel_requests"] == 2
    assert state["maximum"] == 2
    assert (
        conn.execute("SELECT COUNT(*) FROM dataset_examples WHERE grade_score = 0.86").fetchone()[0]
        == 4
    )
    assert conn.execute("SELECT COUNT(*) FROM dataset_grade_runs").fetchone()[0] == 1


def test_grade_can_target_a_private_curation_source_prefix(tmp_path: Path):
    conn = _db(tmp_path)
    wanted = _store(conn, "persona", 1)
    _store(conn, "persona", 2)
    conn.execute(
        "UPDATE dataset_examples SET source_uri = ? WHERE id = ?",
        ("curation://pack-a/1", wanted["id"]),
    )
    conn.commit()

    result = grade_examples(
        conn,
        cfg=_cfg(),
        transport=_transport,
        source_uri_prefix="curation://pack-a/",
    )
    assert result["graded"] == 1
    scores = conn.execute("SELECT id, grade_score FROM dataset_examples ORDER BY id").fetchall()
    assert {row["id"] for row in scores if row["grade_score"] is not None} == {wanted["id"]}


def test_grade_caps_count_attempts_not_only_successes(tmp_path: Path):
    conn = _db(tmp_path)
    for i in range(4):
        _store(conn, "sft", i)
    cfg = _cfg(per_run=2, daily=2)

    def fail(*args):
        raise RuntimeError("local model failed")

    first = grade_examples(
        conn,
        cfg=cfg,
        transport=fail,
        now=datetime(2026, 7, 9, tzinfo=UTC),
    )
    conn.commit()
    assert first["graded"] == 0 and first["errors"] == 2
    run = conn.execute("SELECT item_count, error_count FROM dataset_grade_runs").fetchone()
    assert run["item_count"] == 2 and run["error_count"] == 2
    second = grade_examples(
        conn,
        cfg=cfg,
        transport=_transport,
        now=datetime(2026, 7, 9, 1, tzinfo=UTC),
    )
    assert second["skipped"] == "item_cap"


def test_grade_releases_write_lock_between_local_calls(tmp_path: Path):
    conn = _db(tmp_path)
    _store(conn, "sft", 1)
    _store(conn, "sft", 2)
    conn.commit()
    calls = 0

    def observing_transport(endpoint, model, messages, timeout):
        nonlocal calls
        calls += 1
        if calls == 2:
            observer = connect(tmp_path / "db.sqlite")
            assert (
                observer.execute(
                    "SELECT COUNT(*) FROM dataset_examples WHERE grade_score IS NOT NULL"
                ).fetchone()[0]
                == 1
            )
            run = observer.execute("SELECT status, item_count FROM dataset_grade_runs").fetchone()
            assert run["status"] == "running" and run["item_count"] == 1
            observer.close()
        return _transport(endpoint, model, messages, timeout)

    result = grade_examples(conn, cfg=_cfg(), transport=observing_transport)
    assert result["graded"] == 2


def test_failed_example_is_skipped_until_forced(tmp_path: Path):
    conn = _db(tmp_path)
    first = _store(conn, "sft", 1)
    second = _store(conn, "sft", 2)
    conn.commit()

    def fail(*args):
        raise ValueError("bad local response")

    failed = grade_examples(conn, cfg=_cfg(), limit=1, transport=fail)
    assert failed["status"] == "error"
    row = conn.execute(
        "SELECT grade_score, grade_model, grade_json FROM dataset_examples WHERE id = ?",
        (first["id"],),
    ).fetchone()
    assert row["grade_score"] is None and row["grade_model"] == "local-test-model"
    assert json.loads(row["grade_json"])["error_type"] == "ValueError"

    advanced = grade_examples(conn, cfg=_cfg(), limit=1, transport=_transport)
    assert advanced["graded"] == 1
    assert (
        conn.execute(
            "SELECT grade_score FROM dataset_examples WHERE id = ?", (second["id"],)
        ).fetchone()[0]
        == 0.86
    )

    retried = grade_examples(conn, cfg=_cfg(), limit=1, force=True, transport=_transport)
    assert retried["graded"] == 1


def test_sqlite_infrastructure_error_remains_retryable(tmp_path: Path):
    conn = _db(tmp_path)
    example = _store(conn, "sft", 1)
    conn.commit()

    def locked(*args):
        import sqlite3

        raise sqlite3.OperationalError("database is locked")

    failed = grade_examples(conn, cfg=_cfg(), limit=1, transport=locked)
    assert failed["status"] == "error"
    row = conn.execute(
        "SELECT grade_model, grade_json FROM dataset_examples WHERE id = ?",
        (example["id"],),
    ).fetchone()
    assert row["grade_model"] is None and row["grade_json"] is None

    retried = grade_examples(conn, cfg=_cfg(), limit=1, transport=_transport)
    assert retried["graded"] == 1


def test_progress_lock_returns_blocked_and_next_run_repairs(tmp_path: Path):
    import sqlite3

    conn = _db(tmp_path)
    _store(conn, "sft", 1)
    conn.commit()
    conn.execute("PRAGMA busy_timeout=1")
    locker = None

    def lock_during_inference(endpoint, model, messages, timeout):
        nonlocal locker
        locker = sqlite3.connect(tmp_path / "db.sqlite")
        locker.execute("BEGIN IMMEDIATE")
        return _transport(endpoint, model, messages, timeout)

    blocked = grade_examples(conn, cfg=_cfg(), limit=1, transport=lock_during_inference)
    assert blocked["status"] == "blocked"
    assert blocked["ledger_pending"] is True
    assert locker is not None
    locker.rollback()
    locker.close()

    repaired = grade_examples(conn, cfg=_cfg(), limit=1, transport=_transport)
    assert repaired["repaired_runs"] == 1
    assert repaired["graded"] == 1
    statuses = {row[0] for row in conn.execute("SELECT status FROM dataset_grade_runs")}
    assert statuses == {"interrupted", "ok"}


def test_export_min_grade_withholds_low_and_ungraded_rows(tmp_path: Path):
    conn = _db(tmp_path)
    high = _store(conn, "sft", 1)
    low = _store(conn, "sft", 2)
    _store(conn, "sft", 3)  # ungraded
    conn.execute("UPDATE dataset_examples SET grade_score = 0.91 WHERE id = ?", (high["id"],))
    conn.execute("UPDATE dataset_examples SET grade_score = 0.62 WHERE id = ?", (low["id"],))
    result = export_all(
        conn,
        cfg=OcbrainConfig(),
        datasets=["sft"],
        min_grade=0.8,
        export_dir=tmp_path / "out",
    )
    assert result["manifest"]["min_grade"] == 0.8
    assert result["datasets"]["sft"]["count"] == 1
    assert result["manifest"]["datasets"]["sft"]["graded_count"] == 2
