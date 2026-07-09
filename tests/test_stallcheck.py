"""Synthetic-fixture tests for the stall checker and the mcp lock-retry patch.

No live DB, no real transcripts, no real Telegram: every input is planted in a
tmp dir / tmp sqlite, and the pager is monkeypatched. Covers Reader A (workflow
passive-wait + task .output), Readers B/C (runner sqlite), the deadman-engine
writes, dedup (second run pages nothing), the self-heartbeat row, and the mcp
'database is locked' bounded retry.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from ocbrain import stallcheck
from ocbrain.db import connect, init_db


# --- fixture builders ----------------------------------------------------------
def _age(path, minutes: float) -> None:
    ts = (datetime.now(UTC) - timedelta(minutes=minutes)).timestamp()
    os.utime(path, (ts, ts))


def _write_agent_jsonl(path, *, last_text: str, stop_reason: str = "end_turn") -> None:
    records = [
        {"type": "user", "message": {"role": "user", "content": "do the thing"}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "stop_reason": stop_reason,
                "content": [{"type": "text", "text": last_text}],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _cfg(**overrides) -> stallcheck.StallCheckConfig:
    base = dict(
        workflow_globs=(),
        task_output_globs=(),
        runner_db="/nonexistent",
        stale_threshold_minutes=20,
        terminal_backlog_hours=48,
        ingress_window_hours=720,
        flag_zero_byte_output=True,
    )
    base.update(overrides)
    return stallcheck.StallCheckConfig(**base)


def _make_runner_db(path, *, task_rows=(), ingress_rows=()) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE task_runs (
          task_id TEXT PRIMARY KEY, runtime TEXT, status TEXT NOT NULL,
          label TEXT, task TEXT, last_event_at INTEGER, error TEXT,
          progress_summary TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE channel_ingress_events (
          queue_name TEXT, event_id TEXT, channel_id TEXT, account_id TEXT,
          status TEXT NOT NULL, failed_reason TEXT, failed_at INTEGER,
          last_error TEXT, PRIMARY KEY (queue_name, event_id)
        )
        """
    )
    conn.executemany(
        "INSERT INTO task_runs (task_id, runtime, status, label, task, last_event_at, "
        "error, progress_summary) VALUES (?,?,?,?,?,?,?,?)",
        task_rows,
    )
    conn.executemany(
        "INSERT INTO channel_ingress_events (queue_name, event_id, channel_id, "
        "account_id, status, failed_reason, failed_at, last_error) VALUES (?,?,?,?,?,?,?,?)",
        ingress_rows,
    )
    conn.commit()
    conn.close()


def _brain(tmp_path):
    conn = connect(tmp_path / "brain.sqlite")
    init_db(conn)
    return conn


# --- Reader A: workflow passive-wait ------------------------------------------
def test_reader_a_flags_aged_passive_wait_transcript(tmp_path):
    wf = tmp_path / "wf_abc123"
    wf.mkdir()
    agent = wf / "agent-deadbeef01.jsonl"
    _write_agent_jsonl(agent, last_text="All wired up. Standing by until the monitor notifies me.")
    _age(agent, minutes=90)

    cfg = _cfg(workflow_globs=(f"{tmp_path}/wf_*/",))
    findings = stallcheck.scan_workflow_stalls(cfg, datetime.now(UTC))

    assert len(findings) == 1
    f = findings[0]
    assert f.stall_class == "workflow_passive_wait"
    assert f.unit_id == "deadbeef01"
    assert "standing by" in f.terminal_signature
    assert f.age_seconds > 20 * 60


def test_reader_a_ignores_fresh_transcript(tmp_path):
    wf = tmp_path / "wf_fresh"
    wf.mkdir()
    agent = wf / "agent-fresh01.jsonl"
    _write_agent_jsonl(agent, last_text="Standing by, waiting for the monitor.")
    _age(agent, minutes=2)  # still fresh -> being appended

    cfg = _cfg(workflow_globs=(f"{tmp_path}/wf_*/",))
    assert stallcheck.scan_workflow_stalls(cfg, datetime.now(UTC)) == []


def test_reader_a_ignores_non_passive_end_turn(tmp_path):
    wf = tmp_path / "wf_done"
    wf.mkdir()
    agent = wf / "agent-done01.jsonl"
    _write_agent_jsonl(agent, last_text="Completed and verified: all tests pass. Done.")
    _age(agent, minutes=90)

    cfg = _cfg(workflow_globs=(f"{tmp_path}/wf_*/",))
    assert stallcheck.scan_workflow_stalls(cfg, datetime.now(UTC)) == []


def test_reader_a_journal_recent_result_suppresses(tmp_path):
    wf = tmp_path / "wf_alive"
    wf.mkdir()
    agent = wf / "agent-alive01.jsonl"
    _write_agent_jsonl(agent, last_text="Standing by for the monitor.")
    _age(agent, minutes=90)
    journal = wf / "journal.jsonl"
    journal.write_text(json.dumps({"type": "result", "agentId": "other"}) + "\n")
    # journal is fresh -> workflow is still producing results -> not a stall.

    cfg = _cfg(workflow_globs=(f"{tmp_path}/wf_*/",))
    assert stallcheck.scan_workflow_stalls(cfg, datetime.now(UTC)) == []


# --- Reader A: task .output ----------------------------------------------------
def test_reader_a_task_output_zero_byte_and_start_no_exit(tmp_path):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    empty = tasks / "aaa.output"
    empty.touch()
    _age(empty, minutes=90)
    open_task = tasks / "bbb.output"
    open_task.write_text("start: began work\nrunning step 1\n")
    _age(open_task, minutes=90)
    closed = tasks / "ccc.output"
    closed.write_text("start: began\nexit: 0 ok\n")
    _age(closed, minutes=90)

    cfg = _cfg(task_output_globs=(f"{tmp_path}/tasks/",))
    findings = stallcheck.scan_task_output_stalls(cfg, datetime.now(UTC))
    sigs = {f.unit_id: f.terminal_signature for f in findings}
    assert sigs == {"aaa": "zero_byte", "bbb": "start_no_exit"}  # closed is excluded

    # With the zero-byte clause off, only the precise start-without-exit fires.
    cfg_off = _cfg(task_output_globs=(f"{tmp_path}/tasks/",), flag_zero_byte_output=False)
    off = stallcheck.scan_task_output_stalls(cfg_off, datetime.now(UTC))
    assert {f.unit_id: f.terminal_signature for f in off} == {"bbb": "start_no_exit"}


# --- Reader B: runner task_runs -----------------------------------------------
def test_reader_b_lost_and_stale(tmp_path):
    db = tmp_path / "runner.sqlite"
    now = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)
    now_ms = int(now.timestamp() * 1000)
    _make_runner_db(
        db,
        task_rows=[
            ("t-lost", "claude", "lost", "lost lane", "task a", now_ms - 3600_000, None, None),
            ("t-stale", "claude", "running", "stuck", "task b", now_ms - 3600_000, None, "step 3"),
            ("t-fresh", "claude", "running", "ok", "task c", now_ms - 60_000, None, None),
            ("t-done", "claude", "succeeded", "done", "task d", now_ms - 3600_000, None, None),
        ],
    )
    runner = stallcheck.open_runner_ro(str(db))
    try:
        findings = stallcheck.scan_runner_task_runs(runner, _cfg(), now)
    finally:
        runner.close()
    by_unit = {f.unit_id: f.stall_class for f in findings}
    assert by_unit == {"t-lost": "task_run_lost", "t-stale": "task_run_stale"}


# --- Reader C: ingress handler-timeout ----------------------------------------
def test_reader_c_handler_timeout_in_window(tmp_path):
    db = tmp_path / "runner.sqlite"
    now = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)
    now_ms = int(now.timestamp() * 1000)
    _make_runner_db(
        db,
        ingress_rows=[
            ("telegram", "e1", "c", "a", "failed", "handler-timeout", now_ms - 3600_000, "boom"),
            ("telegram", "e2", "c", "a", "failed", "other-reason", now_ms - 3600_000, "x"),
            ("telegram", "e3", "c", "a", "completed", "handler-timeout", now_ms - 3600_000, "x"),
            ("telegram", "e4", "c", "a", "failed", "handler-timeout", now_ms - 9e9, "old"),
        ],
    )
    runner = stallcheck.open_runner_ro(str(db))
    try:
        findings = stallcheck.scan_ingress_timeouts(runner, _cfg(ingress_window_hours=720), now)
    finally:
        runner.close()
    units = {f.unit_id for f in findings}
    assert units == {"telegram/e1"}  # e2 wrong reason, e3 not failed, e4 out of window


# --- deadman engine writes -----------------------------------------------------
def test_feed_deadman_writes_liveness_and_evidence(tmp_path):
    brain = _brain(tmp_path)
    now = datetime.now(UTC)
    finding = stallcheck.Finding(
        stall_class="workflow_passive_wait",
        unit_id="abc",
        terminal_signature="passive_wait:standing by",
        snippet="standing by",
        artifact_path="/tmp/agent-abc.jsonl",
        age_seconds=5400.0,
        occurred_at=now.isoformat(),
    )
    evidence_id = stallcheck.feed_deadman(brain, finding, now)
    brain.commit()

    live = brain.execute(
        "SELECT deadman_due_at FROM loop_liveness WHERE loop_id=? AND run_id=?",
        (finding.loop_id, "abc"),
    ).fetchone()
    assert live is not None and live["deadman_due_at"] is not None

    ev = brain.execute(
        "SELECT source_type, claim FROM evidence WHERE id=?", (evidence_id,)
    ).fetchone()
    assert ev["source_type"] == "loop_tripwire"
    assert "Stall detected" in ev["claim"]

    # Idempotent: same fingerprint -> same evidence row (upsert, no duplicate).
    again = stallcheck.feed_deadman(brain, finding, now)
    assert again == evidence_id


# --- full run: dedup + self-heartbeat -----------------------------------------
def test_run_pages_once_then_dedups_and_heartbeats(tmp_path, monkeypatch):
    wf = tmp_path / "wf_stall"
    wf.mkdir()
    agent = wf / "agent-stall01.jsonl"
    _write_agent_jsonl(agent, last_text="Nothing more to do. Standing by, waiting on the monitor.")
    _age(agent, minutes=90)

    cfg = _cfg(
        workflow_globs=(f"{tmp_path}/wf_*/",),
        pager_chat_id="123",
        pager_openclaw_json=str(tmp_path / "openclaw.json"),
    )

    sent: list[str] = []
    monkeypatch.setattr(stallcheck, "send_telegram", lambda c, text: sent.append(text) or 200)

    brain = _brain(tmp_path)
    now = datetime.now(UTC)

    report1 = stallcheck.run(cfg, brain, runner=None, now=now)
    assert report1.first_run is True
    assert len(report1.paged) == 1
    assert len(sent) == 1
    assert "Standing by" in sent[0] or "standing by" in sent[0]
    assert report1.page_status == 200

    # self-heartbeat row exists
    hb = brain.execute(
        "SELECT last_heartbeat_at FROM loop_liveness WHERE loop_id='stallcheck' "
        "AND run_id='heartbeat'"
    ).fetchone()
    assert hb is not None and hb["last_heartbeat_at"] is not None

    # Second run, same stall -> NOTHING new is paged (dedup).
    report2 = stallcheck.run(cfg, brain, runner=None, now=now + timedelta(minutes=15))
    assert report2.first_run is False
    assert report2.paged == []
    assert len(sent) == 1  # no second Telegram message


def test_digest_is_bounded_below_telegram_limit(tmp_path):
    findings = [
        stallcheck.Finding(
            stall_class="ingress_handler_timeout",
            unit_id=f"telegram/00000009022238{i:02d}",
            terminal_signature="handler-timeout",
            snippet="Telegram isolated polling spool handler timed out behind update " * 2,
            artifact_path=f"runner:channel_ingress_events/00000009022238{i:02d}",
            age_seconds=3600.0 * i,
            occurred_at=datetime.now(UTC).isoformat(),
        )
        for i in range(40)
    ]
    text, included = stallcheck.build_digest_message(findings, backlog=True)
    assert len(text) <= stallcheck.MAX_MESSAGE_CHARS
    assert 0 < included < len(findings)
    assert "more (next digest)" in text


def test_run_failed_send_keeps_findings_eligible_and_first_run(tmp_path, monkeypatch):
    wf = tmp_path / "wf_stall"
    wf.mkdir()
    agent = wf / "agent-retry01.jsonl"
    _write_agent_jsonl(agent, last_text="Standing by, waiting on the monitor.")
    _age(agent, minutes=90)
    cfg = _cfg(
        workflow_globs=(f"{tmp_path}/wf_*/",),
        pager_chat_id="123",
        pager_openclaw_json=str(tmp_path / "openclaw.json"),
    )
    brain = _brain(tmp_path)

    # First run: Telegram rejects with HTTP 400 -> nothing is marked paged.
    monkeypatch.setattr(stallcheck, "send_telegram", lambda c, t: 400)
    r1 = stallcheck.run(cfg, brain, runner=None, now=datetime.now(UTC))
    assert r1.page_status == 400
    assert stallcheck.is_first_run(brain) is True  # backlog window NOT retired
    fp = r1.paged[0].fingerprint
    assert stallcheck.already_paged(brain, fp) is False  # still eligible

    # Second run: Telegram accepts -> now it pages and dedups thereafter.
    sent: list[str] = []
    monkeypatch.setattr(stallcheck, "send_telegram", lambda c, t: sent.append(t) or 200)
    stallcheck.run(cfg, brain, runner=None, now=datetime.now(UTC) + timedelta(minutes=15))
    assert len(sent) == 1
    assert stallcheck.already_paged(brain, fp) is True
    r3 = stallcheck.run(cfg, brain, runner=None, now=datetime.now(UTC) + timedelta(minutes=30))
    assert r3.paged == []
    assert len(sent) == 1  # no re-page


def test_run_inert_pager_when_unconfigured(tmp_path):
    wf = tmp_path / "wf_stall"
    wf.mkdir()
    agent = wf / "agent-x01.jsonl"
    _write_agent_jsonl(agent, last_text="Standing by, waiting for the monitor.")
    _age(agent, minutes=90)

    cfg = _cfg(workflow_globs=(f"{tmp_path}/wf_*/",))  # no pager config
    brain = _brain(tmp_path)
    report = stallcheck.run(cfg, brain, runner=None, now=datetime.now(UTC))
    assert len(report.paged) == 1
    assert report.page_status is None  # inert: found + recorded, but never sent


def test_read_bot_token_extracts_and_missing(tmp_path):
    cfg_file = tmp_path / "openclaw.json"
    cfg_file.write_text(
        json.dumps(
            {"channels": {"telegram": {"accounts": {"default": {"botToken": "SECRET123"}}}}}
        )
    )
    assert stallcheck.read_bot_token(str(cfg_file), "default") == "SECRET123"
    assert stallcheck.read_bot_token(str(cfg_file), "missing") is None
    assert stallcheck.read_bot_token(str(tmp_path / "nope.json"), "default") is None


# --- mcp lock-retry patch ------------------------------------------------------
class _FakeConn:
    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


def test_mcp_retries_on_database_locked(monkeypatch):
    from ocbrain import mcp

    calls = {"n": 0}

    def flaky_call_tool(conn, params):
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return {"ok": True}

    sleeps: list[float] = []
    monkeypatch.setattr(mcp, "call_tool", flaky_call_tool)
    monkeypatch.setattr(mcp.time, "sleep", lambda s: sleeps.append(s))

    conn = _FakeConn()
    result = mcp._call_tool_with_lock_retry(conn, {"name": "brain.ingest"})
    assert result == {"ok": True}
    assert calls["n"] == 3
    assert sleeps == [0.25, 0.25]
    assert conn.rollbacks == 2  # rolled back before each retry


def test_mcp_reraises_after_exhausting_retries(monkeypatch):
    from ocbrain import mcp

    def always_locked(conn, params):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(mcp, "call_tool", always_locked)
    monkeypatch.setattr(mcp.time, "sleep", lambda s: None)
    with pytest.raises(sqlite3.OperationalError):
        mcp._call_tool_with_lock_retry(_FakeConn(), {"name": "brain.ingest"})


def test_mcp_non_lock_error_not_retried(monkeypatch):
    from ocbrain import mcp

    calls = {"n": 0}

    def other_error(conn, params):
        calls["n"] += 1
        raise sqlite3.OperationalError("no such table: widgets")

    monkeypatch.setattr(mcp, "call_tool", other_error)
    with pytest.raises(sqlite3.OperationalError):
        mcp._call_tool_with_lock_retry(_FakeConn(), {"name": "brain.ingest"})
    assert calls["n"] == 1  # not retried
