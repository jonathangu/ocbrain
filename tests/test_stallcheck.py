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
import re
import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from ocbrain.db import connect, init_db
from ocbrain_ops import stallcheck

# Verbatim terminal text of the third lane that died passive-waiting tonight and
# EVADED the exact-substring seed lexicon. Regression fixture: it MUST be flagged
# by both the primary (pending-monitor) and secondary (regex lexicon) signals.
MAX_LANE_EVASION_TEXT = (
    "Committed diff confirmed (6 files, +380). Now waiting for the clean live "
    "run to land via the monitor."
)


def _write_agent_records(path, records) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _monitor_tool_use(tool_id: str = "tu_mon", name: str = "Monitor") -> dict:
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": {}}],
        },
    }


def _tool_result_record(tool_id: str = "tu_mon") -> dict:
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}],
        },
    }


def _end_turn_record(text: str) -> dict:
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": text}],
        },
    }


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


# --- Reader A: lexicon-free pending-monitor signal + regex lexicon ------------
def test_evasion_text_flagged_by_both_signals_directly():
    """The verbatim corpse text evaded the exact seeds; both new signals catch it."""
    records = [
        {"type": "user", "message": {"role": "user", "content": "ship it"}},
        _monitor_tool_use(),  # background monitor invoked...
        _end_turn_record(MAX_LANE_EVASION_TEXT),  # ...turn closed with it pending
    ]
    monitor_re = re.compile(stallcheck.DEFAULT_MONITOR_TOOL_PATTERN, re.IGNORECASE)
    # PRIMARY: monitor tool_use with no later tool_result/user event.
    assert stallcheck.detect_pending_monitor(records, monitor_re) == "Monitor"
    # SECONDARY: regex lexicon matches the paraphrase (exact seeds did NOT).
    compiled = stallcheck._compile_lexicon(stallcheck.DEFAULT_PASSIVE_WAIT_LEXICON)
    assert stallcheck.match_passive_wait(MAX_LANE_EVASION_TEXT, compiled) is not None
    # Sanity: the OLD exact-substring approach genuinely missed this text.
    low = MAX_LANE_EVASION_TEXT.lower()
    assert not any(
        seed in low for seed in ("waiting on the monitor", "standing by", "waiting for the monitor")
    )


def test_reader_a_scan_flags_evasion_via_pending_monitor(tmp_path):
    wf = tmp_path / "wf_evade"
    wf.mkdir()
    agent = wf / "agent-maxlane01.jsonl"
    _write_agent_records(
        agent,
        [
            {"type": "user", "message": {"role": "user", "content": "ship it"}},
            _monitor_tool_use(),
            _end_turn_record(MAX_LANE_EVASION_TEXT),
        ],
    )
    _age(agent, minutes=90)

    cfg = _cfg(workflow_globs=(f"{tmp_path}/wf_*/",))
    findings = stallcheck.scan_workflow_stalls(cfg, datetime.now(UTC))
    assert len(findings) == 1
    assert findings[0].stall_class == "workflow_passive_wait"
    # Primary signal wins the signature when a monitor is pending.
    assert findings[0].terminal_signature == "pending_monitor:Monitor"


def test_reader_a_pending_monitor_flags_even_without_lexicon_text(tmp_path):
    """A totally paraphrase-free closing line is still caught by the primary signal."""
    wf = tmp_path / "wf_silent"
    wf.mkdir()
    agent = wf / "agent-silent01.jsonl"
    _write_agent_records(
        agent,
        [
            {"type": "user", "message": {"role": "user", "content": "go"}},
            _monitor_tool_use(name="mcp__ocbrain__Monitor"),
            _end_turn_record("All set. I'll pick this back up once the callback fires."),
        ],
    )
    _age(agent, minutes=90)
    cfg = _cfg(workflow_globs=(f"{tmp_path}/wf_*/",))
    findings = stallcheck.scan_workflow_stalls(cfg, datetime.now(UTC))
    assert len(findings) == 1
    assert findings[0].terminal_signature.startswith("pending_monitor:")


def test_reader_a_healthy_monitor_with_tool_result_not_flagged(tmp_path):
    """A monitor whose result LANDED (tool_result after it) is alive -> no flag."""
    wf = tmp_path / "wf_healthy"
    wf.mkdir()
    agent = wf / "agent-healthy01.jsonl"
    _write_agent_records(
        agent,
        [
            {"type": "user", "message": {"role": "user", "content": "go"}},
            _monitor_tool_use(),
            _tool_result_record(),  # the monitor result came back
            _end_turn_record("Monitor reported success; all done and verified."),
        ],
    )
    _age(agent, minutes=90)
    cfg = _cfg(workflow_globs=(f"{tmp_path}/wf_*/",))
    assert stallcheck.scan_workflow_stalls(cfg, datetime.now(UTC)) == []


def test_pending_monitor_signal_gate_off_falls_back_to_lexicon(tmp_path):
    """With the primary signal disabled, a pending monitor alone does NOT flag,
    but the regex lexicon still catches the corpse text."""
    wf = tmp_path / "wf_gate"
    wf.mkdir()
    # Pending monitor, but closing line has NO passive-wait phrasing.
    silent = wf / "agent-gate01.jsonl"
    _write_agent_records(
        silent,
        [
            {"type": "user", "message": {"role": "user", "content": "go"}},
            _monitor_tool_use(),
            _end_turn_record("Everything is committed and green."),
        ],
    )
    _age(silent, minutes=90)
    cfg_off = _cfg(workflow_globs=(f"{tmp_path}/wf_*/",), pending_monitor_signal=False)
    assert stallcheck.scan_workflow_stalls(cfg_off, datetime.now(UTC)) == []

    # Same gate off, but corpse text present -> secondary regex still fires.
    corpse = wf / "agent-gate02.jsonl"
    _write_agent_records(corpse, [_end_turn_record(MAX_LANE_EVASION_TEXT)])
    _age(corpse, minutes=90)
    findings = stallcheck.scan_workflow_stalls(cfg_off, datetime.now(UTC))
    assert {f.unit_id for f in findings} == {"gate02"}
    assert findings[0].terminal_signature.startswith("passive_wait:")


def test_regex_lexicon_matches_paraphrases_and_seeds():
    compiled = stallcheck._compile_lexicon(stallcheck.DEFAULT_PASSIVE_WAIT_LEXICON)
    # Paraphrases the old exact seeds would have missed:
    for text in (
        MAX_LANE_EVASION_TEXT,
        "Holding here until the deploy completes.",
        "Parked; will resume when the notification arrives.",
        "Blocked waiting for the run to finish.",
    ):
        assert stallcheck.match_passive_wait(text, compiled) is not None, text
    # Original seed phrasing still matches:
    assert stallcheck.match_passive_wait("Standing by.", compiled) is not None
    # A genuinely-done line does NOT match:
    assert stallcheck.match_passive_wait("All tests pass. Done.", compiled) is None


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
def test_feed_deadman_writes_liveness_and_companion_finding(tmp_path):
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
    finding_id = stallcheck.feed_deadman(brain, finding, now)
    brain.commit()

    live = brain.execute(
        "SELECT deadman_due_at FROM loop_liveness WHERE loop_id=? AND run_id=?",
        (finding.loop_id, "abc"),
    ).fetchone()
    assert live is not None and live["deadman_due_at"] is not None

    recorded = brain.execute(
        "SELECT stall_class, snippet FROM watchdog_findings WHERE id=?", (finding_id,)
    ).fetchone()
    assert recorded["stall_class"] == "workflow_passive_wait"
    assert recorded["snippet"] == "standing by"

    # Idempotent: same fingerprint -> same companion finding (upsert, no duplicate).
    again = stallcheck.feed_deadman(brain, finding, now)
    assert again == finding_id


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


def test_brain_deadman_reader_watches_autopilot_without_recursing(tmp_path):
    brain = _brain(tmp_path)
    now = datetime(2026, 7, 10, 20, 0, tzinfo=UTC)
    stale = (now - timedelta(hours=2)).isoformat()
    overdue = (now - timedelta(minutes=5)).isoformat()
    future = (now + timedelta(hours=1)).isoformat()
    brain.executemany(
        """
        INSERT INTO loop_liveness (
          loop_id, run_id, last_heartbeat_at, last_ledger_write_at,
          expected_interval_seconds, deadman_due_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            ("autopilot", "light", stale, stale, 3600, overdue),
            ("autopilot", "heavy", stale, stale, 14400, future),
            ("stall/workflow_passive_wait", "old", stale, stale, None, overdue),
        ],
    )
    brain.commit()

    findings = stallcheck.scan_brain_deadmans(brain, now)
    assert len(findings) == 1
    assert findings[0].stall_class == "loop_deadman"
    assert findings[0].unit_id == "autopilot/light"
    first_fingerprint = findings[0].fingerprint

    # Recovery clears the deadline; a later run with a new heartbeat can create
    # a genuinely new fingerprint instead of being suppressed forever.
    recovered = (now + timedelta(minutes=1)).isoformat()
    brain.execute(
        "UPDATE loop_liveness SET last_heartbeat_at=?, last_ledger_write_at=?, "
        "deadman_due_at=NULL WHERE loop_id='autopilot' AND run_id='light'",
        (recovered, recovered),
    )
    assert stallcheck.scan_brain_deadmans(brain, now + timedelta(minutes=2)) == []
    second_due = (now + timedelta(minutes=3)).isoformat()
    brain.execute(
        "UPDATE loop_liveness SET deadman_due_at=? WHERE loop_id='autopilot' AND run_id='light'",
        (second_due,),
    )
    later = stallcheck.scan_brain_deadmans(brain, now + timedelta(minutes=4))
    assert len(later) == 1
    assert later[0].fingerprint != first_fingerprint


def test_autopilot_failure_reader_pages_partial_and_stale_running(tmp_path):
    brain = _brain(tmp_path)
    now = datetime(2026, 7, 10, 22, 0, tzinfo=UTC)
    brain.executemany(
        """
        INSERT INTO autopilot_runs
          (id, started_at, finished_at, status, stages_json, error)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "run_partial",
                (now - timedelta(hours=2)).isoformat(),
                (now - timedelta(hours=1)).isoformat(),
                "partial",
                json.dumps({"harvest": {"error": "database is locked"}}),
                None,
            ),
            (
                "run_running",
                (now - timedelta(hours=6)).isoformat(),
                None,
                "running",
                "{}",
                None,
            ),
        ],
    )
    brain.commit()

    findings = stallcheck.scan_autopilot_failures(brain, _cfg(), now)
    assert {finding.stall_class for finding in findings} == {
        "autopilot_failed",
        "autopilot_stuck_running",
    }
    partial = next(item for item in findings if item.stall_class == "autopilot_failed")
    assert "harvest: database is locked" in partial.snippet


def test_autopilot_failure_reader_detects_judge_substage_streak(tmp_path):
    brain = _brain(tmp_path)
    now = datetime(2026, 7, 10, 22, 0, tzinfo=UTC)
    payload = json.dumps(
        {"autolabel": {"stages": {"judge": {"error": "read operation timed out"}}}}
    )
    for index in range(2):
        started = now - timedelta(minutes=30 + index * 30)
        brain.execute(
            """
            INSERT INTO autopilot_runs
              (id, started_at, finished_at, status, stages_json, error)
            VALUES (?, ?, ?, 'ok', ?, NULL)
            """,
            (
                f"run_judge_{index}",
                started.isoformat(),
                (started + timedelta(minutes=5)).isoformat(),
                payload,
            ),
        )
    brain.commit()

    findings = stallcheck.scan_autopilot_failures(brain, _cfg(judge_failure_streak=2), now)
    streaks = [item for item in findings if item.stall_class == "judge_failure_streak"]
    assert len(streaks) == 1
    assert "2 consecutive" in streaks[0].snippet


def test_dry_run_reports_brain_deadman_without_writing(tmp_path, capsys):
    brain = _brain(tmp_path)
    stale = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    overdue = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    brain.execute(
        """
        INSERT INTO loop_liveness (
          loop_id, run_id, last_heartbeat_at, last_ledger_write_at,
          expected_interval_seconds, deadman_due_at
        ) VALUES ('autopilot', 'light', ?, ?, 3600, ?)
        """,
        (stale, stale, overdue),
    )
    brain.commit()
    db_path = brain.execute("PRAGMA database_list").fetchone()[2]
    before = brain.total_changes
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "stall_check": {
                    "workflow_globs": [str(tmp_path / "no-workflows" / "*")],
                    "task_output_globs": [str(tmp_path / "no-tasks" / "*")],
                    "runner_db": str(tmp_path / "no-runner.sqlite"),
                }
            }
        )
    )

    rc = stallcheck.main(["--config", str(config_path), "--brain-db", db_path, "--dry-run"])

    assert rc == 0
    assert "loop_deadman" in capsys.readouterr().out
    assert brain.total_changes == before


def test_steady_state_retires_out_of_window_finding_once(tmp_path):
    wf = tmp_path / "wf_old"
    wf.mkdir()
    agent = wf / "agent-old01.jsonl"
    _write_agent_jsonl(agent, last_text="Standing by, waiting for the monitor.")
    _age(agent, minutes=72 * 60)
    cfg = _cfg(
        workflow_globs=(f"{tmp_path}/wf_*/",),
        terminal_backlog_hours=48,
    )
    brain = _brain(tmp_path)
    now = datetime.now(UTC)
    stallcheck.ensure_stall_pages(brain)
    stallcheck.mark_first_run_done(brain, now)
    brain.commit()

    first = stallcheck.run(cfg, brain, runner=None, now=now, send=False)
    assert first.first_run is False
    assert len(first.new_findings) == 1
    assert first.paged == []
    assert len(first.retired) == 1
    row = brain.execute(
        "SELECT paged_at, retired_at, retire_reason, run_count FROM stall_pages "
        "WHERE fingerprint = ?",
        (first.retired[0].fingerprint,),
    ).fetchone()
    assert row["paged_at"] is None
    assert row["retired_at"] is not None
    assert row["retire_reason"] == "outside_backlog_window"
    assert row["run_count"] == 1

    second = stallcheck.run(cfg, brain, runner=None, now=now + timedelta(minutes=15), send=False)
    assert second.new_findings == []
    assert second.retired == []
    assert (
        brain.execute(
            "SELECT run_count FROM stall_pages WHERE fingerprint = ?",
            (first.retired[0].fingerprint,),
        ).fetchone()[0]
        == 1
    )


def test_ensure_stall_pages_adds_retirement_columns_to_legacy_table(tmp_path):
    brain = _brain(tmp_path)
    brain.execute(
        """
        CREATE TABLE stall_pages (
          fingerprint TEXT PRIMARY KEY, stall_class TEXT NOT NULL,
          unit_id TEXT NOT NULL, terminal_signature TEXT,
          first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL,
          paged_at TEXT, run_count INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    stallcheck.ensure_stall_pages(brain)
    columns = {row["name"] for row in brain.execute("PRAGMA table_info(stall_pages)")}
    assert {"retired_at", "retire_reason"} <= columns


def test_pager_network_call_never_holds_sqlite_writer_lock(tmp_path, monkeypatch):
    wf = tmp_path / "wf_stall"
    wf.mkdir()
    agent = wf / "agent-stall01.jsonl"
    _write_agent_jsonl(agent, last_text="Standing by, waiting on the monitor.")
    _age(agent, minutes=90)
    cfg = _cfg(
        workflow_globs=(f"{tmp_path}/wf_*/",),
        pager_chat_id="123",
        pager_openclaw_json=str(tmp_path / "openclaw.json"),
    )

    def observing_send(_cfg, _text):
        observer = sqlite3.connect(tmp_path / "brain.sqlite", timeout=0)
        observer.execute("BEGIN IMMEDIATE")
        assert observer.execute("SELECT COUNT(*) FROM watchdog_findings").fetchone()[0] >= 1
        observer.rollback()
        observer.close()
        return 200

    monkeypatch.setattr(stallcheck, "send_telegram", observing_send)
    brain = _brain(tmp_path)
    report = stallcheck.run(cfg, brain, runner=None, now=datetime.now(UTC))
    assert report.page_status == 200


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


def test_daily_canary_sends_once_and_records_real_delivery(tmp_path, monkeypatch):
    brain = _brain(tmp_path)
    cfg = _cfg(
        daily_canary_enabled=True,
        daily_canary_hour_utc=0,
        pager_chat_id="chat",
        pager_openclaw_json="/configured",
    )
    sent: list[str] = []
    monkeypatch.setattr(stallcheck, "send_telegram", lambda _cfg, text: sent.append(text) or 200)
    now = datetime(2026, 7, 10, 20, 0, tzinfo=UTC)

    first = stallcheck.run(cfg, brain, now=now)
    second = stallcheck.run(cfg, brain, now=now + timedelta(minutes=15))

    assert first.canary_status == 200
    assert second.canary_status is None
    assert len(sent) == 1
    assert "pager canary" in sent[0].lower()
    assert (
        brain.execute(
            "SELECT paged_at FROM stall_pages WHERE fingerprint='__canary__:2026-07-10'"
        ).fetchone()["paged_at"]
        is not None
    )


def test_read_bot_token_extracts_and_missing(tmp_path):
    cfg_file = tmp_path / "openclaw.json"
    cfg_file.write_text(
        json.dumps({"channels": {"telegram": {"accounts": {"default": {"botToken": "SECRET123"}}}}})
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

    def flaky_call_tool(conn, params, *, profile="runtime"):
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

    def always_locked(conn, params, *, profile="runtime"):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(mcp, "call_tool", always_locked)
    monkeypatch.setattr(mcp.time, "sleep", lambda s: None)
    with pytest.raises(sqlite3.OperationalError):
        mcp._call_tool_with_lock_retry(_FakeConn(), {"name": "brain.ingest"})


def test_mcp_non_lock_error_not_retried(monkeypatch):
    from ocbrain import mcp

    calls = {"n": 0}

    def other_error(conn, params, *, profile="runtime"):
        calls["n"] += 1
        raise sqlite3.OperationalError("no such table: widgets")

    monkeypatch.setattr(mcp, "call_tool", other_error)
    with pytest.raises(sqlite3.OperationalError):
        mcp._call_tool_with_lock_retry(_FakeConn(), {"name": "brain.ingest"})
    assert calls["n"] == 1  # not retried


# --- stallcheck brain-write lock-retry patch (mirrors mcp.py exactly) ----------
def test_stallcheck_retries_on_database_locked(monkeypatch):
    calls = {"n": 0}

    def flaky_write(x):
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return x + 1

    sleeps: list[float] = []
    monkeypatch.setattr(stallcheck.time, "sleep", lambda s: sleeps.append(s))

    conn = _FakeConn()
    result = stallcheck._write_with_lock_retry(conn, flaky_write, 41)
    assert result == 42
    assert calls["n"] == 3
    assert sleeps == [0.25, 0.25]
    assert conn.rollbacks == 2  # rolled back before each retry


def test_stallcheck_reraises_after_exhausting_retries(monkeypatch):
    def always_locked():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(stallcheck.time, "sleep", lambda s: None)
    with pytest.raises(sqlite3.OperationalError):
        stallcheck._write_with_lock_retry(_FakeConn(), always_locked)


def test_stallcheck_non_lock_error_not_retried():
    calls = {"n": 0}

    def other_error():
        calls["n"] += 1
        raise sqlite3.OperationalError("no such table: widgets")

    with pytest.raises(sqlite3.OperationalError):
        stallcheck._write_with_lock_retry(_FakeConn(), other_error)
    assert calls["n"] == 1  # not retried


def test_feed_deadman_and_heartbeat_wait_out_a_concurrent_writer_lock(tmp_path, monkeypatch):
    """Reproduces the live crash: another connection holds a BEGIN IMMEDIATE
    write lock (standing in for the autopilot light cycle) for longer than a
    single busy_timeout window. Both the deadman write (feed_deadman) and the
    self-heartbeat write (upsert_self_heartbeat) must wait it out via the
    bound retry and succeed instead of raising 'database is locked'."""
    db_path = tmp_path / "brain.sqlite"
    brain = connect(db_path)
    init_db(brain)
    brain.commit()

    # Shrink the busy_timeout so the first attempt exhausts fast (deterministic,
    # fast test) instead of relying on the real 5s window.
    monkeypatch.setattr(stallcheck, "BRAIN_BUSY_TIMEOUT_MS", 50)
    brain.execute(f"PRAGMA busy_timeout={stallcheck.BRAIN_BUSY_TIMEOUT_MS}")

    lock_acquired = threading.Event()
    release_after = 0.2  # longer than the 50ms busy_timeout window

    def hold_writer_lock():
        writer = sqlite3.connect(str(db_path), timeout=0)
        writer.execute("BEGIN IMMEDIATE")
        lock_acquired.set()
        time.sleep(release_after)
        writer.commit()
        writer.close()

    thread = threading.Thread(target=hold_writer_lock)
    thread.start()
    assert lock_acquired.wait(timeout=2)
    time.sleep(0.02)  # let the writer's BEGIN IMMEDIATE actually land

    now = datetime.now(UTC)
    finding = stallcheck.Finding(
        stall_class="workflow_passive_wait",
        unit_id="lock-contend",
        terminal_signature="passive_wait:standing by",
        snippet="standing by",
        artifact_path="/tmp/agent-lock.jsonl",
        age_seconds=5400.0,
        occurred_at=now.isoformat(),
    )
    cfg = _cfg(workflow_globs=(), task_output_globs=(), runner_db="/nonexistent")

    # Both writes must succeed (retry survives the lock) rather than raising.
    finding_id = stallcheck._write_with_lock_retry(
        brain, stallcheck.feed_deadman, brain, finding, now
    )
    stallcheck._write_with_lock_retry(brain, stallcheck.upsert_self_heartbeat, brain, cfg, now)
    brain.commit()
    thread.join(timeout=2)

    assert finding_id
    live = brain.execute(
        "SELECT deadman_due_at FROM loop_liveness WHERE loop_id=? AND run_id=?",
        (finding.loop_id, "lock-contend"),
    ).fetchone()
    assert live is not None and live["deadman_due_at"] is not None

    hb = brain.execute(
        "SELECT last_heartbeat_at FROM loop_liveness WHERE loop_id='stallcheck' "
        "AND run_id='heartbeat'"
    ).fetchone()
    assert hb is not None and hb["last_heartbeat_at"] is not None


def test_main_skips_clean_when_lock_survives_the_retry_budget(tmp_path, monkeypatch, capsys):
    """If a competing writer holds the brain lock beyond the whole bound-retry
    budget (observed live: autopilot's multi-minute review/tripwires stage),
    main() must exit 0 with a clear SKIPPED line instead of crashing — a crash
    is what actually took the watchman down. Nothing is lost: the next
    15-minute cycle re-scans and re-feeds every finding (idempotent)."""
    brain_path = tmp_path / "brain.sqlite"
    seed = connect(brain_path)
    init_db(seed)
    seed.commit()
    seed.close()

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text("{}")

    def always_locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(stallcheck, "run", always_locked)
    monkeypatch.setattr(
        stallcheck, "open_runner_ro", lambda path: (_ for _ in ()).throw(sqlite3.Error())
    )

    rc = stallcheck.main(["--config", str(cfg_path), "--brain-db", str(brain_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SKIPPED" in out
    assert "brain database busy" in out


def test_main_reraises_non_lock_operational_error(tmp_path, monkeypatch):
    brain_path = tmp_path / "brain.sqlite"
    seed = connect(brain_path)
    init_db(seed)
    seed.commit()
    seed.close()
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text("{}")

    def other_error(*args, **kwargs):
        raise sqlite3.OperationalError("no such table: widgets")

    monkeypatch.setattr(stallcheck, "run", other_error)
    monkeypatch.setattr(
        stallcheck, "open_runner_ro", lambda path: (_ for _ in ()).throw(sqlite3.Error())
    )

    with pytest.raises(sqlite3.OperationalError):
        stallcheck.main(["--config", str(cfg_path), "--brain-db", str(brain_path)])
