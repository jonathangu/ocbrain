from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

from ocbrain.config import load_config
from ocbrain.db import connect, init_db
from ocbrain_training.dataset.mine_dpo import mine_dpo
from ocbrain_training.dataset.mine_persona import mine_persona
from ocbrain_training.dataset.transcripts import Session, Turn

CFG = load_config(path=Path("/nonexistent/ocbrain-unit-test-config.json"))
AGENT_PROMPT = "The agent completed the task and needs the operator's final decision."
HUMAN_REPLY = "Ship it after the final checks pass cleanly and the evidence is recorded."


def _conn(tmp_path: Path):
    conn = connect(tmp_path / "db.sqlite")
    init_db(conn)
    return conn


def _session() -> Session:
    return Session(
        session_id="budget-test",
        source_kind="openclaw_session",
        source_uri="/test/budget.jsonl",
        runtime="openclaw",
        agent="main",
        turns=(
            Turn(role="assistant", text=AGENT_PROMPT),
            Turn(role="user", text=HUMAN_REPLY, kind="bare"),
        ),
        occurred_at="2026-07-01T00:00:00Z",
    )


def _repo(path: Path) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Budget Tester"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "budget@example.test"], cwd=path, check=True)
    (path / "note.txt").write_text("verified\n", encoding="utf-8")
    subprocess.run(["git", "add", "note.txt"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "Document the verified operator decision"],
        cwd=path,
        check=True,
    )


def test_persona_budget_stops_before_git_tail(tmp_path: Path):
    conn = _conn(tmp_path)
    repo = tmp_path / "repo"
    _repo(repo)

    result = mine_persona(
        conn,
        cfg=CFG,
        sessions=[_session()],
        repos=[repo],
        time_budget_seconds=0,
    )

    assert result["stored"] == 0
    count = conn.execute("SELECT COUNT(*) FROM evidence WHERE source_type='git_commit'").fetchone()[
        0
    ]
    assert count == 0


def test_dpo_budget_skips_sessions_and_event_tail(tmp_path: Path, monkeypatch):
    module = importlib.import_module("ocbrain_training.dataset.mine_dpo")
    conn = _conn(tmp_path)

    def should_not_scan_events(*args, **kwargs):
        raise AssertionError("event mining ran after the stage deadline")

    monkeypatch.setattr(module, "find_event_pairs", should_not_scan_events)
    result = mine_dpo(
        conn,
        cfg=CFG,
        sessions=[_session()],
        include_events=True,
        time_budget_seconds=0,
    )

    assert result["examined"] == 0
    assert result["stored"] == 0
