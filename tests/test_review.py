from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from ocbrain.config import load_config
from ocbrain.db import connect, init_db, upsert_evidence
from ocbrain.ids import content_hash
from ocbrain.review import review_session, review_sessions


@dataclass
class Turn:
    role: str
    text: str = ""
    is_error: bool = False
    occurred_at: str | None = None


@dataclass
class Session:
    session_key: str
    path: str
    turns: list = field(default_factory=list)
    mtime_ns: int | None = None
    agent: str | None = "main"
    fingerprint: str | None = None


def _cfg(tmp_path: Path):
    return load_config(tmp_path / "cfg.json")


def _db(tmp_path: Path) -> sqlite3.Connection:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    return conn


def _kinds(conn: sqlite3.Connection) -> set[str]:
    return {r["kind"] for r in conn.execute("SELECT kind FROM signal_events")}


def _tool_turns(n: int, *, error: bool = False) -> list[Turn]:
    return [Turn(role="tool", text=f"{i} passed", is_error=error) for i in range(n)]


# --------------------------------------------------------------------------- #
def test_settle_gating_skips_recent_session(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    session = Session(
        "s1", "/p/s1", turns=[Turn("assistant", "did work"), *_tool_turns(5)],
        mtime_ns=time.time_ns(),  # just now -> not settled
    )
    result = review_sessions(conn, [session], cfg)
    assert result["changed"] == 0
    assert conn.execute("SELECT COUNT(*) FROM signal_events").fetchone()[0] == 0


def test_settled_session_is_reviewed(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    old = time.time_ns() - int((cfg.review.settle_minutes * 60 + 60) * 1e9)
    session = Session(
        "s1", "/p/s1", turns=[Turn("assistant", "did work"), *_tool_turns(5)], mtime_ns=old
    )
    result = review_sessions(conn, [session], cfg)
    assert result["changed"] == 1
    assert "task_closeout_success" in _kinds(conn)


def test_user_correction_trigger(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    session = Session(
        "s1", "/p/s1",
        turns=[
            Turn("user", "please build the feature"),
            Turn("assistant", "done, shipped it"),
            Turn("user", "that's wrong, you misunderstood the requirement"),
        ],
    )
    review_session(conn, session, cfg)
    row = conn.execute(
        "SELECT polarity, source, session_key FROM signal_events WHERE kind='user_correction'"
    ).fetchone()
    assert row is not None
    assert row["polarity"] == "bad"
    assert row["source"] == "session"
    assert row["session_key"] == "s1"


def test_user_thanks_trigger(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    session = Session(
        "s1", "/p/s1",
        turns=[Turn("assistant", "here you go"), Turn("user", "thanks, perfect work")],
    )
    review_session(conn, session, cfg)
    assert "user_thanks" in _kinds(conn)


def test_task_closeout_creates_harvest_candidate(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    session = Session(
        "s1", "/p/s1", turns=[Turn("assistant", "working through the task"), *_tool_turns(5)]
    )
    result = review_session(conn, session, cfg)
    assert result["candidates"] >= 1
    row = conn.execute(
        "SELECT origin, status, gate FROM knowledge WHERE origin='harvest' LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row["status"] == "candidate"
    assert row["gate"] == "auto"


def test_error_recovery_trigger(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    session = Session(
        "s1", "/p/s1",
        turns=[
            Turn("assistant", "first attempt"),
            Turn("tool", "boom", is_error=True),
            Turn("assistant", "different approach"),
            Turn("tool", "1 passed", is_error=False),
        ],
    )
    result = review_session(conn, session, cfg)
    assert "error_recovery" in _kinds(conn)
    assert result["candidates"] >= 1


def test_test_pass_and_fail_signals(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    session = Session(
        "s1", "/p/s1",
        turns=[
            Turn("tool", "12 passed in 0.5s"),
            Turn("tool", "AssertionError: boom", is_error=True),
        ],
    )
    review_session(conn, session, cfg)
    kinds = _kinds(conn)
    assert "test_pass" in kinds
    assert "test_fail" in kinds


def test_signal_dedup_on_rereview(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    session = Session(
        "s1", "/p/s1",
        turns=[Turn("assistant", "here you go"), Turn("user", "thanks, perfect")],
    )
    review_session(conn, session, cfg)
    first = conn.execute("SELECT COUNT(*) FROM signal_events").fetchone()[0]
    review_session(conn, session, cfg)
    second = conn.execute("SELECT COUNT(*) FROM signal_events").fetchone()[0]
    assert first == second


def test_candidate_links_session_evidence(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    evd = upsert_evidence(
        conn,
        source_type="openclaw_history_file",
        claim="session transcript",
        content_hash=content_hash("s1"),
        source_uri="/p/s1",
    )
    session = Session(
        "s1", "/p/s1", turns=[Turn("assistant", "working"), *_tool_turns(5)]
    )
    review_session(conn, session, cfg)
    linked = conn.execute(
        "SELECT COUNT(*) FROM knowledge_evidence WHERE evidence_id=?", (evd,)
    ).fetchone()[0]
    assert linked >= 1


def test_watermark_skips_unchanged_session(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    session = Session(
        "s1", "/p/s1", turns=[Turn("assistant", "x"), *_tool_turns(5)], fingerprint="fp1"
    )
    first = review_sessions(conn, [session], cfg)
    second = review_sessions(conn, [session], cfg)
    assert first["changed"] == 1
    assert second["changed"] == 0


def test_review_releases_writer_before_parsing_next_lazy_session(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    first = Session(
        "s1", "/p/s1", turns=[Turn("assistant", "done"), *_tool_turns(5)], fingerprint="1"
    )
    second = Session(
        "s2", "/p/s2", turns=[Turn("assistant", "done"), *_tool_turns(5)], fingerprint="2"
    )

    def lazy_sessions():
        yield first
        observer = sqlite3.connect(tmp_path / "ocbrain.sqlite", timeout=0)
        observer.execute("BEGIN IMMEDIATE")
        observer.rollback()
        observer.close()
        yield second

    result = review_sessions(conn, lazy_sessions(), cfg)
    assert result["changed"] == 2
    assert result["writer_lock"]["boundary"] == "session"
    assert result["writer_lock"]["batches_committed"] == 2
