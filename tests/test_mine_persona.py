from __future__ import annotations

import subprocess
from pathlib import Path

from ocbrain.config import load_config
from ocbrain.dataset.batching import DatasetWriteBatch
from ocbrain.dataset.mine_persona import (
    commit_examples,
    doc_examples,
    mine_persona,
    telegram_examples,
)
from ocbrain.dataset.transcripts import Session, Turn
from ocbrain.db import connect, init_db

# Unit tests must be isolated from the operator's on-disk config
# (data/ocbrain.config.json): loading from a guaranteed-missing path yields the
# pure defaults (e.g. empty persona_git_authors == match-all for the throwaway
# test repos below), independent of whatever real identity config is installed.
CFG = load_config(path=Path("/nonexistent/ocbrain-unit-test-config.json"))
AGENT_PROMPT = "The agent has finished the task and is awaiting your review and next instruction."
JON_MSG = "Let's ship the release tonight after the final smoke test passes cleanly for us."


def _sess(*turns: Turn, agent="main") -> Session:
    return Session(
        session_id="s1",
        source_kind="openclaw_session",
        source_uri="/x/s.jsonl",
        runtime="openclaw",
        agent=agent,
        turns=tuple(turns),
        occurred_at="2026-07-01T00:00:00Z",
    )


def _verified(text: str) -> Turn:
    return Turn(role="user", text=text, kind="telegram_envelope",
                sender_verified=True, authored_by="1000000001")


def _bare(text: str) -> Turn:
    return Turn(role="user", text=text, kind="bare")


def _a(text: str) -> Turn:
    return Turn(role="assistant", text=text)


def _conn(tmp_path: Path):
    conn = connect(tmp_path / "db.sqlite")
    init_db(conn)
    return conn


def test_telegram_verified_target():
    session = _sess(_a(AGENT_PROMPT), _verified(JON_MSG))
    examples = telegram_examples(session, CFG)
    assert len(examples) == 1
    msgs = examples[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == AGENT_PROMPT  # user side = preceding assistant turn
    assert msgs[2]["content"] == JON_MSG  # assistant target = Jonathan's message
    assert examples[0]["confidence"] == 0.85
    assert examples[0]["sender_verified"] is True


def test_opener_skipped():
    # a verified turn with no preceding assistant turn is an opener -> skipped
    session = _sess(_verified(JON_MSG))
    assert telegram_examples(session, CFG) == []


def test_unverified_bare_penalty_and_verified_only():
    session = _sess(_a(AGENT_PROMPT), _bare(JON_MSG), agent="main")
    examples = telegram_examples(session, CFG)
    assert len(examples) == 1
    assert examples[0]["sender_verified"] is False
    assert abs(examples[0]["confidence"] - (0.85 - 0.2)) < 1e-9
    assert "bare_unverified" in examples[0]["reasons"]
    # --verified-only drops the bare turn
    assert telegram_examples(session, CFG, verified_only=True) == []


def test_bare_requires_direct_agent():
    # a non-direct agent's bare turns are not admitted as persona voice
    session = _sess(_a(AGENT_PROMPT), _bare(JON_MSG), agent="planner")
    assert telegram_examples(session, CFG) == []


def test_style_inadmissible_skipped():
    session = _sess(_a(AGENT_PROMPT), _verified("/deploy --now"))
    assert telegram_examples(session, CFG) == []


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Persona Tester"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "persona@example.test"], cwd=path, check=True)


def _commit(path: Path, filename: str, message: str) -> None:
    (path / filename).write_text("content\n", encoding="utf-8")
    subprocess.run(["git", "add", filename], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=path, check=True)


def test_commit_mining_excludes_agent_authored(tmp_path: Path):
    conn = _conn(tmp_path)
    repo = tmp_path / "myrepo"
    _init_repo(repo)
    _commit(repo, "a.txt", "Add the initial project scaffolding and configuration files")
    agent_msg = "Refactor the pipeline runner for clarity\n\nCo-Authored-By: Claude <noreply@x.com>"
    _commit(repo, "b.txt", agent_msg)
    examples = commit_examples(conn, repo, CFG)
    assert len(examples) == 1
    assert "scaffolding" in examples[0]["target_text"]
    # git_commit evidence row created for provenance
    rows = conn.execute(
        "SELECT source_uri FROM evidence WHERE source_type = 'git_commit'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["source_uri"].startswith("git://myrepo#")
    assert examples[0]["evidence_ids"]


def test_commit_mining_releases_writer_between_git_candidates(tmp_path: Path):
    conn = _conn(tmp_path)
    repo = tmp_path / "myrepo"
    _init_repo(repo)
    _commit(repo, "a.txt", "Document the first clear operator decision")
    _commit(repo, "b.txt", "Explain the second verified system tradeoff")
    batch = DatasetWriteBatch(conn, max_operations=50, max_seconds=60)

    examples = commit_examples(conn, repo, CFG, write_batch=batch)
    assert len(examples) == 2
    assert conn.in_transaction is False
    assert batch.metrics()["batches_committed"] == 2


def test_docs_off_by_default(tmp_path: Path):
    conn = _conn(tmp_path)
    assert doc_examples(conn, CFG) == []


def test_mine_persona_end_to_end(tmp_path: Path):
    conn = _conn(tmp_path)
    repo = tmp_path / "myrepo"
    _init_repo(repo)
    _commit(repo, "a.txt", "Add the initial project scaffolding and configuration files")
    session = _sess(_a(AGENT_PROMPT), _verified(JON_MSG))
    result = mine_persona(conn, cfg=CFG, sessions=[session], repos=[repo])
    assert result["stored"] >= 2  # one telegram + one commit
    labels = [
        r["source_kind"]
        for r in conn.execute(
            "SELECT source_kind FROM dataset_examples WHERE dataset='persona'"
        )
    ]
    assert "git_commit" in labels and "openclaw_session" in labels


def _write_openclaw_session(root: Path, sid: str, *messages: dict) -> Path:
    import json

    path = root / "agents" / "main" / "sessions" / f"{sid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [{"type": "session", "id": sid, "version": 1, "cwd": "/w",
              "timestamp": "2026-07-01T00:00:00Z"}]
    for msg in messages:
        lines.append({"type": "message", "message": msg})
    path.write_text("\n".join(json.dumps(o) for o in lines), encoding="utf-8")
    return path


def _persona_corpus(root: Path, n: int) -> None:
    for i in range(n):
        _write_openclaw_session(
            root, f"p{i}",
            {"role": "assistant", "content": AGENT_PROMPT},
            # unique per file so each stores (identical text would near-dup dedup)
            {"role": "user", "content": f"{JON_MSG} Round {i} is a go on our side."},
        )


def test_mine_persona_roots_incremental_second_pass_zero_parses(tmp_path: Path):
    """Spec proof: persona is now watermark-gated — the re-scan every run is gone.

    Before v0.3 mine_persona used iter_transcript_files and re-parsed EVERY
    transcript on every run; here the second unchanged pass parses zero files.
    """
    from ocbrain.fsutil import ParseCache

    conn = _conn(tmp_path)
    root = tmp_path / "corpus"
    _persona_corpus(root, 3)

    first = ParseCache()
    r1 = mine_persona(conn, cfg=CFG, roots=[str(root)], repos=[], parse_cache=first)
    assert r1["files_mined"] == 3
    assert first.parses == 3
    assert r1["stored"] == 3  # one bare persona target per file (agent 'main')

    second = ParseCache()
    r2 = mine_persona(conn, cfg=CFG, roots=[str(root)], repos=[], parse_cache=second)
    assert r2["files_mined"] == 0  # nothing re-scanned
    assert second.parses == 0  # ZERO re-parses of unchanged transcripts


def test_mine_persona_and_sft_share_one_parse_per_file(tmp_path: Path):
    """Run-shared memo: with no founder configured, SFT + persona parse each
    new file ONCE total, not once per miner."""
    from ocbrain.dataset.mine_sft import mine_sft
    from ocbrain.fsutil import ParseCache

    conn = _conn(tmp_path)
    root = tmp_path / "corpus"
    # Files that yield BOTH an SFT exchange and a persona target.
    for i in range(3):
        _write_openclaw_session(
            root, f"m{i}",
            {"role": "user", "content": f"please run release checklist {i} for tonight"},
            {"role": "assistant", "content":
                "This is a sufficiently long assistant answer that clears the SFT floor cleanly."},
            {"role": "user", "content": JON_MSG},
        )
    cache = ParseCache()
    mine_sft(conn, cfg=CFG, roots=[str(root)], parse_cache=cache)
    mine_persona(conn, cfg=CFG, roots=[str(root)], repos=[], parse_cache=cache)
    # SFT parsed 3 (cold); persona reused all 3 from the shared memo → still 3.
    assert cache.parses == 3
    assert cache.hits == 3
