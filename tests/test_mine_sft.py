from __future__ import annotations

import json
from pathlib import Path

from ocbrain.config import load_config
from ocbrain.dataset.mine_sft import label_exchange, mine_sft, segment_exchanges
from ocbrain.dataset.transcripts import Session, Turn
from ocbrain.db import connect, init_db, upsert_evidence

CFG = load_config()
TARGET = "This is a sufficiently long assistant answer that clears the eighty-char SFT floor."


def _sess(*turns: Turn, agent="main", sid="s1", uri="/x/s.jsonl") -> Session:
    return Session(
        session_id=sid,
        source_kind="openclaw_session",
        source_uri=uri,
        runtime="openclaw",
        agent=agent,
        turns=tuple(turns),
        occurred_at="2026-07-01T00:00:00Z",
    )


def _u(text: str, kind: str = "bare") -> Turn:
    return Turn(role="user", text=text, kind=kind)


def _a(text: str, n: int = 0) -> Turn:
    return Turn(role="assistant", text=text, n_tool_calls=n)


def _tool(text: str, error: bool = False) -> Turn:
    return Turn(role="tool", text=text, tool_error=error)


def _conn(tmp_path: Path):
    conn = connect(tmp_path / "db.sqlite")
    init_db(conn)
    return conn


def test_segment_basic_and_min_assistant():
    session = _sess(_u("please do X"), _a(TARGET))
    exchanges = segment_exchanges(session, CFG)
    assert len(exchanges) == 1
    assert exchanges[0].target_text == TARGET
    assert exchanges[0].context[-1] == {"role": "user", "content": "please do X"}
    # too-short assistant target yields no exchange
    assert segment_exchanges(_sess(_u("hi"), _a("ok")), CFG) == []


def test_segment_context_turn_bound():
    turns: list[Turn] = []
    for i in range(20):
        turns.append(_u(f"request number {i} with enough words here"))
        turns.append(_a(TARGET + f" variant {i}"))
    exchanges = segment_exchanges(_sess(*turns), CFG)
    for exchange in exchanges:
        assert len(exchange.context) <= CFG.dataset.sft_max_context_turns
        assert sum(len(m["content"]) for m in exchange.context) <= CFG.dataset.sft_max_context_chars


def test_label_affirmation_good():
    session = _sess(_u("do X"), _a(TARGET), _u("thanks, perfect, ship it"))
    exchange = segment_exchanges(session, CFG)[0]
    label, conf, reasons = label_exchange(session, exchange, CFG)
    assert label == "good" and conf == 0.9 and "affirmation" in reasons


def test_label_correction_bad():
    session = _sess(_u("do X"), _a(TARGET), _u("no, that's wrong, not what I asked"))
    exchange = segment_exchanges(session, CFG)[0]
    label, _conf, reasons = label_exchange(session, exchange, CFG)
    assert label == "bad" and "correction_followup" in reasons


def test_label_task_success_good():
    session = _sess(_u("run the pipeline"), _a(TARGET, n=5))
    exchange = segment_exchanges(session, CFG)[0]
    assert exchange.n_tool_calls == 5
    label, conf, reasons = label_exchange(session, exchange, CFG)
    assert label == "good" and conf == 0.7 and "task_success" in reasons


def test_label_error_recovery_good():
    session = _sess(_u("fix the build"), _tool("Error: boom", error=True), _a(TARGET, n=2))
    exchange = segment_exchanges(session, CFG)[0]
    assert exchange.had_tool_error and not exchange.trailing_tool_error
    label, conf, reasons = label_exchange(session, exchange, CFG)
    assert label == "good" and conf == 0.8 and "error_recovery" in reasons


def test_label_neutral():
    session = _sess(_u("what is the capital"), _a(TARGET))
    exchange = segment_exchanges(session, CFG)[0]
    label, conf, reasons = label_exchange(session, exchange, CFG)
    assert label == "neutral" and conf == 0.5


def test_retrieval_crossref():
    session = _sess(_u("what is the capital"), _a(TARGET))
    exchange = segment_exchanges(session, CFG)[0]
    bad = label_exchange(session, exchange, CFG, retrieval_outcomes=["harmful"])
    assert bad[0] == "bad"
    good = label_exchange(session, exchange, CFG, retrieval_outcomes=["helpful"])
    assert good[0] == "good" and "retrieval_good" in good[2]


def test_injected_only_session_yields_nothing(tmp_path: Path):
    conn = _conn(tmp_path)
    session = _sess(_u("[Subagent Context] work", kind="injected"), _a(TARGET))
    result = mine_sft(conn, cfg=CFG, sessions=[session])
    assert result["examined"] == 0 and result["stored"] == 0
    assert conn.execute("SELECT COUNT(*) FROM dataset_examples").fetchone()[0] == 0


def test_provenance_and_composed_scope(tmp_path: Path):
    conn = _conn(tmp_path)
    # a project-scoped transcript evidence row already exists (harvest stage)
    evidence_id = upsert_evidence(
        conn,
        source_type="openclaw_history_file",
        source_runtime="openclaw",
        source_uri="/x/s.jsonl",
        content_hash="fp1",
        claim="transcript",
        privacy_scope="project",
    )
    session = _sess(_u("do X please"), _a(TARGET))
    result = mine_sft(conn, cfg=CFG, sessions=[session])
    assert result["stored"] == 1
    row = conn.execute(
        "SELECT evidence_ids, privacy_scope, quality_label FROM dataset_examples"
    ).fetchone()
    assert json.loads(row["evidence_ids"]) == [evidence_id]
    assert row["privacy_scope"] == "project"


def _write_openclaw_session(root: Path, sid: str, *messages: dict) -> Path:
    """Write a minimal on-disk openclaw session transcript under agents/main."""
    path = root / "agents" / "main" / "sessions" / f"{sid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [{"type": "session", "id": sid, "version": 1, "cwd": "/w",
              "timestamp": "2026-07-01T00:00:00Z"}]
    for msg in messages:
        lines.append({"type": "message", "message": msg})
    path.write_text("\n".join(json.dumps(o) for o in lines), encoding="utf-8")
    return path


def _corpus(root: Path, n: int) -> None:
    for i in range(n):
        _write_openclaw_session(
            root,
            f"s{i}",
            {"role": "user", "content": f"please run release checklist number {i}"},
            {"role": "assistant", "content": TARGET},
        )


def test_mine_sft_roots_incremental_second_pass_zero_parses(tmp_path: Path):
    """Spec proof: mine twice; the second unchanged pass parses zero files."""
    from ocbrain.fsutil import ParseCache

    conn = _conn(tmp_path)
    root = tmp_path / "corpus"
    _corpus(root, 3)

    first = ParseCache()
    r1 = mine_sft(conn, cfg=CFG, roots=[str(root)], parse_cache=first)
    assert r1["files_mined"] == 3
    assert first.parses == 3  # every file parsed once on the cold pass

    second = ParseCache()
    r2 = mine_sft(conn, cfg=CFG, roots=[str(root)], parse_cache=second)
    assert r2["files_mined"] == 0  # watermark skipped every unchanged file
    assert second.parses == 0  # ZERO re-parses of unchanged transcripts


def test_mine_sft_roots_reparses_only_changed_file(tmp_path: Path):
    from ocbrain.fsutil import ParseCache

    conn = _conn(tmp_path)
    root = tmp_path / "corpus"
    _corpus(root, 3)
    mine_sft(conn, cfg=CFG, roots=[str(root)])

    # Mutate exactly one transcript so its fingerprint (size/mtime) advances.
    _write_openclaw_session(
        root, "s1",
        {"role": "user", "content": "please run the updated release checklist now"},
        {"role": "assistant", "content": TARGET + " Extra."},
    )
    cache = ParseCache()
    r = mine_sft(conn, cfg=CFG, roots=[str(root)], parse_cache=cache)
    assert r["files_mined"] == 1  # only the changed file is revisited
    assert cache.parses == 1
