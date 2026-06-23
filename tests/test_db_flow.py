import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ocbrain import cli
from ocbrain.db import (
    connect,
    counts,
    init_db,
    link_knowledge_evidence,
    mark_knowledge_stale,
    search,
    upsert_evidence,
    upsert_knowledge,
)
from ocbrain.excerpt import write_excerpt
from ocbrain.maintenance import check_loop_liveness, heal_conflicts, prune_knowledge
from ocbrain.proposals import write_proposal


def test_schema_burns_down_legacy_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE events (id TEXT)")
    conn.execute("CREATE TABLE candidates (id TEXT)")
    conn.commit()
    conn.close()

    conn = connect(db_path)
    init_db(conn)
    names = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
    }

    assert {"evidence", "knowledge", "knowledge_evidence", "memory"} <= names
    assert "events" not in names
    assert "candidates" not in names
    assert "invalidations" not in names
    assert "candidate_decisions" not in names


def test_value_knowledge_requires_exactly_one_typed_value(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    try:
        upsert_knowledge(
            conn,
            knowledge_type="value",
            gate="auto",
            subject="runtime",
            predicate="bad",
            value_text="yes",
            value_bool=True,
        )
    except ValueError as exc:
        assert "exactly one typed value" in str(exc)
    else:
        raise AssertionError("expected typed value validation")


def test_identity_spine_dedupes_value_across_runtime_evidence(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    codex_evidence = upsert_evidence(
        conn,
        source_type="closeout",
        source_runtime="codex",
        source_uri="/tmp/codex.json",
        content_hash="hash-codex",
        claim="Codex uses the shared ocbrain MCP server.",
    )
    claude_evidence = upsert_evidence(
        conn,
        source_type="closeout",
        source_runtime="claude_code",
        source_uri="/tmp/claude.json",
        content_hash="hash-claude",
        claim="Claude Code uses the shared ocbrain MCP server.",
    )
    first_id = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime_memory",
        predicate="shared_brain_enabled",
        value_bool=True,
        status="current",
        inject=True,
        confidence=0.9,
    )
    second_id = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime_memory",
        predicate="shared_brain_enabled",
        value_bool=True,
        status="current",
        inject=True,
        confidence=0.91,
    )
    link_knowledge_evidence(conn, first_id, codex_evidence)
    link_knowledge_evidence(conn, second_id, claude_evidence)
    conn.commit()

    assert first_id == second_id
    assert conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM knowledge_evidence").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0] == 1


def test_capability_is_human_gated_candidate(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="capability",
        gate="auto",
        slug="verified-test-workflow",
        title="Verified test workflow",
        status="current",
        risk="high",
    )
    row = conn.execute(
        "SELECT gate, status FROM knowledge WHERE id = ?",
        (knowledge_id,),
    ).fetchone()

    assert row["gate"] == "human"
    assert row["status"] == "candidate"


def test_search_filters_loop_tagged_knowledge(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="loop:repo-quality-loop",
        predicate="typecheck_errors",
        value_numeric=9,
        status="current",
        loop_tags={"loop_id": "repo-quality-loop", "family": "typecheck_narrowing"},
        project="ocbrain",
    )
    upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="loop:other",
        predicate="typecheck_errors",
        value_numeric=100,
        status="current",
        loop_tags={"loop_id": "other", "family": "typecheck_narrowing"},
        project="ocbrain",
    )
    conn.commit()

    rows = search(
        conn,
        "typecheck errors",
        filters={"loop_id": "repo-quality-loop", "family": "typecheck_narrowing"},
    )

    assert len(rows) == 1
    assert rows[0]["kind"] == "knowledge:value"


def test_cli_evidence_and_value_digest(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "ocbrain.sqlite"

    assert cli.main(["--db", str(db_path), "evidence", "--claim", "Codex emitted evidence."]) == 0
    capsys.readouterr()
    assert (
        cli.main(
            [
                "--db",
                str(db_path),
                "value",
                "--subject",
                "runtime:codex",
                "--predicate",
                "shared_brain",
                "--bool",
                "true",
                "--status",
                "current",
                "--inject",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert cli.main(["--db", str(db_path), "--pretty", "digest"]) == 0
    payload = json.loads(capsys.readouterr().out)

    conn = connect(db_path)
    init_db(conn)
    summary = counts(conn)
    assert summary["evidence"] == 1
    assert summary["knowledge"] == 1
    assert conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0] == 1
    assert payload["counts"]["knowledge"] == 1


def test_human_gated_knowledge_proposal_and_stale(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    evidence_id = upsert_evidence(
        conn,
        source_type="loop_iteration",
        source_uri="/tmp/result.json",
        content_hash="hash-capability-result",
        claim="Repeated verified success suggests a reusable test workflow.",
        verifier_status="passed",
    )
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="capability",
        gate="human",
        slug="verified-test-workflow",
        title="Verified test workflow",
        body_uri="/tmp/result.json",
        status="candidate",
        risk="high",
        confidence=0.82,
    )
    link_knowledge_evidence(conn, knowledge_id, evidence_id, relation="derived_from")
    conn.commit()

    proposal = write_proposal(conn, knowledge_id, tmp_path / "proposals")
    content = proposal.read_text(encoding="utf-8")
    assert "object_kind: knowledge" in content
    assert "Human-gated. Do not auto-apply." in content
    assert "Repeated verified success" in content

    assert mark_knowledge_stale(conn, knowledge_id)
    row = conn.execute("SELECT status FROM knowledge WHERE id = ?", (knowledge_id,)).fetchone()
    assert row["status"] == "stale"


def test_excerpt_reads_injected_current_knowledge(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime:codex",
        predicate="shared_brain",
        value_bool=True,
        status="current",
        inject=True,
    )
    conn.commit()

    output = tmp_path / "AGENTS.md"
    write_excerpt(conn, output, runtime="codex", scope=None, limit=5)
    text = output.read_text(encoding="utf-8")

    assert "BEGIN OCBRAIN MANAGED BLOCK" in text
    assert knowledge_id in text
    assert "edit source knowledge" in text


def test_prune_marks_unrefreshed_unreferenced_knowledge_stale(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    now = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    old = (now - timedelta(days=45)).isoformat()
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime:codex",
        predicate="stale_fact",
        value_text="old",
        status="current",
    )
    conn.execute("UPDATE knowledge SET updated_at = ? WHERE id = ?", (old, knowledge_id))
    conn.commit()

    result = prune_knowledge(conn, ttl_days=30, now=now)
    conn.commit()
    row = conn.execute("SELECT status, invalidation_reason FROM knowledge").fetchone()

    assert result.changed == 1
    assert row["status"] == "stale"
    assert row["invalidation_reason"] == "stale"
    assert conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0] == 1


def test_heal_supersedes_conflicting_current_values_with_evidence(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    timestamp = "2026-06-23T12:00:00+00:00"
    winner_id = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="loop:repo-quality-loop",
        predicate="typecheck_errors",
        value_numeric=9,
        status="current",
        confidence=0.9,
    )
    conn.execute(
        """
        INSERT INTO knowledge (
          id, type, subject, predicate, value_numeric, status, gate,
          confidence, privacy_scope, created_at, updated_at
        )
        VALUES (
          'know_conflict_loser', 'value', 'loop:repo-quality-loop',
          'typecheck_errors', 17, 'current', 'auto', 0.4, 'workspace', ?, ?
        )
        """,
        (timestamp, timestamp),
    )
    conn.commit()

    result = heal_conflicts(
        conn,
        numeric_threshold=1.0,
        now=datetime(2026, 6, 23, 12, 0, tzinfo=UTC),
    )
    conn.commit()
    loser = conn.execute(
        "SELECT status, superseded_by, invalidation_reason FROM knowledge WHERE id = ?",
        ("know_conflict_loser",),
    ).fetchone()

    assert result.changed == 1
    assert loser["status"] == "superseded"
    assert loser["superseded_by"] == winner_id
    assert loser["invalidation_reason"] == "contradicted"
    correction_count = conn.execute(
        "SELECT COUNT(*) FROM evidence WHERE source_type = 'correction'"
    ).fetchone()[0]
    assert correction_count == 1


def test_liveness_check_reads_runner_ledger_and_writes_tripwire_evidence(tmp_path: Path) -> None:
    runner_path = tmp_path / "runner.sqlite"
    runner = connect(runner_path)
    init_db(runner)
    runner.execute(
        """
        INSERT INTO loop_liveness (
          loop_id, run_id, last_heartbeat_at, last_ledger_write_at,
          expected_interval_seconds, deadman_due_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "repo-quality-loop",
            "2026-06-23-nightly",
            None,
            None,
            300,
            "2026-06-23T11:55:00+00:00",
        ),
    )
    runner.commit()
    runner.close()

    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    result = check_loop_liveness(
        conn,
        runner_ledger=runner_path,
        now=datetime(2026, 6, 23, 12, 0, tzinfo=UTC),
    )
    conn.commit()
    tripwires = {
        json.loads(row["loop_tags"])["tripwire"]
        for row in conn.execute(
            "SELECT loop_tags FROM evidence WHERE source_type = 'loop_tripwire'"
        )
    }

    assert result.changed == 2
    assert {"heartbeat_starved", "no_ledger_writes"} <= tripwires
