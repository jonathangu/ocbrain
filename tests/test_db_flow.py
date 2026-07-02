import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ocbrain import cli
from ocbrain.db import (
    connect,
    counts,
    get_current_doc,
    init_db,
    knowledge_digest,
    link_knowledge_evidence,
    log_retrieval_use,
    mark_knowledge_stale,
    search,
    upsert_evidence,
    upsert_knowledge,
)
from ocbrain.excerpt import write_excerpt
from ocbrain.maintenance import check_loop_liveness, heal_conflicts, prune_knowledge
from ocbrain.proposals import write_proposal
from ocbrain.text import find_probable_secret_leaks, redact_secrets


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


_CANONICAL_RETRIEVAL_COLUMNS = {
    "id",
    "knowledge_id",
    "served_to_runtime",
    "task_ref",
    "affected_decision",
    "corrected",
    "outcome",
    "note",
    "served_at",
}

# The exact live legacy retrieval_uses shape: the original 7 columns plus the
# 6 ALTER-added columns. init_db's CREATE TABLE IF NOT EXISTS won't reshape it.
_LEGACY_RETRIEVAL_DDL = """
CREATE TABLE {name} (
  id TEXT PRIMARY KEY,
  artifact_or_candidate_id TEXT NOT NULL,
  runtime TEXT,
  query TEXT,
  outcome TEXT,
  note TEXT,
  created_at TEXT NOT NULL
, knowledge_id TEXT, served_to_runtime TEXT, task_ref TEXT,
  affected_decision INTEGER, corrected INTEGER, served_at TEXT)
"""

# Canonical shape, matching SCHEMA, used to stand up the half-migrated State B case.
_CANONICAL_RETRIEVAL_DDL = """
CREATE TABLE retrieval_uses (
  id TEXT PRIMARY KEY,
  knowledge_id TEXT REFERENCES knowledge(id),
  served_to_runtime TEXT,
  task_ref TEXT,
  affected_decision INTEGER,
  corrected INTEGER,
  outcome TEXT CHECK (
    outcome IN (
      'improved','failed','neutral','unknown',
      'served','helpful','used','irrelevant','ignored','harmful'
    )
  ) DEFAULT 'unknown',
  note TEXT,
  served_at TEXT NOT NULL
)
"""


def test_init_db_migrates_legacy_retrieval_uses(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    # State A: a live legacy table holding rows including one with an outcome
    # ('included') that is NOT in the canonical CHECK set, plus NULL knowledge_id and
    # NULL served_at. A naive COALESCE(outcome,'unknown') copy would trip the CHECK.
    conn = sqlite3.connect(db_path)
    conn.execute(_LEGACY_RETRIEVAL_DDL.format(name="retrieval_uses"))
    conn.executemany(
        """
        INSERT INTO retrieval_uses (id, artifact_or_candidate_id, runtime, outcome, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("ret_legacy_1", "cand_1", "codex", "served", "2026-06-01T00:00:00+00:00"),
            ("ret_legacy_2", "evt_2", "claude_code", "helpful", "2026-06-02T00:00:00+00:00"),
            # outcome='included' is invalid; knowledge_id/served_at NULL.
            ("ret_legacy_3", "cand_3", "mcp", "included", "2026-06-03T00:00:00+00:00"),
        ],
    )
    conn.commit()
    conn.row_factory = sqlite3.Row

    init_db(conn)

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(retrieval_uses)")}
    assert "artifact_or_candidate_id" not in columns
    assert columns == _CANONICAL_RETRIEVAL_COLUMNS

    rows = {
        row["id"]: row
        for row in conn.execute(
            "SELECT id, knowledge_id, served_to_runtime, outcome, served_at FROM retrieval_uses"
        )
    }
    # All rows preserved.
    assert set(rows) == {"ret_legacy_1", "ret_legacy_2", "ret_legacy_3"}
    # served_at backfilled from created_at; served_to_runtime backfilled from runtime.
    assert rows["ret_legacy_1"]["served_at"] == "2026-06-01T00:00:00+00:00"
    assert rows["ret_legacy_1"]["served_to_runtime"] == "codex"
    # knowledge_id backfilled from artifact_or_candidate_id.
    assert rows["ret_legacy_1"]["knowledge_id"] == "cand_1"
    assert rows["ret_legacy_3"]["knowledge_id"] == "cand_3"
    # served_at backfilled even when the legacy served_at column was NULL.
    assert rows["ret_legacy_3"]["served_at"] == "2026-06-03T00:00:00+00:00"
    # Valid outcomes are preserved verbatim.
    assert rows["ret_legacy_1"]["outcome"] == "served"
    assert rows["ret_legacy_2"]["outcome"] == "helpful"
    # The invalid 'included' outcome is sanitized to 'unknown' (not dropped).
    assert rows["ret_legacy_3"]["outcome"] == "unknown"

    # The exact path that was failing on legacy DBs now succeeds.
    log_retrieval_use(conn, None, runtime="mcp", task_ref="brain.digest", outcome="served")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM retrieval_uses").fetchone()[0] == 4

    # Idempotent: a second init_db is a no-op and does not error.
    init_db(conn)
    columns_again = {row["name"] for row in conn.execute("PRAGMA table_info(retrieval_uses)")}
    assert "artifact_or_candidate_id" not in columns_again
    assert conn.execute("SELECT COUNT(*) FROM retrieval_uses").fetchone()[0] == 4


def test_init_db_recovers_half_migrated_retrieval_uses(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    # State B: an earlier rebuild crashed mid-copy, leaving a canonical (empty)
    # retrieval_uses alongside a stray retrieval_uses_legacy that still holds the
    # real rows, including one with the invalid 'included' outcome.
    conn = sqlite3.connect(db_path)
    conn.execute(_CANONICAL_RETRIEVAL_DDL)
    conn.execute(_LEGACY_RETRIEVAL_DDL.format(name="retrieval_uses_legacy"))
    conn.executemany(
        """
        INSERT INTO retrieval_uses_legacy
          (id, artifact_or_candidate_id, runtime, outcome, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("ret_stray_1", "cand_1", "codex", "served", "2026-06-01T00:00:00+00:00"),
            ("ret_stray_2", "evt_2", "claude_code", "included", "2026-06-02T00:00:00+00:00"),
        ],
    )
    conn.commit()
    conn.row_factory = sqlite3.Row

    init_db(conn)

    names = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    # The stray table is dropped after recovery.
    assert "retrieval_uses_legacy" not in names
    assert "retrieval_uses_new" not in names

    rows = {
        row["id"]: row
        for row in conn.execute(
            "SELECT id, knowledge_id, served_to_runtime, outcome, served_at FROM retrieval_uses"
        )
    }
    # Rows recovered into the canonical table, with sanitization applied.
    assert set(rows) == {"ret_stray_1", "ret_stray_2"}
    assert rows["ret_stray_1"]["outcome"] == "served"
    assert rows["ret_stray_2"]["outcome"] == "unknown"
    assert rows["ret_stray_1"]["knowledge_id"] == "cand_1"
    assert rows["ret_stray_1"]["served_at"] == "2026-06-01T00:00:00+00:00"

    # Idempotent: a second init_db neither re-creates the stray table nor dupes rows.
    init_db(conn)
    assert conn.execute("SELECT COUNT(*) FROM retrieval_uses").fetchone()[0] == 2
    names_again = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert "retrieval_uses_legacy" not in names_again


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


def test_import_memory_makes_markdown_searchable_and_digestible(
    tmp_path: Path, capsys
) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    memory_path = tmp_path / "memory" / "2026-06-28.md"
    memory_path.parent.mkdir()
    memory_path.write_text(
        "# OCBrain product check\n\n"
        "The actual ocbrain product must return source-backed memory, not empty arrays.\n",
        encoding="utf-8",
    )

    assert (
        cli.main(
            [
                "--db",
                str(db_path),
                "import-memory",
                str(memory_path),
                "--project",
                "ocbrain",
                "--privacy-scope",
                "workspace",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["imported"] == 1
    assert payload["counts"]["evidence"] == 1
    assert payload["counts"]["knowledge"] == 1

    assert (
        cli.main(
            [
                "--db",
                str(db_path),
                "search",
                "source-backed memory",
                "--project",
                "ocbrain",
                "--type",
                "doc",
            ]
        )
        == 0
    )
    search_payload = json.loads(capsys.readouterr().out)
    assert search_payload["results"][0]["kind"] == "knowledge:doc"
    assert "source-backed" in search_payload["results"][0]["snippet"]

    assert (
        cli.main(
            [
                "--db",
                str(db_path),
                "digest",
                "--project",
                "ocbrain",
            ]
        )
        == 0
    )
    digest_payload = json.loads(capsys.readouterr().out)
    assert digest_payload["documents"][0]["title"] == "OCBrain product check"


def test_event_backfill_batches_past_already_projected_rows(
    tmp_path: Path, capsys
) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    conn = connect(db_path)
    init_db(conn)
    upsert_knowledge(
        conn,
        knowledge_type="doc",
        gate="auto",
        slug="bountiful-closeout",
        title="Bountiful closeout",
        body_uri="task-artifacts/bountiful.md",
        doc_kind="closeout",
        status="current",
        confidence=0.82,
        project="bountiful",
        privacy_scope="project",
    )
    upsert_knowledge(
        conn,
        knowledge_type="doc",
        gate="auto",
        slug="ocbrain-closeout",
        title="OCBrain closeout",
        body_uri="task-artifacts/ocbrain.md",
        doc_kind="closeout",
        status="current",
        confidence=0.84,
        project="ocbrain",
        privacy_scope="project",
    )
    conn.commit()
    conn.close()

    command = ["--db", str(db_path), "event-backfill", "--limit", "1"]
    assert cli.main(command) == 0
    first_payload = json.loads(capsys.readouterr().out)
    assert first_payload["imported"] == 1

    assert cli.main(command) == 0
    second_payload = json.loads(capsys.readouterr().out)
    assert second_payload["imported"] == 1
    assert second_payload["items"][0]["knowledge_id"] != first_payload["items"][0]["knowledge_id"]

    assert cli.main(command + ["--dry-run"]) == 0
    dry_payload = json.loads(capsys.readouterr().out)
    assert dry_payload["dry_run"] is True
    assert dry_payload["would_import"] == 0
    conn = connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM current_beliefs").fetchone()[0] == 2


def test_event_backfill_compilation_references_source_evidence_id(
    tmp_path: Path, capsys
) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    conn = connect(db_path)
    init_db(conn)
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="doc",
        gate="auto",
        slug="bountiful-stack",
        title="Bountiful stack",
        body_uri="task-artifacts/bountiful-stack.md",
        doc_kind="closeout",
        status="current",
        confidence=0.88,
        project="bountiful",
        privacy_scope="project",
    )
    conn.commit()
    conn.close()

    assert cli.main(["--db", str(db_path), "event-backfill", "--limit", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    evidence_id = payload["items"][0]["evidence_id"]
    assert evidence_id.startswith("evd_")

    conn = connect(db_path)
    proposed = conn.execute(
        "SELECT body_json FROM brain_events WHERE kind = 'compilation_proposed'"
    ).fetchone()
    proposed_body = json.loads(proposed["body_json"])
    assert proposed_body["belief_id"] == f"legacy:{knowledge_id}"
    assert proposed_body["evidence_ids"] == [evidence_id]
    assert not proposed_body["evidence_ids"][0].startswith("evt_")


def test_event_backfill_all_classifies_and_rebuilds_once(
    tmp_path: Path, capsys
) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    conn = connect(db_path)
    init_db(conn)
    for slug, title, project in [
        ("bountiful-closeout", "Bountiful closeout", "workspace"),
        ("pelican-plan", "Pelican plan", "workspace"),
        ("ocbrain-guide", "OCBrain guide", "ocbrain"),
    ]:
        upsert_knowledge(
            conn,
            knowledge_type="doc",
            gate="auto",
            slug=slug,
            title=title,
            body_uri=f"task-artifacts/{slug}.md",
            doc_kind="closeout",
            status="current",
            confidence=0.8,
            project=project,
            privacy_scope="workspace",
        )
    conn.commit()
    conn.close()

    assert cli.main(["--db", str(db_path), "event-backfill", "--all"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["imported"] == 3
    assert payload["scope_counts"] == {
        "personal_finance:pelican": 1,
        "project:bountiful": 1,
        "project:ocbrain": 1,
    }
    assert all(item["classification_reason"] for item in payload["items"])
    conn = connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM current_beliefs").fetchone()[0] == 3


def test_import_history_catalogs_runtime_transcripts(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    codex_path = tmp_path / ".codex" / "sessions" / "2026" / "06" / "rollout.jsonl"
    openclaw_path = tmp_path / ".openclaw" / "agents" / "main" / "sessions" / "turn.jsonl"
    claude_path = tmp_path / ".claude" / "projects" / "workspace" / "session.jsonl"
    for path, runtime in [
        (codex_path, "codex"),
        (openclaw_path, "openclaw"),
        (claude_path, "claude"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "type": "message",
                    "runtime": runtime,
                    "content": f"{runtime} transcript contains harvest sentinel",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    assert (
        cli.main(
            [
                "--db",
                str(db_path),
                "import-history",
                str(tmp_path / ".codex" / "sessions"),
                str(tmp_path / ".openclaw" / "agents"),
                str(tmp_path / ".claude" / "projects"),
                "--privacy-scope",
                "workspace",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["imported"] == 3
    assert payload["existing"] == 0
    assert payload["by_runtime"] == {"claude": 1, "codex": 1, "openclaw": 1}
    assert payload["counts"]["evidence"] == 3
    assert payload["counts"]["knowledge"] == 3

    assert (
        cli.main(
            [
                "--db",
                str(db_path),
                "search",
                "harvest sentinel",
                "--type",
                "doc",
            ]
        )
        == 0
    )
    search_payload = json.loads(capsys.readouterr().out)
    assert {item["kind"] for item in search_payload["results"]} == {"knowledge:doc"}
    assert {item["scope"] for item in search_payload["results"]} == {"workspace"}
    assert len(search_payload["results"]) == 3


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
    assert "Surface assumptions or ambiguity before acting." in text
    assert "Prefer the smallest change that satisfies the verified goal." in text
    assert "Keep edits surgical; do not refactor unrelated code." in text
    assert "Verify the result and record the evidence." in text


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


def test_prune_archives_stale_rows_without_deleting_them(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    now = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    old = (now - timedelta(days=120)).isoformat()
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime:codex",
        predicate="archive_fact",
        value_text="old",
        status="current",
    )
    conn.execute(
        "UPDATE knowledge SET status = 'stale', updated_at = ? WHERE id = ?",
        (old, knowledge_id),
    )
    conn.commit()

    result = prune_knowledge(conn, archive_stale_days=90, now=now)
    conn.commit()
    row = conn.execute("SELECT status FROM knowledge WHERE id = ?", (knowledge_id,)).fetchone()

    assert result.changed == 1
    assert row["status"] == "archived"
    assert conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0] == 1


def test_prune_decays_served_but_never_useful_knowledge_faster(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    now = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    old = (now - timedelta(days=15)).isoformat()
    useless_id = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime:codex",
        predicate="ignored_fact",
        value_text="ignored",
        status="current",
    )
    useful_id = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime:codex",
        predicate="useful_fact",
        value_text="useful",
        status="current",
    )
    log_retrieval_use(conn, useless_id, outcome="ignored")
    log_retrieval_use(conn, useful_id, outcome="helpful")
    conn.execute("UPDATE knowledge SET updated_at = ?", (old,))
    conn.commit()

    result = prune_knowledge(conn, ttl_days=30, unhelpful_ttl_days=7, now=now)
    conn.commit()
    statuses = {
        row["id"]: row["status"]
        for row in conn.execute("SELECT id, status FROM knowledge ORDER BY id")
    }

    assert result.changed == 1
    assert statuses[useless_id] == "stale"
    assert statuses[useful_id] == "current"
    assert result.details[0]["reason"] == "ttl_served_without_usefulness"


def test_private_evidence_tightens_public_doc_scope(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    evidence_id = upsert_evidence(
        conn,
        source_type="closeout",
        source_uri="/private/source.md",
        content_hash="private-hash",
        claim="Private source backs a derived doc.",
        privacy_scope="private",
    )
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="doc",
        gate="auto",
        slug="public-derived-doc",
        title="Public Derived Doc",
        body_uri="/tmp/public.md",
        doc_kind="wiki",
        status="current",
        privacy_scope="public",
    )
    link_knowledge_evidence(conn, knowledge_id, evidence_id)
    conn.commit()

    row = conn.execute(
        "SELECT privacy_scope FROM knowledge WHERE id = ?",
        (knowledge_id,),
    ).fetchone()
    digest = knowledge_digest(conn)

    assert row["privacy_scope"] == "private"
    assert not any(item["id"] == knowledge_id for item in digest["documents"])
    assert get_current_doc(conn, slug="public-derived-doc") is None


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


def test_heal_leaves_numeric_values_within_threshold_current(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    timestamp = "2026-06-23T12:00:00+00:00"
    upsert_knowledge(
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
          'know_near_value', 'value', 'loop:repo-quality-loop',
          'typecheck_errors', 9.2, 'current', 'auto', 0.4, 'workspace', ?, ?
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
    statuses = {
        row["id"]: row["status"]
        for row in conn.execute("SELECT id, status FROM knowledge ORDER BY id")
    }

    assert result.changed == 0
    assert set(statuses.values()) == {"current"}
    correction_count = conn.execute(
        "SELECT COUNT(*) FROM evidence WHERE source_type = 'correction'"
    ).fetchone()[0]
    assert correction_count == 0


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


def _write_history_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_redact_secrets_covers_json_and_assignment_shapes() -> None:
    cases = {
        '{"api_key": "hunter2-super-secret"}': "hunter2-super-secret",
        '{"ANTHROPIC_API_KEY": "sk-ant-api03-abcdef123456"}': "sk-ant-api03",
        'export OPENAI_API_KEY="sk-proj-abc"': "sk-proj-abc",
        'password = "hunter2"': "hunter2",
        "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG": "wJalrXUtnFEMI",
        '{\\"api_key\\": \\"zq9-topsecret-hunter2\\"}': "zq9-topsecret",
        '{"accessToken": "sk-ant-oat01-verysecret1234"}': "sk-ant-oat01",
        '{"refreshToken": "rt-999-refresh-secret"}': "rt-999-refresh-secret",
        '{"apiKey": "camelCaseSecretValue"}': "camelCaseSecretValue",
        '"Authorization": "Bearer abcdef0123456789abcdef"': "abcdef0123456789abcdef",
        "sk-ant-api03-longtokenvalue1234567890": "longtokenvalue",
    }
    for text, secret in cases.items():
        redacted = redact_secrets(text)
        assert secret not in redacted, text
        assert "[REDACTED]" in redacted, text
        assert find_probable_secret_leaks(text), text
        assert not find_probable_secret_leaks(redacted), text
    benign = "no secrets here, just a sentence about tokens of appreciation"
    assert redact_secrets("plain prose stays untouched") == "plain prose stays untouched"
    assert find_probable_secret_leaks(benign) == []


def test_import_history_defaults_to_private_scope(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    _write_history_file(
        tmp_path / ".claude" / "projects" / "ws" / "sess.jsonl",
        json.dumps({"content": "private harvest sentinel"}) + "\n",
    )

    assert (
        cli.main(
            ["--db", str(db_path), "import-history", str(tmp_path / ".claude" / "projects")]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["imported"] == 1

    conn = connect(db_path)
    scopes = {row["privacy_scope"] for row in conn.execute("SELECT privacy_scope FROM evidence")}
    assert scopes == {"private"}
    scopes = {row["privacy_scope"] for row in conn.execute("SELECT privacy_scope FROM knowledge")}
    assert scopes == {"private"}
    conn.close()

    # Default (non --include-private) search must not serve the transcript.
    assert cli.main(["--db", str(db_path), "search", "private harvest sentinel"]) == 0
    assert json.loads(capsys.readouterr().out)["results"] == []
    assert (
        cli.main(
            ["--db", str(db_path), "search", "private harvest sentinel", "--include-private"]
        )
        == 0
    )
    assert len(json.loads(capsys.readouterr().out)["results"]) > 0


def test_import_memory_defaults_to_private_scope(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    memory_path = tmp_path / "memory" / "note.md"
    _write_history_file(memory_path, "# Note\n\nprivate memory sentinel\n")

    assert cli.main(["--db", str(db_path), "import-memory", str(memory_path)]) == 0
    assert json.loads(capsys.readouterr().out)["imported"] == 1

    conn = connect(db_path)
    scopes = {row["privacy_scope"] for row in conn.execute("SELECT privacy_scope FROM knowledge")}
    assert scopes == {"private"}
    conn.close()

    assert cli.main(["--db", str(db_path), "search", "private memory sentinel"]) == 0
    assert json.loads(capsys.readouterr().out)["results"] == []


@pytest.mark.parametrize(
    "argv",
    [
        ["evidence", "--claim", "x", "--privacy-scope", "Private"],
        ["value", "--subject", "s", "--predicate", "p", "--text", "v", "--privacy-scope", "secret"],
        ["import-memory", "note.md", "--privacy-scope", "personal"],
        ["import-history", ".claude", "--privacy-scope", "Workspace"],
    ],
)
def test_privacy_scope_flags_reject_invalid_values(tmp_path: Path, argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--db", str(tmp_path / "ocbrain.sqlite"), *argv])
    assert excinfo.value.code == 2


def test_import_history_sweep_skips_credentials_and_hidden_files(
    tmp_path: Path, capsys
) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    root = tmp_path / ".codex"
    _write_history_file(root / "sessions" / "rollout.jsonl", '{"content": "safe sentinel"}\n')
    _write_history_file(root / "auth.json", '{"OPENAI_API_KEY": "sk-proj-swept-credential"}')
    _write_history_file(root / "sessions" / "settings.json", '{"theme": "dark"}')
    _write_history_file(
        root / ".hidden" / "session.jsonl", '{"content": "dotdir transcript"}\n'
    )
    _write_history_file(root / "sessions" / ".env.json", '{"TOKEN": "abc"}')

    assert cli.main(["--db", str(db_path), "import-history", str(root)]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["imported"] == 1
    assert payload["counts"]["evidence"] == 1
    reasons = {item["path"]: item["reason"] for item in payload["skipped"]}
    assert reasons[str((root / "auth.json").resolve())] == "sensitive_filename"
    assert reasons[str((root / "sessions" / "settings.json").resolve())] == "sensitive_filename"
    assert reasons[str((root / ".hidden" / "session.jsonl").resolve())] == "hidden_path"
    assert reasons[str((root / "sessions" / ".env.json").resolve())] == "hidden_path"
    assert payload["skipped_count"] == 4

    conn = connect(db_path)
    stored = " ".join(row["claim"] for row in conn.execute("SELECT claim FROM evidence"))
    assert "sk-proj-swept-credential" not in stored
    assert "safe sentinel" in stored


def test_import_history_reharvest_is_idempotent_across_path_spellings(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    _write_history_file(
        tmp_path / ".claude" / "projects" / "ws" / "sess.jsonl",
        '{"content": "idempotent sentinel"}\n',
    )
    monkeypatch.chdir(tmp_path)

    assert cli.main(["--db", str(db_path), "import-history", ".claude/projects"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["imported"] == 1

    # Same tree via a different spelling: absolute path instead of relative.
    absolute = str(tmp_path / ".claude" / "projects")
    assert cli.main(["--db", str(db_path), "import-history", absolute]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["imported"] == 0
    assert second["existing"] == 1
    assert second["counts"]["evidence"] == 1
    assert second["counts"]["knowledge"] == 1


def test_import_history_redacts_before_windowing(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    key_body = "SECRETKEYMATERIAL" * 120
    pem = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        f"{key_body}\n"
        "-----END OPENSSH PRIVATE KEY-----"
    )
    # The key starts inside the head window and ends beyond it: the old
    # truncate-then-redact order stored the key body verbatim.
    content = ("x" * 9500) + "\n" + pem + "\n" + ("y" * 15000)
    _write_history_file(tmp_path / ".claude" / "projects" / "ws" / "big.jsonl", content)

    assert (
        cli.main(
            [
                "--db",
                str(db_path),
                "import-history",
                str(tmp_path / ".claude" / "projects"),
                "--max-bytes",
                "20000",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["imported"] == 1

    conn = connect(db_path)
    bodies = " ".join(row["body"] for row in conn.execute("SELECT body FROM search_index"))
    claims = " ".join(row["claim"] for row in conn.execute("SELECT claim FROM evidence"))
    assert "SECRETKEYMATERIAL" not in bodies
    assert "SECRETKEYMATERIAL" not in claims
    assert "[REDACTED_PRIVATE_KEY]" in bodies
    assert "bytes omitted from middle" in bodies


def test_import_history_dry_run_writes_nothing_and_reports_leaks(
    tmp_path: Path, capsys
) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    _write_history_file(
        tmp_path / ".claude" / "projects" / "ws" / "sess.jsonl",
        '{"api_key": "dryrun-secret-value", "content": "dry sentinel"}\n',
    )
    _write_history_file(tmp_path / ".claude" / "auth.json", "{}")

    assert (
        cli.main(
            [
                "--db",
                str(db_path),
                "import-history",
                str(tmp_path / ".claude"),
                "--dry-run",
                "--event-core",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["would_import"] == 1
    assert payload["by_runtime"] == {"claude": 1}
    assert payload["skipped_count"] == 1
    assert payload["secret_leak_count"] == 1
    assert payload["secret_leaks"][0]["leaks"] == ["json_quoted_secret"]
    assert payload["counts"]["evidence"] == 0
    assert payload["counts"]["knowledge"] == 0

    conn = connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM search_index").fetchone()[0] == 0


def test_import_memory_dry_run_writes_nothing(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    memory_path = tmp_path / "memory" / "note.md"
    _write_history_file(memory_path, "password = \"hunter2\"\n")

    assert (
        cli.main(
            ["--db", str(db_path), "import-memory", str(memory_path), "--dry-run"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["would_import"] == 1
    assert payload["secret_leak_count"] == 1
    assert "assigned_secret" in payload["secret_leaks"][0]["leaks"]

    conn = connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0] == 0


def test_import_memory_event_core_emits_scoped_deduped_events(
    tmp_path: Path, capsys
) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    memory_path = tmp_path / "memory" / "2026-07-01.md"
    _write_history_file(memory_path, "# Daily\n\nevent core sentinel apiKey: \"topsecret99\"\n")

    command = ["--db", str(db_path), "import-memory", str(memory_path), "--event-core"]
    assert cli.main(command) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["imported"] == 1
    assert first["event_core"] == {"events_appended": 1, "deduped": 0}
    evidence_id = first["files"][0]["event_core_evidence_id"]
    assert evidence_id.startswith("evd_")

    # Re-import: same content-derived evidence id must be deduped, not re-appended.
    assert cli.main(command) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["event_core"] == {"events_appended": 0, "deduped": 1}
    assert second["files"][0]["event_core_evidence_id"] == evidence_id

    conn = connect(db_path)
    events = list(
        conn.execute(
            "SELECT writer, body_json FROM brain_events WHERE kind = 'evidence_recorded'"
        )
    )
    assert len(events) == 1
    assert events[0]["writer"] == "harvest:openclaw"
    body = json.loads(events[0]["body_json"])
    assert body["evidence_id"] == evidence_id
    assert body["kind"] == "openclaw_memory"
    assert body["artifact_ref"] == str(memory_path.resolve())
    assert len(body["body"]) <= 900
    assert "topsecret99" not in body["body"]
    assert body["scope"]["scope_type"] == "session"
    assert body["scope"]["scope_id"].startswith("session:")
    assert body["scope"]["visibility"] == "confidential"
    assert body["scope"]["egress_policy"] == "local_only"


def test_import_history_event_core_uses_runtime_writer_and_kind(
    tmp_path: Path, capsys
) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    history_path = tmp_path / ".codex" / "sessions" / "rollout.jsonl"
    _write_history_file(history_path, '{"content": "codex event sentinel"}\n')

    assert (
        cli.main(
            [
                "--db",
                str(db_path),
                "import-history",
                str(tmp_path / ".codex"),
                "--event-core",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["event_core"] == {"events_appended": 1, "deduped": 0}

    conn = connect(db_path)
    event = conn.execute(
        "SELECT writer, body_json FROM brain_events WHERE kind = 'evidence_recorded'"
    ).fetchone()
    assert event["writer"] == "harvest:codex"
    body = json.loads(event["body_json"])
    assert body["kind"] == "codex_history"
    assert body["artifact_ref"] == str(history_path.resolve())
    assert body["scope"]["visibility"] == "confidential"
    assert body["scope"]["egress_policy"] == "local_only"


def test_import_memory_distinct_paths_with_same_tail_do_not_collide(
    tmp_path: Path, capsys
) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    for clone, title in [("cloneA", "Alpha notes"), ("cloneB", "Beta notes")]:
        _write_history_file(
            tmp_path / clone / "docs" / "memory" / "notes.md", f"# {title}\n\nbody\n"
        )

    assert (
        cli.main(
            [
                "--db",
                str(db_path),
                "import-memory",
                str(tmp_path / "cloneA" / "docs" / "memory"),
                str(tmp_path / "cloneB" / "docs" / "memory"),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["imported"] == 2
    assert payload["counts"]["knowledge"] == 2

    conn = connect(db_path)
    titles = {row["title"] for row in conn.execute("SELECT title FROM knowledge")}
    assert titles == {"Alpha notes", "Beta notes"}
    slugs = {row["slug"] for row in conn.execute("SELECT slug FROM knowledge")}
    assert len(slugs) == 2


def test_harvest_commands_do_not_disable_sqlite_durability() -> None:
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert "synchronous=OFF" not in source
    assert "synchronous=NORMAL" in source
