from pathlib import Path

from ocbrain import cli
from ocbrain.db import connect, counts, init_db, list_candidates
from ocbrain.ingest import IngestOptions, event_from_file


def test_ingest_triage_and_propose(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    artifact = tmp_path / "brief.md"
    artifact.write_text(
        "# OpenClawBrain Brief\n\nArchitecture with MCP, memory, wiki, skills, and policy.\n",
        encoding="utf-8",
    )

    assert cli.main(["--db", str(db_path), "ingest", str(artifact)]) == 0
    assert cli.main(["--db", str(db_path), "triage"]) == 0

    conn = connect(db_path)
    init_db(conn)
    summary = counts(conn)
    assert summary["events"] == 1
    assert summary["candidates"] >= 1

    candidate_id = list_candidates(conn, target="wiki", limit=1)[0]["id"]
    proposal_dir = tmp_path / "proposals"
    assert (
        cli.main(["--db", str(db_path), "propose", candidate_id, "--output-dir", str(proposal_dir)])
        == 0
    )
    assert list(proposal_dir.glob("wiki-*.md"))

    excerpt = tmp_path / "AGENTS.md"
    assert (
        cli.main(["--db", str(db_path), "excerpt", "--runtime", "codex", "--output", str(excerpt)])
        == 0
    )
    assert "BEGIN OCBRAIN MANAGED BLOCK" in excerpt.read_text(encoding="utf-8")


def test_ingest_redacts_secret_like_values(tmp_path: Path) -> None:
    artifact = tmp_path / "note.md"
    artifact.write_text("api_key = should_not_survive\n", encoding="utf-8")

    event = event_from_file(artifact, IngestOptions())

    assert event is not None
    assert "should_not_survive" not in event.body
    assert "[REDACTED]" in event.body


def test_closeout_store_classifies_redacted_text(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    artifact = tmp_path / "policy.md"
    artifact.write_text(
        "# Rule\n\nNever store api_key = should_not_survive in evidence.\n",
        encoding="utf-8",
    )

    assert cli.main(["--db", str(db_path), "closeout", "--input", str(artifact), "--store"]) == 0

    conn = connect(db_path)
    init_db(conn)
    rows = list_candidates(conn, target="policy", limit=5)
    assert rows
    assert "should_not_survive" not in rows[0]["evidence_json"]
    assert "[REDACTED]" in rows[0]["evidence_json"]


def test_search_handles_punctuation_query(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    artifact = tmp_path / "brief.md"
    artifact.write_text("# Brief\n\nArchitecture uses MCP search.\n", encoding="utf-8")

    assert cli.main(["--db", str(db_path), "ingest", str(artifact)]) == 0
    assert cli.main(["--db", str(db_path), "search", '"architecture:" NEAR/mcp']) == 0
