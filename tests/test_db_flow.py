import json
from pathlib import Path

from ocbrain import cli
from ocbrain.db import (
    EventInput,
    connect,
    counts,
    init_db,
    insert_candidate,
    list_candidates,
    search,
    upsert_event,
)
from ocbrain.ingest import IngestOptions, event_from_file
from ocbrain.schema import Candidate, Evidence, Target


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
        cli.main(
            [
                "--db",
                str(db_path),
                "propose",
                candidate_id,
                "--output-dir",
                str(proposal_dir),
                "--allow-draft",
            ]
        )
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
    assert rows[0]["claim_key"]
    assert "should_not_survive" not in rows[0]["evidence_json"]
    assert "[REDACTED]" in rows[0]["evidence_json"]


def test_search_handles_punctuation_query(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    artifact = tmp_path / "brief.md"
    artifact.write_text("# Brief\n\nArchitecture uses MCP search.\n", encoding="utf-8")

    assert cli.main(["--db", str(db_path), "ingest", str(artifact)]) == 0
    assert cli.main(["--db", str(db_path), "search", '"architecture:" NEAR/mcp']) == 0


def test_backfill_claim_keys_updates_older_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    artifact = tmp_path / "brief.md"
    artifact.write_text("# Brief\n\nArchitecture uses MCP search.\n", encoding="utf-8")
    assert cli.main(["--db", str(db_path), "ingest", str(artifact)]) == 0
    assert cli.main(["--db", str(db_path), "triage"]) == 0

    conn = connect(db_path)
    init_db(conn)
    candidate_id = list_candidates(conn, target="wiki", limit=1)[0]["id"]
    conn.execute("UPDATE candidates SET claim_key = '' WHERE id = ?", (candidate_id,))
    conn.commit()

    assert cli.main(["--db", str(db_path), "backfill-claim-keys"]) == 0
    row = list_candidates(conn, target="wiki", limit=1)[0]
    assert row["claim_key"]
    assert "architecture uses mcp search" in row["claim_key"]


def test_search_scope_filter_excludes_private_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    conn = connect(db_path)
    init_db(conn)
    assert upsert_event(
        conn,
        EventInput(
            id="evt_private",
            source_type="note",
            source_uri="/tmp/private.md",
            content_hash="hash-private",
            title="Private Alpha",
            summary="private alpha needle",
            body="private alpha needle",
            scope="private",
        ),
    )
    assert upsert_event(
        conn,
        EventInput(
            id="evt_workspace",
            source_type="note",
            source_uri="/tmp/workspace.md",
            content_hash="hash-workspace",
            title="Workspace Alpha",
            summary="workspace alpha needle",
            body="workspace alpha needle",
            scope="workspace",
        ),
    )
    conn.commit()

    rows = search(conn, "alpha needle", scopes=("workspace", "project", "public"))

    assert {row["doc_id"] for row in rows} == {"evt_workspace"}


def test_rebuild_candidates_stales_generic_drafts(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    conn = connect(db_path)
    init_db(conn)
    event = EventInput(
        id="evt_rebuild",
        source_type="doc",
        source_uri="/tmp/rebuild.md",
        content_hash="hash-rebuild",
        title="Rebuild",
        summary="Architecture uses MCP for shared context.",
        body="Architecture uses MCP for shared context.",
    )
    assert upsert_event(conn, event)
    old_id = insert_candidate(
        conn,
        Candidate(
            target=Target.WIKI,
            title="Old generic",
            body=(
                "Artifact appears to contain stable architecture or design synthesis. "
                "Route to wiki draft rather than long-form memory."
            ),
            confidence=0.72,
            evidence=[Evidence(uri="/tmp/rebuild.md", excerpt=event.summary)],
        ),
        event.id,
    )
    conn.commit()

    assert cli.main(["--db", str(db_path), "rebuild-candidates"]) == 0
    old_status = conn.execute(
        "SELECT status FROM candidates WHERE id = ?",
        (old_id,),
    ).fetchone()[0]
    assert old_status == "draft"

    assert cli.main(["--db", str(db_path), "rebuild-candidates", "--apply"]) == 0
    old_status = conn.execute("SELECT status FROM candidates WHERE id = ?", (old_id,)).fetchone()[0]
    new_bodies = [
        row["body"]
        for row in conn.execute(
            "SELECT body FROM candidates WHERE event_id = ? AND status = 'draft'",
            (event.id,),
        )
    ]

    assert old_status == "stale"
    assert any("Architecture uses MCP for shared context" in body for body in new_bodies)


def test_backfill_preview_uses_copy_and_reports_diff(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    conn = connect(db_path)
    init_db(conn)
    event = EventInput(
        id="evt_preview",
        source_type="doc",
        source_uri="/tmp/preview.md",
        content_hash="hash-preview",
        title="Preview",
        summary="Architecture uses MCP for compact reviewed context.",
        body="Architecture uses MCP for compact reviewed context.",
    )
    assert upsert_event(conn, event)
    old_id = insert_candidate(
        conn,
        Candidate(
            target=Target.POLICY,
            title="Old generic",
            body=(
                "Artifact appears to contain stable architecture or design synthesis. "
                "Route to wiki draft rather than long-form memory."
            ),
            confidence=0.72,
            evidence=[Evidence(uri="/tmp/preview.md", excerpt=event.summary)],
        ),
        event.id,
    )
    conn.commit()

    output_dir = tmp_path / "preview"
    assert (
        cli.main(
            [
                "--db",
                str(db_path),
                "--pretty",
                "backfill-preview",
                "--output-dir",
                str(output_dir),
                "--sample-size",
                "10",
                "--allow-score-drop",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["preview_db"].endswith("backfill-preview.sqlite")
    assert payload["rebuild"]["generic_candidates_to_stale"] == 1
    assert payload["distribution_diff"]
    assert (output_dir / "backfill-preview.json").exists()

    source_status = conn.execute("SELECT status FROM candidates WHERE id = ?", (old_id,)).fetchone()
    preview_conn = connect(output_dir / "backfill-preview.sqlite")
    init_db(preview_conn)
    preview_status = preview_conn.execute(
        "SELECT status FROM candidates WHERE id = ?",
        (old_id,),
    ).fetchone()

    assert source_status["status"] == "draft"
    assert preview_status["status"] == "stale"


def test_invalidate_temporal_records_supersession(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    conn = connect(db_path)
    init_db(conn)
    old_event = EventInput(
        id="evt_old_version",
        source_type="artifact",
        source_uri="/tmp/old.md",
        content_hash="hash-old",
        title="Old install",
        summary="OpenClawBrain installed version 0.4.40",
        body="OpenClawBrain installed version 0.4.40",
        created_at="2026-04-09T00:00:00Z",
    )
    new_event = EventInput(
        id="evt_new_version",
        source_type="artifact",
        source_uri="/tmp/new.md",
        content_hash="hash-new",
        title="New install",
        summary="OpenClawBrain installed version 0.4.42",
        body="OpenClawBrain installed version 0.4.42",
        created_at="2026-04-11T00:00:00Z",
    )
    assert upsert_event(conn, old_event)
    assert upsert_event(conn, new_event)
    old_id = insert_candidate(
        conn,
        Candidate(
            target=Target.MEMORY,
            title="Operational fact candidate: OpenClawBrain installed version",
            body="Stage operational fact from source: OpenClawBrain installed version 0.4.40",
            confidence=0.68,
            evidence=[Evidence(uri="/tmp/old.md", excerpt=old_event.summary)],
        ),
        old_event.id,
    )
    new_id = insert_candidate(
        conn,
        Candidate(
            target=Target.MEMORY,
            title="Operational fact candidate: OpenClawBrain installed version",
            body="Stage operational fact from source: OpenClawBrain installed version 0.4.42",
            confidence=0.68,
            evidence=[Evidence(uri="/tmp/new.md", excerpt=new_event.summary)],
        ),
        new_event.id,
    )
    conn.commit()

    assert cli.main(["--db", str(db_path), "--pretty", "invalidate-temporal"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidates_to_stale"] == 1
    assert payload["invalidations"][0]["old_candidate_id"] == old_id
    assert payload["invalidations"][0]["new_candidate_id"] == new_id
    assert conn.execute("SELECT status FROM candidates WHERE id = ?", (old_id,)).fetchone()[
        "status"
    ] == "draft"

    assert cli.main(["--db", str(db_path), "invalidate-temporal", "--apply"]) == 0
    assert conn.execute("SELECT status FROM candidates WHERE id = ?", (old_id,)).fetchone()[
        "status"
    ] == "stale"
    assert conn.execute("SELECT status FROM candidates WHERE id = ?", (new_id,)).fetchone()[
        "status"
    ] == "draft"
    invalidation = conn.execute("SELECT * FROM invalidations").fetchone()
    assert invalidation["old_candidate_id"] == old_id
    assert invalidation["new_candidate_id"] == new_id


def test_review_approve_gates_proposal_generation(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    artifact = tmp_path / "brief.md"
    artifact.write_text("# Brief\n\nArchitecture uses MCP search.\n", encoding="utf-8")
    assert cli.main(["--db", str(db_path), "ingest", str(artifact)]) == 0
    assert cli.main(["--db", str(db_path), "triage"]) == 0

    conn = connect(db_path)
    init_db(conn)
    candidate_id = list_candidates(conn, target="wiki", limit=1)[0]["id"]
    proposal_dir = tmp_path / "proposals"

    try:
        cli.main(["--db", str(db_path), "propose", candidate_id, "--output-dir", str(proposal_dir)])
    except PermissionError:
        pass
    else:
        raise AssertionError("draft proposal should require approval")

    assert (
        cli.main(
            [
                "--db",
                str(db_path),
                "review",
                "approve",
                candidate_id,
                "--reason",
                "good source-backed wiki item",
            ]
        )
        == 0
    )
    assert (
        cli.main(
            ["--db", str(db_path), "propose", candidate_id, "--output-dir", str(proposal_dir)]
        )
        == 0
    )
    decisions = list(
        conn.execute("SELECT action, next_status FROM candidate_decisions ORDER BY created_at")
    )
    assert [row["action"] for row in decisions] == ["approve", "propose"]


def test_review_reject_records_decision(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    seed_candidate = Candidate(
        target=Target.WIKI,
        title="Candidate",
        body="Draft wiki synthesis from source: Architecture uses MCP.",
        confidence=0.8,
        evidence=[Evidence(uri="/tmp/source.md", excerpt="Architecture uses MCP.")],
    )
    conn = connect(db_path)
    init_db(conn)
    candidate_id = insert_candidate(conn, seed_candidate)
    conn.commit()

    assert (
        cli.main(
            [
                "--db",
                str(db_path),
                "review",
                "reject",
                candidate_id,
                "--reason",
                "duplicate",
            ]
        )
        == 0
    )
    row = conn.execute("SELECT status FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    decision = conn.execute(
        "SELECT action, reason, next_status FROM candidate_decisions WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    assert row["status"] == "rejected"
    assert dict(decision) == {
        "action": "reject",
        "reason": "duplicate",
        "next_status": "rejected",
    }


def test_review_list_filters_low_value_groups_by_default(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    conn = connect(db_path)
    init_db(conn)
    low_value = Candidate(
        target=Target.WIKI,
        title="Wiki synthesis: STATUS ok",
        body="Draft wiki synthesis from source: STATUS ok",
        confidence=0.72,
        evidence=[Evidence(uri="/tmp/status.md", excerpt="STATUS ok")],
        claim_key="wiki status ok",
    )
    useful = Candidate(
        target=Target.WIKI,
        title="Wiki synthesis: MCP search",
        body="Draft wiki synthesis from source: MCP search supports compact retrieval.",
        confidence=0.72,
        evidence=[Evidence(uri="/tmp/mcp.md", excerpt="MCP search supports compact retrieval.")],
        claim_key="wiki mcp search supports compact retrieval",
    )
    assert insert_candidate(conn, low_value)
    assert insert_candidate(conn, useful)
    conn.commit()

    assert cli.main(["--db", str(db_path), "--pretty", "review", "list"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert [group["claim_key"] for group in payload["groups"]] == [
        "wiki mcp search supports compact retrieval"
    ]

    assert (
        cli.main(["--db", str(db_path), "--pretty", "review", "list", "--include-low-value"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert {group["claim_key"] for group in payload["groups"]} == {
        "wiki status ok",
        "wiki mcp search supports compact retrieval",
    }


def test_review_group_transition_records_each_candidate(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    conn = connect(db_path)
    init_db(conn)
    for index, uri in enumerate(("/tmp/a.md", "/tmp/b.md"), start=1):
        assert insert_candidate(
            conn,
            Candidate(
                target=Target.WIKI,
                title=f"Wiki synthesis: MCP search {index}",
                body=(
                    "Draft wiki synthesis from source: MCP search supports "
                    f"compact retrieval example {index}."
                ),
                confidence=0.72,
                evidence=[
                    Evidence(uri=uri, excerpt=f"MCP search supports compact retrieval {index}.")
                ],
                claim_key="wiki mcp search supports compact retrieval",
            ),
        )
    conn.commit()

    assert (
        cli.main(
            [
                "--db",
                str(db_path),
                "review",
                "approve-group",
                "--target",
                "wiki",
                "--claim-key",
                "wiki mcp search supports compact retrieval",
                "--reason",
                "reviewed duplicate source-backed group",
            ]
        )
        == 0
    )

    statuses = [row["status"] for row in conn.execute("SELECT status FROM candidates")]
    decisions = list(conn.execute("SELECT action, next_status FROM candidate_decisions"))
    assert statuses == ["approved", "approved"]
    assert [row["action"] for row in decisions] == ["approve_group", "approve_group"]
    assert {row["next_status"] for row in decisions} == {"approved"}


def test_excerpt_defaults_to_reviewed_candidates(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    conn = connect(db_path)
    init_db(conn)
    draft_id = insert_candidate(
        conn,
        Candidate(
            target=Target.WIKI,
            title="Draft item",
            body="Draft wiki synthesis from source: still needs review.",
            confidence=0.72,
            evidence=[Evidence(uri="/tmp/draft.md", excerpt="still needs review")],
        ),
    )
    approved_id = insert_candidate(
        conn,
        Candidate(
            target=Target.WIKI,
            title="Approved item",
            body="Draft wiki synthesis from source: reviewed context.",
            confidence=0.72,
            evidence=[Evidence(uri="/tmp/approved.md", excerpt="reviewed context")],
        ),
    )
    conn.execute("UPDATE candidates SET status = 'approved' WHERE id = ?", (approved_id,))
    conn.commit()

    excerpt = tmp_path / "AGENTS.md"
    assert (
        cli.main(["--db", str(db_path), "excerpt", "--runtime", "codex", "--output", str(excerpt)])
        == 0
    )
    text = excerpt.read_text(encoding="utf-8")
    assert approved_id in text
    assert draft_id not in text

    assert (
        cli.main(
            [
                "--db",
                str(db_path),
                "excerpt",
                "--runtime",
                "codex",
                "--output",
                str(excerpt),
                "--include-draft",
            ]
        )
        == 0
    )
    text = excerpt.read_text(encoding="utf-8")
    assert approved_id in text
    assert draft_id in text


def test_proposal_output_is_idempotent_after_approval(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    conn = connect(db_path)
    init_db(conn)
    candidate_id = insert_candidate(
        conn,
        Candidate(
            target=Target.WIKI,
            title="Approved item",
            body="Draft wiki synthesis from source: reviewed context.",
            confidence=0.72,
            evidence=[Evidence(uri="/tmp/approved.md", excerpt="reviewed context")],
        ),
    )
    conn.execute("UPDATE candidates SET status = 'approved' WHERE id = ?", (candidate_id,))
    conn.commit()
    proposal_dir = tmp_path / "proposals"

    assert (
        cli.main(
            ["--db", str(db_path), "propose", candidate_id, "--output-dir", str(proposal_dir)]
        )
        == 0
    )
    path = next(proposal_dir.glob("wiki-*.md"))
    first = path.read_text(encoding="utf-8")
    assert "proposal_hash:" in first

    assert (
        cli.main(
            ["--db", str(db_path), "propose", candidate_id, "--output-dir", str(proposal_dir)]
        )
        == 0
    )
    second = path.read_text(encoding="utf-8")
    decisions = list(conn.execute("SELECT action FROM candidate_decisions"))

    assert first == second
    assert [row["action"] for row in decisions] == ["propose"]
