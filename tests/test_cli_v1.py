from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import ocbrain.cli as cli_module
from ocbrain.cli import build_parser, main
from ocbrain.core_v1 import get_core_v1_evidence, is_core_v1
from ocbrain.db import connect
from ocbrain.mcp_v1 import expand_source_v1
from ocbrain.scope import ScopeContext


def _run(capsys, db: Path, argv: list[str], *, expected: int = 0) -> dict:
    assert main(["--db", str(db), *argv]) == expected
    output = capsys.readouterr().out
    return json.loads(output) if output else {}


def _strict_v1(db: Path) -> sqlite3.Connection:
    conn = connect(db)
    assert is_core_v1(conn)
    names = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        )
    }
    assert not {"evidence", "knowledge", "knowledge_evidence", "memory"} & names
    return conn


def test_every_advertised_core_command_has_a_v1_acceptance_route() -> None:
    commands = set(
        next(action for action in build_parser()._actions if action.dest == "command").choices
    )
    exercised_here = {
        "init",
        "status",
        "sync",
        "evidence",
        "knowledge",
        "value",
        "search",
        "preview",
        "event-ingest",
        "event-compile",
        "event-correct",
        "event-forget",
        "event-proposals",
        "event-decide",
        "event-digest",
        "egress-preview",
        "event-backfill",
        "import-memory",
        "import-history",
        "digest",
    }
    subprocess_or_migration_acceptance = {
        "doctor",
        "runtime-check",
        "backup",
        "restore",
        "core-migrate-v1",
        "mcp",
    }
    assert commands == exercised_here | subprocess_or_migration_acceptance


def test_fresh_v1_operational_cli_routes(tmp_path, capsys, monkeypatch) -> None:
    """Exercise the non-interactive seam of every operational core command."""
    db = tmp_path / "core.sqlite"
    _run(capsys, db, ["init"])

    backup = tmp_path / "backup.sqlite"
    manifest = tmp_path / "backup.json"
    backed_up = _run(
        capsys,
        db,
        ["backup", "--output", str(backup), "--manifest", str(manifest)],
    )
    assert backed_up["status"] == "verified"
    restored = tmp_path / "restored.sqlite"
    assert _run(
        capsys,
        db,
        [
            "restore",
            "--backup",
            str(backup),
            "--output-db",
            str(restored),
            "--manifest",
            str(manifest),
        ],
    )["status"] == "verified"
    _strict_v1(restored).close()

    core = tmp_path / "migrated.sqlite"
    archive = tmp_path / "archive.sqlite"
    migration_manifest = tmp_path / "migration.json"
    plan = _run(
        capsys,
        db,
        [
            "core-migrate-v1",
            "--core-db",
            str(core),
            "--archive-db",
            str(archive),
            "--manifest",
            str(migration_manifest),
            "--plan",
        ],
    )
    assert plan["ready"] is True
    assert not core.exists() and not archive.exists() and not migration_manifest.exists()

    doctor_calls: list[bool] = []

    def healthy_doctor(*_args, check_clients: bool, **_kwargs):
        doctor_calls.append(check_clients)
        return {"healthy": True, "status": "ok"}

    monkeypatch.setattr(cli_module, "doctor", healthy_doctor)
    assert _run(capsys, db, ["doctor"])["healthy"] is True
    assert _run(capsys, db, ["runtime-check"])["healthy"] is True
    assert doctor_calls == [False, True]

    mcp_calls: list[tuple[Path, bool, str]] = []

    def fake_serve(path: Path, *, allow_writes: bool, profile: str) -> int:
        mcp_calls.append((path, allow_writes, profile))
        return 0

    monkeypatch.setattr(cli_module, "serve", fake_serve)
    assert main(["--db", str(db), "mcp", "--profile", "runtime"]) == 0
    assert mcp_calls == [(db, False, "runtime")]


def test_fresh_v1_cli_read_surfaces_and_event_lifecycle(tmp_path, capsys) -> None:
    db = tmp_path / "core.sqlite"
    initialized = _run(capsys, db, ["init"])
    assert initialized["core"] == "v1"

    assert _run(capsys, db, ["status"])["database"]["healthy"] is True
    assert _run(capsys, db, ["sync"])["status"] == "ok"
    assert _run(capsys, db, ["knowledge"])["knowledge"] == []
    assert _run(capsys, db, ["search", "missing", "--project", "ocbrain"])["items"] == []
    assert (
        _run(capsys, db, ["preview", "missing", "--project", "ocbrain"])["items"]
        == []
    )
    assert _run(capsys, db, ["event-proposals"])["proposals"] == []
    assert _run(capsys, db, ["event-digest", "--project", "ocbrain"])["current"] == []
    assert _run(capsys, db, ["digest", "--project", "ocbrain"])["current"] == []
    assert (
        _run(
            capsys,
            db,
            ["egress-preview", "--target", "local_model", "--project", "ocbrain"],
        )["included_count"]
        == 0
    )

    explicit = _run(
        capsys,
        db,
        ["evidence", "--claim", "A source-backed v1 CLI fact", "--project", "ocbrain"],
    )
    assert explicit["evidence_id"].startswith("evd_")

    ingested = _run(
        capsys,
        db,
        [
            "event-ingest",
            "--body",
            "The v1 CLI lifecycle preserves event authority.",
            "--project",
            "ocbrain",
        ],
    )
    evidence_id = ingested["evidence_id"]
    proposed = _run(
        capsys,
        db,
        [
            "event-compile",
            "--belief-id",
            "belief:cli-lifecycle",
            "--body",
            "The v1 CLI lifecycle is event authoritative.",
            "--evidence-id",
            evidence_id,
            "--project",
            "ocbrain",
        ],
    )
    proposal_id = proposed["proposal_event_id"]
    pending = _run(capsys, db, ["event-proposals"])["proposals"]
    assert [item["proposal_event_id"] for item in pending] == [proposal_id]

    decided = _run(
        capsys,
        db,
        ["event-decide", "--proposal-event-id", proposal_id, "--decision", "approve"],
    )
    assert decided["decision"] == "approve"
    searched = _run(
        capsys,
        db,
        ["search", "event authoritative", "--project", "ocbrain"],
    )
    assert searched["items"][0]["id"] == "belief:cli-lifecycle"
    preview = _run(
        capsys,
        db,
        ["preview", "event authoritative", "--project", "ocbrain"],
    )
    assert preview["coverage"]["source_handle_count"] == 1
    assert preview["retrieval_use_status"] == "recorded"
    assert _run(capsys, db, ["knowledge"])["knowledge"][0]["belief_id"] == (
        "belief:cli-lifecycle"
    )

    corrected = _run(
        capsys,
        db,
        [
            "event-correct",
            "--target-layer",
            "belief",
            "--target-id",
            "belief:cli-lifecycle",
            "--op",
            "edit",
            "--body",
            "The corrected v1 CLI lifecycle remains event authoritative.",
        ],
    )
    assert corrected["kind"] == "correction_recorded"
    digest = _run(capsys, db, ["digest", "--project", "ocbrain"])
    assert digest["current"][0]["body"].startswith("The corrected")

    forgotten = _run(
        capsys,
        db,
        ["event-forget", "--target", "belief:cli-lifecycle", "--reason", "test"],
    )
    assert forgotten["kind"] == "tombstone_recorded"
    assert _run(
        capsys,
        db,
        ["search", "event authoritative", "--project", "ocbrain"],
    )["items"] == []

    value = _run(
        capsys,
        db,
        [
            "value",
            "--subject",
            "ocbrain",
            "--predicate",
            "core_version",
            "--text",
            "v1",
            "--status",
            "current",
            "--project",
            "ocbrain",
        ],
    )
    assert value["status"] == "current"

    refused = _run(capsys, db, ["event-backfill", "--dry-run"], expected=2)
    assert refused["reason"] == "legacy_compatibility_command_on_v1_core"
    _strict_v1(db).close()


def test_v1_source_imports_are_searchable_expandable_and_idempotent(tmp_path, capsys) -> None:
    db = tmp_path / "core.sqlite"
    _run(capsys, db, ["init"])
    memory = tmp_path / "memory.md"
    memory.write_text(
        "# Transfer learning\n\nActions and outcomes need durable, general representations.\n",
        encoding="utf-8",
    )

    first = _run(
        capsys,
        db,
        ["import-memory", str(memory), "--project", "ocbrain"],
    )
    assert (first["imported"], first["existing"]) == (1, 0)
    event_count = first["counts"]["brain_events"]
    second = _run(
        capsys,
        db,
        ["import-memory", str(memory), "--project", "ocbrain"],
    )
    assert (second["imported"], second["existing"]) == (0, 1)
    assert second["counts"]["brain_events"] == event_count

    search = _run(
        capsys,
        db,
        ["search", "durable general representations", "--project", "ocbrain"],
    )
    assert len(search["items"]) == 1
    preview = _run(
        capsys,
        db,
        ["preview", "durable general representations", "--project", "ocbrain"],
    )
    source = preview["items"][0]["sources"][0]
    assert source["uri"] == str(memory.resolve())
    conn = _strict_v1(db)
    evidence = get_core_v1_evidence(conn, preview["items"][0]["evidence_ids"][0])
    assert evidence is not None
    assert "Actions and outcomes" in evidence["body"]
    expanded = expand_source_v1(
        conn,
        source["id"],
        context=ScopeContext(project="ocbrain"),
        max_chars=4_000,
    )
    assert expanded["hash_verified"] is True
    assert "Actions and outcomes" in expanded["content"]
    conn.close()

    memory.write_text(
        "# Transfer learning\n\nActions, context, and measured outcomes enable future transfer.\n",
        encoding="utf-8",
    )
    changed = _run(
        capsys,
        db,
        ["import-memory", str(memory), "--project", "ocbrain"],
    )
    assert changed["imported"] == 1
    assert changed["counts"]["brain_events"] == event_count + 3
    assert _run(
        capsys,
        db,
        ["search", "measured outcomes future transfer", "--project", "ocbrain"],
    )["items"]

    history = tmp_path / ".codex" / "sessions" / "rollout.jsonl"
    history.parent.mkdir(parents=True)
    history.write_text(
        json.dumps({"role": "assistant", "content": "runtime bridge acceptance sentinel"})
        + "\n",
        encoding="utf-8",
    )
    imported = _run(
        capsys,
        db,
        ["import-history", str(history), "--project", "ocbrain"],
    )
    assert (imported["imported"], imported["existing"]) == (1, 0)
    repeated = _run(
        capsys,
        db,
        ["import-history", str(history), "--project", "ocbrain"],
    )
    assert (repeated["imported"], repeated["existing"]) == (0, 1)
    assert _run(
        capsys,
        db,
        ["search", "runtime bridge acceptance sentinel", "--project", "ocbrain"],
    )["items"]
    _strict_v1(db).close()
