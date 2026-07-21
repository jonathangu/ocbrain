from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import ocbrain.cli as cli_module
from ocbrain.cli import build_parser, main
from ocbrain.core_v1 import get_core_v1_evidence, is_core_v1
from ocbrain.db import connect


def _run(capsys, db: Path, argv: list[str], *, expected: int = 0) -> dict:
    assert main(["--db", str(db), *argv]) == expected
    output = capsys.readouterr().out
    return json.loads(output) if output else {}


def _strict_v1(db: Path) -> sqlite3.Connection:
    conn = connect(db)
    assert is_core_v1(conn)
    names = {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
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
        "automatic-activation",
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
        "export-bundle",
        "import-bundle",
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
    curated_and_vector_acceptance = {
        "curated-apply",
        "vector-build",
        "vector-status",
    }
    assert commands == (
        exercised_here | subprocess_or_migration_acceptance | curated_and_vector_acceptance
    )


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
    assert (
        _run(
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
        )["status"]
        == "verified"
    )
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

    mcp_calls: list[tuple[Path, bool, str, Path | None, str]] = []

    def fake_serve(
        path: Path,
        *,
        allow_writes: bool,
        profile: str,
        active_db_file: Path | None,
        delivery_target: str = "local_model",
    ) -> int:
        mcp_calls.append((path, allow_writes, profile, active_db_file, delivery_target))
        return 0

    monkeypatch.setattr(cli_module, "serve", fake_serve)
    assert main(["--db", str(db), "mcp", "--profile", "runtime"]) == 0
    assert mcp_calls == [(db, False, "runtime", None, "local_model")]

    mcp_calls.clear()
    assert (
        main(["--db", str(db), "mcp", "--profile", "runtime", "--delivery-target", "hosted_model"])
        == 0
    )
    assert mcp_calls == [(db, False, "runtime", None, "hosted_model")]

    assert _run(capsys, db, ["automatic-activation"])["automatic_activation"] is False
    assert _run(capsys, db, ["automatic-activation", "--enable"])["automatic_activation"] is True
    assert _run(capsys, db, ["automatic-activation", "--disable"])["automatic_activation"] is False


def test_fresh_v1_cli_read_surfaces_and_event_lifecycle(tmp_path, capsys) -> None:
    db = tmp_path / "core.sqlite"
    initialized = _run(capsys, db, ["init"])
    assert initialized["core"] == "v1"

    assert _run(capsys, db, ["status"])["database"]["healthy"] is True
    assert _run(capsys, db, ["sync"])["status"] == "ok"
    assert _run(capsys, db, ["knowledge"])["knowledge"] == []
    assert _run(capsys, db, ["search", "missing", "--project", "ocbrain"])["items"] == []
    assert _run(capsys, db, ["preview", "missing", "--project", "ocbrain"])["items"] == []
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
    assert _run(capsys, db, ["knowledge"])["knowledge"][0]["belief_id"] == ("belief:cli-lifecycle")

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
    assert (
        _run(
            capsys,
            db,
            ["search", "event authoritative", "--project", "ocbrain"],
        )["items"]
        == []
    )

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


def test_v1_private_source_imports_are_cold_evidence_and_idempotent(tmp_path, capsys) -> None:
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
    assert search["items"] == []
    preview = _run(
        capsys,
        db,
        ["preview", "durable general representations", "--project", "ocbrain"],
    )
    assert preview["items"] == []
    conn = _strict_v1(db)
    evidence = get_core_v1_evidence(conn, first["files"][0]["evidence_id"])
    assert evidence is not None
    assert "Actions and outcomes" in evidence["body"]
    belief = conn.execute(
        "SELECT egress_policy FROM current_beliefs WHERE belief_id=?",
        (first["files"][0]["belief_id"],),
    ).fetchone()
    assert belief is not None and belief["egress_policy"] == "prohibited"
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
    assert not _run(
        capsys,
        db,
        ["search", "measured outcomes future transfer", "--project", "ocbrain"],
    )["items"]

    history = tmp_path / ".codex" / "sessions" / "rollout.jsonl"
    history.parent.mkdir(parents=True)
    history.write_text(
        json.dumps({"role": "assistant", "content": "runtime bridge acceptance sentinel"}) + "\n",
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
    assert not _run(
        capsys,
        db,
        ["search", "runtime bridge acceptance sentinel", "--project", "ocbrain"],
    )["items"]
    _strict_v1(db).close()


def test_v1_history_import_persists_stat_fingerprint_gate(tmp_path, capsys) -> None:
    db = tmp_path / "core.sqlite"
    history = tmp_path / ".codex" / "sessions" / "rollout.jsonl"
    history.parent.mkdir(parents=True)
    history.write_text(
        json.dumps({"role": "user", "content": "fingerprint gate sentinel"}) + "\n",
        encoding="utf-8",
    )

    first = _run(capsys, db, ["import-history", str(history), "--project", "ocbrain"])
    assert (first["imported"], first["existing"]) == (1, 0)

    conn = _strict_v1(db)
    blob = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'history_file_fingerprints_v1'"
    ).fetchone()
    assert blob is not None
    stored = json.loads(blob[0])
    assert any("rollout.jsonl" in key for key in stored)
    conn.close()

    # Unchanged file: the fingerprint gate reports it existing without a
    # changed/unchanged recompilation.
    second = _run(capsys, db, ["import-history", str(history), "--project", "ocbrain"])
    assert (second["imported"], second["existing"]) == (0, 1)
    assert second["counts"]["brain_events"] == first["counts"]["brain_events"]

    # Appending to the file busts the fingerprint and forces a real re-import.
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"role": "assistant", "content": "gate reopened"}) + "\n")
    third = _run(capsys, db, ["import-history", str(history), "--project", "ocbrain"])
    assert (third["imported"], third["existing"]) == (1, 0)

    # And the gate settles again after the re-import.
    fourth = _run(capsys, db, ["import-history", str(history), "--project", "ocbrain"])
    assert (fourth["imported"], fourth["existing"]) == (0, 1)
    _strict_v1(db).close()


def test_v1_history_import_commits_per_file(tmp_path, capsys, monkeypatch):
    """The writer lock must be released between files: one implicit
    transaction spanning many slow redactions blocks every concurrent MCP
    writer with 'database is locked'. Count commits over a 3-file import."""
    import sqlite3 as _sqlite3

    db = tmp_path / "core.sqlite"
    history = tmp_path / ".codex" / "sessions"
    history.mkdir(parents=True)
    for i in range(3):
        (history / f"rollout-{i}.jsonl").write_text(
            json.dumps({"role": "user", "content": f"per-file commit sentinel {i}"}) + "\n",
            encoding="utf-8",
        )

    commits = 0

    class CountingConnection(_sqlite3.Connection):
        def commit(self):
            nonlocal commits
            commits += 1
            return super().commit()

    real_connect = _sqlite3.connect

    def counting_connect(*args, **kwargs):
        kwargs["factory"] = CountingConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(_sqlite3, "connect", counting_connect)
    result = _run(capsys, db, ["import-history", str(history), "--project", "ocbrain"])

    assert result["imported"] == 3
    assert commits >= 3, f"expected per-file commits, got {commits}"


def test_memory_import_dry_run_is_db_free_and_default_import_is_private(tmp_path, capsys) -> None:
    db = tmp_path / "dry-run-must-not-exist.sqlite"
    memory = tmp_path / "memory.md"
    key = "api_" + "key"
    secret = "quoted-value-0123456789"
    memory.write_text(
        "# Private source\n\n" + '{"' + key + '": "' + secret + '"}\n',
        encoding="utf-8",
    )

    preview = _run(
        capsys,
        db,
        ["import-memory", str(memory), "--project", "ocbrain", "--dry-run"],
    )
    assert preview["dry_run"] is True
    assert preview["database_touched"] is False
    assert preview["privacy_scope"] == "private"
    assert preview["would_import"] == 1
    assert preview["secret_leaks"][0]["leaks"] == ["json_quoted_secret"]
    assert preview["secret_leaks"][0]["redaction_residue"] == []
    assert not db.exists()

    imported = _run(
        capsys,
        db,
        ["import-memory", str(memory), "--project", "ocbrain"],
    )
    conn = _strict_v1(db)
    evidence = get_core_v1_evidence(conn, imported["files"][0]["evidence_id"])
    assert evidence is not None
    assert secret not in evidence["body"]
    assert "[REDACTED]" in evidence["body"]
    assert evidence["scope"]["visibility"] == "confidential"
    assert evidence["scope"]["egress_policy"] == "prohibited"
    conn.close()


def test_history_import_excludes_credential_files_from_dry_run_and_write(tmp_path, capsys) -> None:
    root = tmp_path / "history"
    root.mkdir()
    session = root / "session.jsonl"
    session.write_text(
        json.dumps({"role": "assistant", "content": "safe history sentinel"}) + "\n",
        encoding="utf-8",
    )
    credential_sentinel = "credential-file-must-never-land"
    (root / "auth.json").write_text(
        json.dumps({"credential": credential_sentinel}),
        encoding="utf-8",
    )
    (root / "credentials.json").write_text(
        json.dumps({"credential": credential_sentinel + "-second"}),
        encoding="utf-8",
    )

    dry_db = tmp_path / "history-dry-run.sqlite"
    preview = _run(
        capsys,
        dry_db,
        ["import-history", str(root), "--project", "ocbrain", "--dry-run"],
    )
    assert preview["would_import"] == 1
    assert preview["skipped_count"] == 2
    assert {item["reason"] for item in preview["skipped"]} == {"sensitive_filename"}
    assert not dry_db.exists()

    db = tmp_path / "history.sqlite"
    imported = _run(
        capsys,
        db,
        ["import-history", str(root), "--project", "ocbrain"],
    )
    assert imported["imported"] == 1
    assert imported["skipped_count"] == 2
    conn = _strict_v1(db)
    bodies = "\n".join(str(row[0]) for row in conn.execute("SELECT body FROM evidence_objects"))
    assert "safe history sentinel" in bodies
    assert credential_sentinel not in bodies
    conn.close()


def test_actual_imports_redact_before_byte_and_window_truncation(tmp_path, capsys) -> None:
    key = "api_" + "key"
    secret = "boundary-secret-value-abcdefghijklmnopqrstuvwxyz"

    memory = tmp_path / "boundary.md"
    memory.write_text("prefix-0123456789 " + '{"' + key + '": "' + secret + '"}', encoding="utf-8")
    memory_db = tmp_path / "memory.sqlite"
    memory_result = _run(
        capsys,
        memory_db,
        ["import-memory", str(memory), "--project", "ocbrain", "--max-bytes", "48"],
    )
    memory_conn = _strict_v1(memory_db)
    memory_evidence = get_core_v1_evidence(memory_conn, memory_result["files"][0]["evidence_id"])
    assert memory_evidence is not None
    assert "boundary-secret" not in memory_evidence["body"]
    memory_conn.close()

    history = tmp_path / "history.jsonl"
    history.write_text(
        "head-0123456789 " + '{"' + key + '": "' + secret + '"} ' + ("tail-padding " * 20),
        encoding="utf-8",
    )
    history_db = tmp_path / "history-boundary.sqlite"
    history_result = _run(
        capsys,
        history_db,
        ["import-history", str(history), "--project", "ocbrain", "--max-bytes", "80"],
    )
    history_conn = _strict_v1(history_db)
    history_evidence = get_core_v1_evidence(
        history_conn, history_result["sample_files"][0]["evidence_id"]
    )
    assert history_evidence is not None
    assert "boundary-secret" not in history_evidence["body"]
    history_conn.close()


def test_history_window_streams_and_redacts_multiline_private_keys(tmp_path, monkeypatch) -> None:
    history = tmp_path / "large-history.jsonl"
    key = "api_" + "key"
    secret = "stream-boundary-secret-abcdefghijklmnopqrstuvwxyz"
    history.write_text(
        '{"'
        + key
        + '": "'
        + secret
        + '"}\n'
        + ("middle-padding\n" * 200)
        + "-----BEGIN PRIVATE KEY-----\n"
        + "private-material-that-must-not-survive\n"
        + "-----END PRIVATE KEY-----\n"
        + "tail-visible\n",
        encoding="utf-8",
    )

    def refuse_whole_file_read(self, *args, **kwargs):
        if self == history:
            raise AssertionError("history window must stream instead of using Path.read_text")
        return original_read_text(self, *args, **kwargs)

    original_read_text = Path.read_text
    monkeypatch.setattr(Path, "read_text", refuse_whole_file_read)

    window = cli_module.history_text_window(history, max_bytes=512)

    assert secret not in window
    assert "private-material" not in window
    assert "[REDACTED]" in window
    assert "[REDACTED_PRIVATE_KEY]" in window
    assert "tail-visible" in window
    assert "bytes omitted from middle" in window


def test_history_dry_run_streams_without_opening_the_database(
    tmp_path, capsys, monkeypatch
) -> None:
    history = tmp_path / "dry-run-history.jsonl"
    history.write_text(
        '{"api_key": "dry-run-secret-value-abcdefghijklmnopqrstuvwxyz"}\n'
        + ("safe history row\n" * 200),
        encoding="utf-8",
    )
    db = tmp_path / "must-not-exist.sqlite"
    original_read_text = Path.read_text

    def refuse_whole_file_read(self, *args, **kwargs):
        if self == history:
            raise AssertionError("dry-run inspection must stream source files")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", refuse_whole_file_read)
    result = _run(
        capsys,
        db,
        ["import-history", str(history), "--project", "ocbrain", "--dry-run"],
    )

    assert result["database_touched"] is False
    assert result["would_import"] == 1
    assert result["secret_leak_count"] == 1
    assert "json_quoted_secret" in result["secret_leaks"][0]["leaks"]
    assert result["secret_leaks"][0]["redaction_residue"] == []
    assert not db.exists()


def test_history_import_uses_normal_synchronous_mode(tmp_path, capsys, monkeypatch) -> None:
    db = tmp_path / "core.sqlite"
    _run(capsys, db, ["init"])
    conn = _strict_v1(db)
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    monkeypatch.setattr(cli_module, "open_db", lambda _args: conn)
    history = tmp_path / "session.jsonl"
    history.write_text('{"message": "safe"}\n', encoding="utf-8")

    _run(capsys, db, ["import-history", str(history), "--project", "ocbrain"])

    assert any(statement.upper() == "PRAGMA SYNCHRONOUS=NORMAL" for statement in statements)
    conn.close()


def test_directory_sweeps_skip_hidden_descendants_but_explicit_hidden_root_is_allowed(
    tmp_path, capsys
) -> None:
    root = tmp_path / "history-root"
    hidden_root = root / ".codex"
    hidden_root.mkdir(parents=True)
    (hidden_root / "hidden-session.jsonl").write_text(
        '{"message": "hidden but explicitly selectable"}\n', encoding="utf-8"
    )
    (root / "visible-session.jsonl").write_text('{"message": "visible"}\n', encoding="utf-8")

    broad = _run(
        capsys,
        tmp_path / "broad.sqlite",
        ["import-history", str(root), "--project", "ocbrain", "--dry-run"],
    )
    assert broad["would_import"] == 1
    assert broad["skipped_count"] == 1
    assert broad["skipped"][0]["reason"] == "hidden_path"

    explicit = _run(
        capsys,
        tmp_path / "explicit.sqlite",
        ["import-history", str(hidden_root), "--project", "ocbrain", "--dry-run"],
    )
    assert explicit["would_import"] == 1
    assert explicit["skipped_count"] == 0

    memory_root = tmp_path / "memory-root"
    hidden_memory_root = memory_root / ".private"
    hidden_memory_root.mkdir(parents=True)
    (hidden_memory_root / "hidden.md").write_text("hidden memory\n", encoding="utf-8")
    (memory_root / "visible.md").write_text("visible memory\n", encoding="utf-8")
    broad_memory = _run(
        capsys,
        tmp_path / "broad-memory.sqlite",
        ["import-memory", str(memory_root), "--project", "ocbrain", "--dry-run"],
    )
    assert broad_memory["would_import"] == 1
    assert broad_memory["skipped"][0]["reason"] == "hidden_path"
    explicit_memory = _run(
        capsys,
        tmp_path / "explicit-memory.sqlite",
        ["import-memory", str(hidden_memory_root), "--project", "ocbrain", "--dry-run"],
    )
    assert explicit_memory["would_import"] == 1
    assert explicit_memory["skipped_count"] == 0


def test_directory_sweeps_reject_symlinks_outside_root_and_into_hidden_targets(
    tmp_path,
) -> None:
    history_root = tmp_path / "history-root"
    history_root.mkdir()
    outside_history = tmp_path / "outside.jsonl"
    outside_history.write_text('{"message": "outside"}\n', encoding="utf-8")
    (history_root / "outside-link.jsonl").symlink_to(outside_history)
    hidden_history = history_root / ".hidden" / "session.jsonl"
    hidden_history.parent.mkdir()
    hidden_history.write_text('{"message": "hidden"}\n', encoding="utf-8")
    (history_root / "hidden-link.jsonl").symlink_to(hidden_history)

    history_skipped: list[dict[str, str]] = []
    selected_history = cli_module.history_files([history_root], skipped=history_skipped)

    assert selected_history == []
    assert "outside_sweep_root" in {item["reason"] for item in history_skipped}
    assert "hidden_path" in {item["reason"] for item in history_skipped}

    memory_root = tmp_path / "memory-root"
    memory_root.mkdir()
    outside_memory = tmp_path / "outside.md"
    outside_memory.write_text("outside memory\n", encoding="utf-8")
    (memory_root / "outside-link.md").symlink_to(outside_memory)
    hidden_memory = memory_root / ".hidden" / "memory.md"
    hidden_memory.parent.mkdir()
    hidden_memory.write_text("hidden memory\n", encoding="utf-8")
    (memory_root / "hidden-link.md").symlink_to(hidden_memory)

    memory_skipped: list[dict[str, str]] = []
    selected_memory = cli_module.memory_files([memory_root], skipped=memory_skipped)

    assert selected_memory == []
    assert "outside_sweep_root" in {item["reason"] for item in memory_skipped}
    assert "hidden_path" in {item["reason"] for item in memory_skipped}
