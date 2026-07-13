from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

import ocbrain.bundle as bundle_module
from ocbrain.bundle import (
    MAX_ITEM_BODY_CHARS,
    BundleExportError,
    BundleImportError,
    export_bundle,
    import_bundle,
    load_bundle,
)
from ocbrain.cli import main
from ocbrain.core_v1 import (
    canonical_json,
    get_core_v1_evidence,
    init_core_v1,
    record_core_v1_evidence,
    sha256_text,
)
from ocbrain.db import connect
from ocbrain.scope import ScopeContext, ScopeTag


def _core(path: Path):
    conn = connect(path)
    init_core_v1(conn)
    conn.commit()
    return conn


def _record(conn, body: str, *, policy: str = "hosted_ok") -> str:
    evidence_id, _ = record_core_v1_evidence(
        conn,
        body=body,
        kind="observation",
        scope=ScopeTag(
            "project",
            "project:sender",
            visibility="internal",
            egress_policy=policy,
        ),
        writer="test",
    )
    conn.commit()
    return evidence_id


def _export_one(tmp_path: Path, body: str = "portable evidence") -> tuple[Path, str]:
    source = _core(tmp_path / "source.sqlite")
    evidence_id = _record(source, body)
    path = tmp_path / "evidence.bundle.json"
    export_bundle(
        source,
        path,
        evidence_ids=[evidence_id],
        context=ScopeContext(project="sender"),
    )
    source.close()
    return path, evidence_id


def _rewrite_bundle(path: Path, transform) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    transform(data)
    path.write_text(canonical_json(data) + "\n", encoding="utf-8")
    return data


def test_export_redacts_audits_and_publishes_fresh_owner_only_file(tmp_path: Path) -> None:
    source = _core(tmp_path / "source.sqlite")
    key = "api_" + "key"
    secret = "quoted-export-value-0123456789"
    evidence_id = _record(source, '{"' + key + '": "' + secret + '"}')
    path = tmp_path / "bundle.json"

    result = export_bundle(
        source,
        path,
        evidence_ids=[evidence_id],
        context=ScopeContext(project="sender"),
    )

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    payload = load_bundle(path)
    assert payload["item_count"] == 1
    assert secret not in payload["items"][0]["body"]
    assert "[REDACTED]" in payload["items"][0]["body"]
    audit = source.execute(
        "SELECT target, payload_hash FROM egress_audits WHERE id=?",
        (result["egress_audit_id"],),
    ).fetchone()
    assert tuple(audit) == ("human_export", payload["payload_hash"])

    with pytest.raises(BundleExportError, match="overwrite"):
        export_bundle(
            source,
            path,
            evidence_ids=[evidence_id],
            context=ScopeContext(project="sender"),
        )
    assert source.execute("SELECT COUNT(*) FROM egress_audits").fetchone()[0] == 1
    source.close()


@pytest.mark.parametrize("policy", ["local_only", "prohibited"])
def test_export_refuses_non_exportable_selected_evidence(tmp_path: Path, policy: str) -> None:
    source = _core(tmp_path / f"{policy}.sqlite")
    evidence_id = _record(source, f"{policy} evidence", policy=policy)
    path = tmp_path / f"{policy}.json"

    with pytest.raises(BundleExportError, match=policy):
        export_bundle(
            source,
            path,
            evidence_ids=[evidence_id],
            context=ScopeContext(project="sender"),
        )

    assert not path.exists()
    assert source.execute("SELECT COUNT(*) FROM egress_audits").fetchone()[0] == 0
    source.close()


def test_export_requires_explicit_approval_and_refuses_oversized_body(tmp_path: Path) -> None:
    source = _core(tmp_path / "approval.sqlite")
    approval_id = _record(source, "approved portable evidence", policy="approval_required")
    path = tmp_path / "approval.json"
    with pytest.raises(BundleExportError, match="approval_required"):
        export_bundle(
            source,
            path,
            evidence_ids=[approval_id],
            context=ScopeContext(project="sender"),
        )
    export_bundle(
        source,
        path,
        evidence_ids=[approval_id],
        context=ScopeContext(project="sender"),
        approve_egress=True,
    )

    oversized_id = _record(source, "x" * (MAX_ITEM_BODY_CHARS + 1))
    with pytest.raises(BundleExportError, match="invalid_body_size"):
        export_bundle(
            source,
            tmp_path / "oversized.json",
            evidence_ids=[oversized_id],
            context=ScopeContext(project="sender"),
        )
    source.close()


def test_bundle_validation_rejects_tamper_and_naive_timestamp(tmp_path: Path) -> None:
    path, _ = _export_one(tmp_path)
    original = path.read_text(encoding="utf-8")
    data = json.loads(original)
    data["items"][0]["body"] = "tampered"
    path.write_text(canonical_json(data), encoding="utf-8")
    with pytest.raises(BundleImportError, match="payload hash mismatch"):
        load_bundle(path)

    path.write_text(original, encoding="utf-8")
    data = json.loads(original)
    data["created_at"] = "2026-07-13T12:00:00"
    path.write_text(canonical_json(data), encoding="utf-8")
    with pytest.raises(BundleImportError, match="timezone"):
        load_bundle(path)


def test_import_defaults_to_db_free_dry_run_then_applies_and_dedups(tmp_path: Path) -> None:
    path, _ = _export_one(tmp_path)
    missing_db = tmp_path / "must-not-exist.sqlite"

    preview = import_bundle(None, path, project="receiver")
    assert preview["dry_run"] is True
    assert preview["database_touched"] is False
    assert preview["would_append"] == 1
    assert not missing_db.exists()

    destination = _core(tmp_path / "destination.sqlite")
    before_tables = {
        str(row[0])
        for row in destination.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        )
    }
    first = import_bundle(destination, path, project="receiver", apply=True)
    assert first["appended"] == 1
    assert first["projection"]["applied_events"] == 1
    evidence = get_core_v1_evidence(destination, first["local_evidence_ids"][0])
    assert evidence is not None
    assert evidence["scope"] == {
        "scope_type": "project",
        "scope_id": "project:receiver",
        "visibility": "confidential",
        "egress_policy": "local_only",
        "provenance": "bundle_import",
    }
    assert evidence["kind"] == "bundle_import"
    assert evidence["metadata"]["event_body"]["bundle_provenance"]["payload_hash"]
    assert destination.execute("SELECT COUNT(*) FROM current_beliefs").fetchone()[0] == 0
    assert destination.execute("SELECT COUNT(*) FROM task_closeouts").fetchone()[0] == 0

    second = import_bundle(destination, path, project="receiver", apply=True)
    assert second["appended"] == 0
    assert second["deduped"] == 1
    after_tables = {
        str(row[0])
        for row in destination.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        )
    }
    assert after_tables == before_tables
    destination.close()


def test_import_ignores_malicious_remote_id_collision(tmp_path: Path) -> None:
    path, _ = _export_one(tmp_path, "new remote evidence")
    destination = _core(tmp_path / "destination.sqlite")
    existing_id = _record(destination, "existing local evidence")

    def collide(data: dict) -> None:
        data["items"][0]["source_evidence_id"] = existing_id
        data["payload_hash"] = sha256_text(canonical_json(data["items"]))

    _rewrite_bundle(path, collide)
    result = import_bundle(destination, path, project="receiver", apply=True)

    assert result["local_evidence_ids"] != [existing_id]
    assert get_core_v1_evidence(destination, existing_id)["body"] == "existing local evidence"
    assert get_core_v1_evidence(destination, result["local_evidence_ids"][0])["body"] == (
        "new remote evidence"
    )
    destination.close()


def test_import_rolls_back_all_events_when_projection_fails(tmp_path: Path, monkeypatch) -> None:
    path, _ = _export_one(tmp_path)
    destination = _core(tmp_path / "destination.sqlite")
    before_events = destination.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0]

    def fail_projection(_conn):
        raise RuntimeError("projection failed")

    monkeypatch.setattr(bundle_module, "project_core_v1", fail_projection)
    with pytest.raises(RuntimeError, match="projection failed"):
        import_bundle(destination, path, project="receiver", apply=True)

    assert destination.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0] == before_events
    assert destination.execute("SELECT COUNT(*) FROM evidence_objects").fetchone()[0] == 0
    destination.close()


def test_bundle_cli_routes_export_dry_run_and_apply(tmp_path: Path, capsys) -> None:
    source_path = tmp_path / "source.sqlite"
    source = _core(source_path)
    evidence_id = _record(source, "CLI portable evidence")
    source.close()
    bundle = tmp_path / "cli-bundle.json"

    assert (
        main(
            [
                "--db",
                str(source_path),
                "export-bundle",
                "--output",
                str(bundle),
                "--evidence-id",
                evidence_id,
                "--project",
                "sender",
            ]
        )
        == 0
    )
    capsys.readouterr()

    missing = tmp_path / "dry-run-does-not-create.sqlite"
    assert (
        main(
            [
                "--db",
                str(missing),
                "import-bundle",
                str(bundle),
                "--project",
                "receiver",
            ]
        )
        == 0
    )
    preview = json.loads(capsys.readouterr().out)
    assert preview["dry_run"] is True
    assert not missing.exists()

    destination_path = tmp_path / "destination.sqlite"
    _core(destination_path).close()
    assert (
        main(
            [
                "--db",
                str(destination_path),
                "import-bundle",
                str(bundle),
                "--project",
                "receiver",
                "--apply",
            ]
        )
        == 0
    )
    applied = json.loads(capsys.readouterr().out)
    assert applied["appended"] == 1
