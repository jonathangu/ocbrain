import json
from pathlib import Path

import pytest

from ocbrain import cli
from ocbrain.bundle import (
    BundleExportError,
    BundleImportError,
    bundle_payload_hash,
    cap_scope_for_import,
    export_bundle,
    import_bundle,
)
from ocbrain.db import connect, init_db
from ocbrain.dream import dream
from ocbrain.events import decide_compilation, evidence_id_for, record_evidence
from ocbrain.retrieve import retrieve
from ocbrain.scope import ScopeContext, ScopeTag

SECRET = "sk-123456789012345678901234"

HOSTED_OK_BODY = f"Bountiful garden sensors use LoRa uplinks; api_key={SECRET}."
APPROVAL_BODY = "Bountiful watering schedule shifts to dawn cycles during heat waves."
LOCAL_ONLY_BODY = "Cormorant Bihua lane registry notes stay on this machine."
PROHIBITED_BODY = "Never-egress operational notes for the blacksite scope."

HOSTED_OK_SCOPE = ScopeTag(
    "project", "project:bountiful", visibility="internal", egress_policy="hosted_ok"
)
APPROVAL_SCOPE = ScopeTag(
    "project", "project:bountiful", visibility="internal", egress_policy="approval_required"
)
LOCAL_ONLY_SCOPE = ScopeTag(
    "client", "client:bihua", visibility="confidential", egress_policy="local_only"
)
PROHIBITED_SCOPE = ScopeTag(
    "project", "project:blacksite", visibility="secret", egress_policy="prohibited"
)

BOUNTIFUL_FILTER = [
    "--scope-type",
    "project",
    "--scope-id",
    "project:bountiful",
    "--scope-type",
    "client",
    "--scope-id",
    "client:bihua",
]


def seeded_brain_a(tmp_path: Path) -> Path:
    db_path = tmp_path / "brain_a.sqlite"
    conn = connect(db_path)
    init_db(conn)
    record_evidence(conn, body=HOSTED_OK_BODY, scope=HOSTED_OK_SCOPE, writer="codex")
    record_evidence(conn, body=APPROVAL_BODY, scope=APPROVAL_SCOPE, writer="claude")
    record_evidence(conn, body=LOCAL_ONLY_BODY, scope=LOCAL_ONLY_SCOPE, writer="claude")
    record_evidence(conn, body=PROHIBITED_BODY, scope=PROHIBITED_SCOPE, writer="ocbrain")
    conn.commit()
    conn.close()
    return db_path


def exported_bundle_path(tmp_path: Path, capsys) -> tuple[Path, Path, dict]:
    db_a = seeded_brain_a(tmp_path)
    bundle_path = tmp_path / "brain_a.bundle.json"
    rc = cli.main(
        [
            "--db",
            str(db_a),
            "export-bundle",
            "--output",
            str(bundle_path),
            "--label",
            "mini-a",
            *BOUNTIFUL_FILTER,
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    return db_a, bundle_path, payload


def event_rows(db_path: Path) -> list[dict]:
    conn = connect(db_path)
    init_db(conn)
    rows = [
        {"writer": row["writer"], "session_id": row["session_id"]}
        | json.loads(row["body_json"])
        for row in conn.execute(
            "SELECT writer, session_id, body_json FROM brain_events "
            "WHERE kind = 'evidence_recorded' ORDER BY rowid"
        )
    ]
    conn.close()
    return rows


def test_export_refuses_while_prohibited_evidence_is_selected(tmp_path: Path, capsys) -> None:
    db_a = seeded_brain_a(tmp_path)
    bundle_path = tmp_path / "refused.bundle.json"
    rc = cli.main(["--db", str(db_a), "export-bundle", "--output", str(bundle_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "refused" in err and "prohibited" in err
    assert not bundle_path.exists()

    # No audit row is recorded for a refused export.
    conn = connect(db_a)
    init_db(conn)
    assert conn.execute("SELECT COUNT(*) AS n FROM egress_audits").fetchone()["n"] == 0
    conn.close()


def test_export_refusal_ignores_limit(tmp_path: Path) -> None:
    db_a = seeded_brain_a(tmp_path)
    conn = connect(db_a)
    init_db(conn)
    with pytest.raises(BundleExportError):
        export_bundle(conn, db_path=db_a, limit=1)
    conn.close()


def test_export_gates_redacts_and_audits(tmp_path: Path, capsys) -> None:
    db_a, bundle_path, payload = exported_bundle_path(tmp_path, capsys)

    assert payload["count"] == 2
    assert payload["skipped_count"] == 1
    (skipped,) = payload["skipped"]
    assert skipped["reason"] == "egress_policy:local_only"
    assert skipped["scope"]["scope_id"] == "client:bihua"

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["schema_version"] == "ocbrain.bundle.v1"
    assert bundle["origin"]["hostname_free_label"] == "mini-a"
    assert bundle["origin"]["writer"] == "ocbrain-export"
    assert bundle["origin"]["db_path_hash"]
    assert bundle["created_at"]
    assert bundle["count"] == 2
    assert bundle["payload_hash"] == bundle_payload_hash(bundle["evidence"])

    bodies = {item["evidence_id"]: item for item in bundle["evidence"]}
    hosted_id = evidence_id_for(
        body=HOSTED_OK_BODY, kind="observation", artifact_ref=None, scope=HOSTED_OK_SCOPE
    )
    assert hosted_id in bodies
    assert SECRET not in json.dumps(bundle)
    assert "[REDACTED]" in bodies[hosted_id]["body"]
    assert bodies[hosted_id]["scope"] == {
        "scope_type": "project",
        "scope_id": "project:bountiful",
        "visibility": "internal",
        "egress_policy": "hosted_ok",
    }
    assert bodies[hosted_id]["writer"] == "codex"
    assert bodies[hosted_id]["ts"]

    # The export left one human_export egress audit row behind.
    conn = connect(db_a)
    init_db(conn)
    audit = conn.execute("SELECT * FROM egress_audits").fetchall()
    conn.close()
    assert len(audit) == 1
    assert audit[0]["target"] == "human_export"
    assert audit[0]["payload_hash"] == bundle["payload_hash"]
    assert audit[0]["id"] == payload["audit_id"]


def test_import_dry_run_plans_without_writing(tmp_path: Path, capsys) -> None:
    _, bundle_path, _ = exported_bundle_path(tmp_path, capsys)
    db_b = tmp_path / "brain_b.sqlite"
    rc = cli.main(["--db", str(db_b), "import-bundle", str(bundle_path), "--dry-run"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["counts"] == {"new": 2, "duplicates": 0, "skipped": 0}
    assert payload["writer"] == "import:mini-a"
    assert len(payload["sample"]) == 2
    assert payload["imported_event_ids"] == []
    assert event_rows(db_b) == []


def test_round_trip_import_caps_egress_and_dedups(tmp_path: Path, capsys) -> None:
    _, bundle_path, _ = exported_bundle_path(tmp_path, capsys)
    db_b = tmp_path / "brain_b.sqlite"

    rc = cli.main(
        ["--db", str(db_b), "import-bundle", str(bundle_path), "--actor", "human:friend"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"] == {"new": 2, "duplicates": 0, "skipped": 0}
    assert len(payload["imported_event_ids"]) == 2

    rows = event_rows(db_b)
    assert len(rows) == 2
    hosted_id = evidence_id_for(
        body=HOSTED_OK_BODY, kind="observation", artifact_ref=None, scope=HOSTED_OK_SCOPE
    )
    by_id = {row["evidence_id"]: row for row in rows}
    assert set(by_id) == {
        hosted_id,
        evidence_id_for(
            body=APPROVAL_BODY, kind="observation", artifact_ref=None, scope=APPROVAL_SCOPE
        ),
    }
    for row in rows:
        assert row["writer"] == "import:mini-a"
        assert row["session_id"] == "human:friend"
        assert row["scope"]["scope_id"] == "project:bountiful"
        # hosted_ok is capped so a friend's evidence cannot silently re-egress.
        assert row["scope"]["egress_policy"] == "approval_required"
    assert SECRET not in json.dumps(rows)

    # Re-import: everything dedups on the content-derived evidence id.
    rc = cli.main(["--db", str(db_b), "import-bundle", str(bundle_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"] == {"new": 0, "duplicates": 2, "skipped": 0}
    assert len(event_rows(db_b)) == 2


def test_import_rejects_tampered_payload_and_bad_schema(tmp_path: Path, capsys) -> None:
    _, bundle_path, _ = exported_bundle_path(tmp_path, capsys)
    db_b = tmp_path / "brain_b.sqlite"

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["evidence"][0]["body"] = "tampered claim injected in transit"
    tampered = tmp_path / "tampered.bundle.json"
    tampered.write_text(json.dumps(bundle), encoding="utf-8")
    rc = cli.main(["--db", str(db_b), "import-bundle", str(tampered)])
    assert rc == 1
    assert "payload_hash mismatch" in capsys.readouterr().err
    assert event_rows(db_b) == []

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["schema_version"] = "ocbrain.bundle.v999"
    wrong_schema = tmp_path / "wrong_schema.bundle.json"
    wrong_schema.write_text(json.dumps(bundle), encoding="utf-8")
    rc = cli.main(["--db", str(db_b), "import-bundle", str(wrong_schema)])
    assert rc == 1
    assert "schema_version" in capsys.readouterr().err
    assert event_rows(db_b) == []


def test_import_skips_prohibited_items_in_handcrafted_bundle(tmp_path: Path) -> None:
    evidence = [
        {
            "evidence_id": "evd_deadbeef",
            "kind": "observation",
            "body": "hand-built prohibited item",
            "artifact_ref": None,
            "scope": PROHIBITED_SCOPE.to_dict(),
            "writer": "codex",
            "ts": "2026-07-02T00:00:00+00:00",
        }
    ]
    bundle = {
        "schema_version": "ocbrain.bundle.v1",
        "created_at": "2026-07-02T00:00:00+00:00",
        "origin": {"writer": "ocbrain-export", "hostname_free_label": "evil", "db_path_hash": "x"},
        "evidence": evidence,
        "count": 1,
        "payload_hash": bundle_payload_hash(evidence),
    }
    path = tmp_path / "handmade.bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    conn = connect(tmp_path / "brain_b.sqlite")
    init_db(conn)
    result = import_bundle(conn, path)
    assert result["counts"] == {"new": 0, "duplicates": 0, "skipped": 1}
    assert result["skipped"][0]["reason"] == "egress_policy:prohibited"
    assert (
        conn.execute("SELECT COUNT(*) AS n FROM brain_events").fetchone()["n"] == 0
    )
    conn.close()


def test_import_load_errors_are_bundle_errors(tmp_path: Path) -> None:
    conn = connect(tmp_path / "brain.sqlite")
    init_db(conn)
    missing = tmp_path / "missing.bundle.json"
    with pytest.raises(BundleImportError):
        import_bundle(conn, missing)
    garbage = tmp_path / "garbage.bundle.json"
    garbage.write_text("not json", encoding="utf-8")
    with pytest.raises(BundleImportError):
        import_bundle(conn, garbage)
    conn.close()


def test_payload_hash_is_order_independent() -> None:
    first = {"evidence_id": "evd_a", "body": "alpha"}
    second = {"evidence_id": "evd_b", "body": "beta"}
    assert bundle_payload_hash([first, second]) == bundle_payload_hash([second, first])
    assert bundle_payload_hash([first]) != bundle_payload_hash([second])


def test_cap_scope_for_import_only_downgrades_hosted_ok() -> None:
    assert cap_scope_for_import(HOSTED_OK_SCOPE).egress_policy == "approval_required"
    assert cap_scope_for_import(APPROVAL_SCOPE) is APPROVAL_SCOPE
    assert cap_scope_for_import(LOCAL_ONLY_SCOPE) is LOCAL_ONLY_SCOPE
    assert cap_scope_for_import(PROHIBITED_SCOPE) is PROHIBITED_SCOPE


def test_export_cli_requires_paired_scope_flags(tmp_path: Path, capsys) -> None:
    db_a = seeded_brain_a(tmp_path)
    rc = cli.main(
        [
            "--db",
            str(db_a),
            "export-bundle",
            "--output",
            str(tmp_path / "x.json"),
            "--scope-type",
            "project",
        ]
    )
    assert rc == 2
    assert "matched pairs" in capsys.readouterr().err


def test_dream_and_decide_compile_imported_evidence_into_beliefs(
    tmp_path: Path, capsys
) -> None:
    _, bundle_path, _ = exported_bundle_path(tmp_path, capsys)
    db_b = tmp_path / "brain_b.sqlite"
    assert cli.main(["--db", str(db_b), "import-bundle", str(bundle_path)]) == 0
    capsys.readouterr()

    conn = connect(db_b)
    init_db(conn)
    context = ScopeContext(project="bountiful")
    result = dream(conn, context=context)
    assert result["summary"]["proposals"] == 1
    proposal = result["proposed"][0]
    assert len(proposal["evidence_ids"]) == 2
    decide_compilation(
        conn,
        proposal_event_id=proposal["proposal_event_id"],
        decision="approve",
        actor="human:friend",
    )
    conn.commit()

    retrieved = retrieve(conn, "LoRa garden sensors", context=context)
    assert retrieved["items"], retrieved
    top = retrieved["items"][0]
    assert top["belief_id"] == proposal["belief_id"]
    assert "[REDACTED]" in top["body"]
    assert SECRET not in top["body"]
    conn.close()


def test_export_bundle_api_supports_query_and_limit(tmp_path: Path) -> None:
    db_a = seeded_brain_a(tmp_path)
    conn = connect(db_a)
    init_db(conn)
    result = export_bundle(
        conn,
        db_path=db_a,
        scopes=[("project", "project:bountiful")],
        query="watering schedule",
    )
    assert result["count"] == 1
    assert result["bundle"]["evidence"][0]["body"] == APPROVAL_BODY

    limited = export_bundle(
        conn,
        db_path=db_a,
        scopes=[("project", "project:bountiful")],
        limit=1,
    )
    assert limited["count"] == 1
    assert limited["bundle"]["count"] == 1
    conn.close()
