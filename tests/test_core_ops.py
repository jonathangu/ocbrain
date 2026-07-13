from __future__ import annotations

import json
import sqlite3
import sys

import pytest

from ocbrain import core_ops
from ocbrain.cli import build_parser, main
from ocbrain.core_ops import (
    FORBIDDEN_SYNC_STAGES,
    backup_database,
    database_status,
    restore_database,
    sha256_file,
    stdio_mcp_smoke,
    sync_core,
)
from ocbrain.db import connect, init_db
from ocbrain.events import append_event


def _database(path):
    conn = connect(path)
    init_db(conn)
    conn.commit()
    return conn


def test_sync_is_bounded_and_explicitly_forbids_non_core_stages(tmp_path):
    path = tmp_path / "brain.sqlite"
    conn = _database(path)
    append_event(
        conn,
        "evidence_recorded",
        {
            "evidence_id": "evd_one",
            "kind": "observation",
            "body": "local evidence",
            "artifact_ref": None,
            "scope": {
                "scope_type": "workspace",
                "scope_id": "workspace",
                "visibility": "private",
                "egress_policy": "local_only",
            },
        },
    )
    conn.commit()
    conn.close()

    result = sync_core(path, max_events=10, time_budget_seconds=5)

    assert result["status"] == "ok"
    assert result["processed_events"] == 1
    assert result["policy"]["hosted_calls"] == 0
    assert result["policy"]["scheduled"] is False
    assert result["policy"]["network_allowed"] is False
    assert result["policy"]["forbidden_stages"] == list(FORBIDDEN_SYNC_STAGES)
    assert set(result["policy"]["stages"]) == {
        "event_projection",
        "sqlite_quick_check",
        "foreign_key_check",
    }


def test_sync_refuses_before_writing_when_event_cap_would_be_exceeded(tmp_path):
    path = tmp_path / "brain.sqlite"
    conn = _database(path)
    for index in range(2):
        append_event(
            conn,
            "evidence_recorded",
            {
                "evidence_id": f"evd_{index}",
                "kind": "observation",
                "body": f"evidence {index}",
                "artifact_ref": None,
                "scope": {
                    "scope_type": "workspace",
                    "scope_id": "workspace",
                    "visibility": "private",
                    "egress_policy": "local_only",
                },
            },
        )
    conn.commit()
    conn.close()

    result = sync_core(path, max_events=1, time_budget_seconds=5)

    assert result["status"] == "bounded_refusal"
    assert result["changed"] is False
    check = connect(path)
    assert check.execute("SELECT COUNT(*) FROM projection_cursor").fetchone()[0] == 0


def test_cli_sync_never_loads_config_or_dispatches_companion_work(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "brain.sqlite"
    _database(path).close()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("sync crossed its local core boundary")

    monkeypatch.setattr(core_ops.subprocess, "run", forbidden)
    loaded_before = set(sys.modules)

    assert main(["--db", str(path), "sync"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["policy"]["hosted_calls"] == 0
    assert result["policy"]["forbidden_stages"] == list(FORBIDDEN_SYNC_STAGES)
    newly_loaded = set(sys.modules) - loaded_before
    assert not any(name.startswith(("ocbrain_ops", "ocbrain_training")) for name in newly_loaded)


def test_status_does_not_create_a_missing_database(tmp_path):
    path = tmp_path / "missing.sqlite"
    result = database_status(path)
    assert result["status"] == "missing"
    assert not path.exists()


def test_real_stdio_mcp_subprocess_smoke():
    result = stdio_mcp_smoke(timeout_seconds=15)
    assert result["healthy"] is True, result
    assert result["response_ids"] == [1, 2, 3]
    assert result["protocol_version"]
    assert result["tool_count"] >= 3


def test_backup_and_restore_are_verified_and_fresh_only(tmp_path):
    source = tmp_path / "brain.sqlite"
    conn = _database(source)
    conn.execute(
        "INSERT INTO harvest_watermarks(domain, stream, watermark, updated_at) "
        "VALUES ('test', 'stream', 'one', '2026-07-13T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()
    backup = tmp_path / "backup.sqlite"
    restored = tmp_path / "restored.sqlite"

    backed_up = backup_database(source, backup)
    restored_result = restore_database(backup, restored)

    assert backed_up["sha256"] == sha256_file(backup)
    assert backed_up["integrity"] == "ok"
    assert restored_result["integrity"] == "ok"
    assert restored_result["live_database_replaced"] is False
    source_conn = sqlite3.connect(source)
    restored_conn = sqlite3.connect(restored)
    assert source_conn.execute("SELECT COUNT(*) FROM harvest_watermarks").fetchone()[0] == 1
    assert restored_conn.execute("SELECT COUNT(*) FROM harvest_watermarks").fetchone()[0] == 1
    source_conn.close()
    restored_conn.close()
    with pytest.raises(FileExistsError):
        restore_database(backup, restored)


def test_core_operation_cli_surfaces_parse_and_status_is_json(tmp_path, capsys):
    parser = build_parser()
    assert parser.parse_args(["status"]).func.__name__ == "cmd_status"
    assert parser.parse_args(["sync"]).func.__name__ == "cmd_sync"
    assert parser.parse_args(["doctor"]).func.__name__ == "cmd_doctor"
    assert parser.parse_args(["runtime-check"]).func.__name__ == "cmd_runtime_check"
    assert parser.parse_args(["backup", "--output", "x"]).func.__name__ == "cmd_backup"
    assert (
        parser.parse_args(["restore", "--backup", "x", "--output-db", "y"]).func.__name__
        == "cmd_restore"
    )
    migrate = parser.parse_args(
        [
            "core-migrate-v1",
            "--core-db",
            "core",
            "--archive-db",
            "archive",
            "--manifest",
            "manifest",
            "--plan",
        ]
    )
    assert migrate.func.__name__ == "cmd_core_migrate_v1" and migrate.plan is True

    db = tmp_path / "brain.sqlite"
    _database(db).close()
    assert main(["--db", str(db), "status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["database"]["healthy"] is True
    assert payload["operating_model"]["scheduler_installed_by_core"] is False
