from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ocbrain import runtime_call
from ocbrain.core_v1 import init_core_v1
from ocbrain.db import connect
from ocbrain.runtime_call import invoke


def test_one_shot_runtime_fallback_records_closeout(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    conn = connect(db_path)
    init_core_v1(conn)
    conn.close()

    result = invoke(
        db_path,
        "brain.closeout",
        {
            "summary": "Verified fallback closeout.",
            "status": "completed",
            "task_ref": "fallback-test",
            "context": {"project": "test", "task": "fallback-test"},
            "decision_impact": "none",
            "retrieval_use_ids": [],
            "artifact_refs": [],
            "verifier_refs": [
                {
                    "uri": "pytest://runtime-fallback",
                    "kind": "pytest",
                    "status": "passed",
                    "detail": "One-shot runtime call completed.",
                }
            ],
        },
    )

    assert result["status"] == "completed"
    conn = connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM task_closeouts").fetchone()[0] == 1


def test_one_shot_runtime_fallback_rejects_admin_tool(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    conn = connect(db_path)
    init_core_v1(conn)
    conn.close()

    try:
        invoke(db_path, "brain.correct", {})
    except PermissionError as exc:
        assert "runtime tools only" in str(exc)
    else:
        raise AssertionError("admin tool unexpectedly allowed")


def test_one_shot_cli_passes_hosted_delivery_target_to_handle_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments_file = tmp_path / "arguments.json"
    arguments_file.write_text("{}", encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_handle_request(conn, request, **kwargs):
        observed["request"] = request
        observed["profile"] = kwargs.get("profile")
        observed["delivery_target"] = kwargs.get("delivery_target")
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"delivery_target": kwargs.get("delivery_target")}),
                    }
                ]
            },
        }

    monkeypatch.setattr(runtime_call, "handle_request", fake_handle_request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ocbrain-runtime-call",
            "brain.briefing",
            "--db",
            str(tmp_path / "ocbrain.sqlite"),
            "--delivery-target",
            "hosted_model",
            "--arguments-file",
            str(arguments_file),
        ],
    )

    assert runtime_call.main() == 0
    assert observed["profile"] == "runtime"
    assert observed["delivery_target"] == "hosted_model"
    assert observed["request"] == {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "brain.briefing", "arguments": {}},
    }
    assert json.loads(capsys.readouterr().out) == {"delivery_target": "hosted_model"}


@pytest.mark.parametrize("field", ["delivery_target", "deliveryTarget"])
def test_one_shot_hosted_delivery_rejects_model_supplied_override(
    tmp_path: Path, field: str
) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    conn = connect(db_path)
    init_core_v1(conn)
    conn.close()

    with pytest.raises(RuntimeError, match="server-controlled"):
        invoke(
            db_path,
            "brain.briefing",
            {field: "local_model"},
            delivery_target="hosted_model",
        )
