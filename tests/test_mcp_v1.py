from __future__ import annotations

import io
import json
import sys

from ocbrain.core_v1 import (
    append_core_event,
    init_core_v1,
    record_core_v1_evidence,
)
from ocbrain.db import connect
from ocbrain.mcp import (
    ACTIVE_DB_CHANGED_ERROR_CODE,
    ACTIVE_DB_CHANGED_EXIT_CODE,
    handle_request,
    serve,
)
from ocbrain.scope import ScopeTag


def _payload(response):
    return json.loads(response["result"]["content"][0]["text"])


def _seed_v1(tmp_path):
    conn = connect(tmp_path / "core-v1.sqlite")
    init_core_v1(conn)
    scope = ScopeTag("project", "project:ocbrain")
    evidence_id, _event_id = record_core_v1_evidence(
        conn,
        body="OCBrain Shared Context gives every runtime one scoped evidence packet.",
        kind="observation",
        scope=scope,
        writer="test",
    )
    proposal_id = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": "belief:shared-context",
            "body": "Shared Context is the stable bridge across Codex, Claude, and OpenClaw.",
            "evidence_ids": [evidence_id],
            "scope": scope.to_dict(),
            "confidence": 0.95,
        },
        writer="test",
    )
    append_core_event(
        conn,
        "compilation_decided",
        {
            "proposal_event_id": proposal_id,
            "decision": "approve",
            "actor": "test",
            "edited_body": None,
            "reason": "fixture",
        },
        writer="test",
        project=True,
    )
    conn.commit()
    return conn


def test_v1_context_source_feedback_closeout_round_trip(tmp_path):
    conn = _seed_v1(tmp_path)
    context = _payload(
        handle_request(
            conn,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "brain.context",
                    "arguments": {
                        "query": "Shared Context bridge runtimes",
                        "context": {
                            "project": "ocbrain",
                            "runtime": "codex",
                            "task": "v1-acceptance",
                        },
                    },
                },
            },
        )
    )
    assert context["schema_version"] == "ocbrain.context.v1"
    assert context["core_schema"] == "ocbrain.core.v1"
    assert context["coverage"]["returned"] == 1
    assert context["coverage"]["source_handle_count"] == 1
    assert context["retrieval_use_status"] == "recorded"

    source_id = context["items"][0]["sources"][0]["id"]
    source = _payload(
        handle_request(
            conn,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "brain.source",
                    "arguments": {
                        "id": source_id,
                        "context": {"project": "ocbrain", "runtime": "codex"},
                    },
                },
            },
        )
    )
    assert source["hash_verified"] is True
    assert source["content"].startswith("OCBrain Shared Context")

    feedback = _payload(
        handle_request(
            conn,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "brain.feedback",
                    "arguments": {
                        "retrieval_use_id": context["retrieval_use_id"],
                        "outcome": "used",
                    },
                },
            },
        )
    )
    assert feedback["outcome"] == "used"

    closeout = _payload(
        handle_request(
            conn,
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "brain.closeout",
                    "arguments": {
                        "task_ref": "v1-acceptance",
                        "status": "completed",
                        "summary": "Verified the v1 shared context round trip.",
                        "retrieval_use_ids": [context["retrieval_use_id"]],
                        "decision_impact": "informed",
                        "verifier_refs": [
                            {
                                "uri": "pytest://test_mcp_v1",
                                "kind": "pytest",
                                "status": "passed",
                            }
                        ],
                    },
                },
            },
        )
    )
    assert closeout["schema_version"] == "ocbrain.closeout.v1"
    assert closeout["verification_status"] == "verified"
    assert (
        conn.execute(
            "SELECT affected_decision FROM retrieval_uses WHERE id=?",
            (context["retrieval_use_id"],),
        ).fetchone()[0]
        == 1
    )


def test_v1_runtime_and_admin_profiles_are_distinct(tmp_path):
    conn = _seed_v1(tmp_path)
    runtime = handle_request(conn, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    runtime_names = {item["name"] for item in runtime["result"]["tools"]}
    assert runtime_names == {
        "brain.context",
        "brain.source",
        "brain.search",
        "brain.digest",
        "brain.get",
        "brain.feedback",
        "brain.ingest",
        "brain.closeout",
    }

    admin = handle_request(
        conn,
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        profile="admin",
    )
    admin_names = {item["name"] for item in admin["result"]["tools"]}
    assert {"brain.correct", "brain.proposal_decide", "brain.forget"} <= admin_names
    correct_tool = next(
        item for item in admin["result"]["tools"] if item["name"] == "brain.correct"
    )
    assert correct_tool["inputSchema"]["properties"]["layer"]["enum"] == [
        "knowledge",
        "belief",
    ]
    assert "brain.teacher_request" not in admin_names
    assert "brain.mark_stale" not in admin_names


def test_v1_non_object_jsonrpc_frames_are_invalid_requests(tmp_path):
    conn = _seed_v1(tmp_path)

    for frame in ([], [{"jsonrpc": "2.0", "id": 1, "method": "ping"}], 5, "x", True, None):
        response = handle_request(conn, frame)
        assert response == {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32600,
                "message": "invalid request: message must be a JSON object",
            },
        }

    assert handle_request(conn, {"jsonrpc": "2.0", "id": 9, "method": "ping"}) == {
        "jsonrpc": "2.0",
        "id": 9,
        "result": {},
    }


def test_v1_non_object_params_are_invalid_params_not_internal_errors(tmp_path):
    conn = _seed_v1(tmp_path)

    for request_id, params in enumerate(([], "x", 5, True, None), start=1):
        response = handle_request(
            conn,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": params,
            },
        )
        assert response == {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32602,
                "message": "invalid params: params must be a JSON object",
            },
        }
        assert "attribute" not in response["error"]["message"].lower()

    resource_response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "resources/read",
            "params": ["brain://wiki/runtime-integration"],
        },
    )
    assert resource_response["error"]["code"] == -32602
    assert "attribute" not in resource_response["error"]["message"].lower()


def test_v1_malformed_notifications_are_never_answered(tmp_path):
    conn = _seed_v1(tmp_path)

    assert (
        handle_request(
            conn,
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": [],
            },
        )
        is None
    )
    assert handle_request(conn, {"jsonrpc": "2.0", "method": "unknown"}) is None


def test_v1_stdio_loop_survives_malformed_frames_and_keeps_runtime_surface(tmp_path, monkeypatch):
    frames = "\n".join(
        [
            "[]",
            '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":[]}',
            '{"jsonrpc":"2.0","id":3,"method":"tools/list"}',
        ]
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(frames + "\n"))
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    assert serve(tmp_path / "stdio-v1.sqlite") == 0

    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert responses[0]["error"]["code"] == -32600
    assert responses[1]["error"]["code"] == -32602
    assert {tool["name"] for tool in responses[2]["result"]["tools"]} == {
        "brain.context",
        "brain.source",
        "brain.search",
        "brain.digest",
        "brain.get",
        "brain.feedback",
        "brain.ingest",
        "brain.closeout",
    }


def test_v1_pointer_selected_server_exits_before_serving_after_pointer_change(
    tmp_path, monkeypatch
):
    first_db = tmp_path / "first.sqlite"
    second_db = tmp_path / "second.sqlite"
    active_db_file = tmp_path / "active-core.path"
    active_db_file.write_text(f"{first_db}\n")

    def frames():
        yield '{"jsonrpc":"2.0","id":1,"method":"ping"}\n'
        active_db_file.write_text(f"{second_db}\n")
        yield '{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n'

    monkeypatch.setattr(sys, "stdin", frames())
    output = io.StringIO()
    errors = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", errors)

    assert serve(first_db, active_db_file=active_db_file) == ACTIVE_DB_CHANGED_EXIT_CODE

    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert responses[0] == {"jsonrpc": "2.0", "id": 1, "result": {}}
    assert responses[1]["jsonrpc"] == "2.0"
    assert responses[1]["id"] == 2
    assert responses[1]["error"]["code"] == ACTIVE_DB_CHANGED_ERROR_CODE
    assert "reconnect" in responses[1]["error"]["message"]
    assert "reconnect" in errors.getvalue()
    assert not second_db.exists()


def test_v1_initialize_teaches_the_shared_context_closeout_contract(tmp_path):
    conn = _seed_v1(tmp_path)
    response = handle_request(
        conn,
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    )
    instructions = response["result"]["instructions"]
    assert "brain.context" in instructions
    assert "brain.source" in instructions
    assert "brain.feedback" in instructions
    assert "brain.closeout" in instructions
    assert "hosted judgment" in instructions
    assert "exhausted loop families" not in instructions


def test_v1_get_enforces_scope_and_context_does_not_ignore_at_ts(tmp_path):
    conn = _seed_v1(tmp_path)
    denied = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "brain.get",
                "arguments": {
                    "id": "belief:shared-context",
                    "context": {"project": "bountiful"},
                },
            },
        },
    )
    assert "scope does not match" in denied["error"]["message"]

    allowed = _payload(
        handle_request(
            conn,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "brain.get",
                    "arguments": {
                        "id": "belief:shared-context",
                        "context": {"project": "ocbrain", "runtime": "codex"},
                    },
                },
            },
        )
    )
    assert allowed["canonical_id"] == "belief:shared-context"

    append_core_event(
        conn,
        "correction_recorded",
        {
            "target_layer": "belief",
            "target_id": "belief:shared-context",
            "op": "retract",
            "author": "test",
            "hard": True,
        },
        writer="test",
        project=True,
    )
    lifecycle_denied = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "brain.get",
                "arguments": {
                    "id": "belief:shared-context",
                    "context": {"project": "ocbrain"},
                },
            },
        },
    )
    assert "non-current beliefs are not served" in lifecycle_denied["error"]["message"]

    unsupported = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "brain.context",
                "arguments": {
                    "query": "Shared Context",
                    "at_ts": "2026-07-01T00:00:00Z",
                    "context": {"project": "ocbrain"},
                },
            },
        },
    )
    assert "at_ts is not supported" in unsupported["error"]["message"]
