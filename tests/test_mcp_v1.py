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


def _seed_delivery_v1(tmp_path):
    conn = connect(tmp_path / "delivery-v1.sqlite")
    init_core_v1(conn)
    fixtures = {
        "local": (
            "belief:delivery-local",
            "Delivery target local-only sentinel must stay on this Mac.",
            ScopeTag(
                "global",
                "global:doctrine",
                visibility="internal",
                egress_policy="local_only",
            ),
        ),
        "hosted": (
            "belief:delivery-hosted",
            "Delivery target hosted-safe sentinel may reach hosted models.",
            ScopeTag(
                "global",
                "global:doctrine",
                visibility="internal",
                egress_policy="hosted_ok",
            ),
        ),
        "confidential": (
            "belief:delivery-confidential",
            "Delivery target confidential sentinel must never reach hosted models.",
            ScopeTag(
                "global",
                "global:doctrine",
                visibility="confidential",
                egress_policy="hosted_ok",
            ),
        ),
    }
    ids = {}
    for key, (belief_id, body, scope) in fixtures.items():
        evidence_scope = fixtures["local"][2] if key == "hosted" else scope
        evidence_id, _event_id = record_core_v1_evidence(
            conn,
            body=body,
            kind="observation",
            scope=evidence_scope,
            writer="test",
        )
        proposal_id = append_core_event(
            conn,
            "compilation_proposed",
            {
                "belief_id": belief_id,
                "body": body,
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
                "reason": "delivery fixture",
            },
            writer="test",
            project=True,
        )
        ids[key] = {"belief": belief_id, "evidence": evidence_id}
    conn.commit()
    return conn, ids


def _tool_call(name, arguments, *, request_id=1):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def test_v1_context_source_feedback_closeout_round_trip(tmp_path):
    conn = _seed_v1(tmp_path)
    context_response = handle_request(
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
    context_text = context_response["result"]["content"][0]["text"]
    context = json.loads(context_text)
    assert context["schema_version"] == "ocbrain.context.v1"
    assert context["core_schema"] == "ocbrain.core.v1"
    assert context["coverage"]["returned"] == 1
    assert context["coverage"]["source_handle_count"] == 1
    assert context["retrieval_use_status"] == "recorded"
    assert context["coverage"]["serialized_bytes"] == len(context_text.encode())
    assert len(context_text.encode()) <= context["coverage"]["hard_packet_limit_bytes"]

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


def test_v1_source_provenance_sample_is_bounded(tmp_path):
    conn = _seed_v1(tmp_path)
    contexts = []
    for request_id in range(12):
        contexts.append(
            _payload(
                handle_request(
                    conn,
                    _tool_call(
                        "brain.context",
                        {
                            "query": "Shared Context bridge runtimes",
                            "context": {"project": "ocbrain", "runtime": "codex"},
                        },
                        request_id=request_id + 1,
                    ),
                )
            )
        )
    source_id = contexts[-1]["items"][0]["sources"][0]["id"]
    source = _payload(
        handle_request(
            conn,
            _tool_call(
                "brain.source",
                {
                    "id": source_id,
                    "context": {"project": "ocbrain", "runtime": "codex"},
                },
                request_id=20,
            ),
        )
    )
    assert source["issued_by_count"] == 12
    assert len(source["issued_by_retrieval_use_ids"]) == 8


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


def test_idle_reader_preserves_multiple_frames_buffered_in_one_input(tmp_path, monkeypatch):
    db = tmp_path / "core.sqlite"
    frames = "".join(
        [
            '{"jsonrpc":"2.0","id":1,"method":"ping"}\n',
            '{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n',
        ]
    )
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(frames))
    monkeypatch.setattr(sys, "stdout", output)

    assert serve(db, idle_timeout_seconds=0.2) == 0
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [response["id"] for response in responses] == [1, 2]


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
    assert "at_ts" in unsupported["error"]["message"]
    assert "not supported" in unsupported["error"]["message"]


def test_v1_hosted_delivery_filters_context_search_digest_and_resource(tmp_path):
    conn, ids = _seed_delivery_v1(tmp_path)
    arguments = {
        "query": "Delivery target sentinel",
        "context": {"project": "ocbrain", "runtime": "codex"},
    }

    local_context = _payload(handle_request(conn, _tool_call("brain.context", arguments)))
    assert {item["id"] for item in local_context["items"]} == {
        fixture["belief"] for fixture in ids.values()
    }
    assert local_context["delivery_target"] == "local_model"

    hosted_context = _payload(
        handle_request(
            conn,
            _tool_call("brain.context", arguments, request_id=2),
            delivery_target="hosted_model",
        )
    )
    assert hosted_context["delivery_target"] == "hosted_model"
    assert [item["id"] for item in hosted_context["items"]] == [ids["hosted"]["belief"]]
    assert hosted_context["items"][0]["scope"] == {
        "scope_type": "global",
        "scope_id": "global:doctrine",
        "visibility": "internal",
        "egress_policy": "hosted_ok",
        "provenance": "explicit",
    }
    assert hosted_context["items"][0]["evidence_ids"] == []

    hosted_search = _payload(
        handle_request(
            conn,
            _tool_call("brain.search", arguments, request_id=3),
            delivery_target="hosted_model",
        )
    )
    assert [item["id"] for item in hosted_search["items"]] == [ids["hosted"]["belief"]]

    hosted_digest = _payload(
        handle_request(
            conn,
            _tool_call(
                "brain.digest",
                {"context": {"project": "ocbrain", "runtime": "codex"}},
                request_id=4,
            ),
            delivery_target="hosted_model",
        )
    )
    assert [item["id"] for item in hosted_digest["current"]] == [ids["hosted"]["belief"]]

    resource_response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/read",
            "params": {"uri": "brain://digest/current"},
        },
        delivery_target="hosted_model",
    )
    hosted_resource = json.loads(resource_response["result"]["contents"][0]["text"])
    assert [item["id"] for item in hosted_resource["current"]] == [ids["hosted"]["belief"]]

    for payload in (hosted_context, hosted_search, hosted_digest, hosted_resource):
        encoded = json.dumps(payload, sort_keys=True)
        assert "local-only sentinel" not in encoded
        assert "confidential sentinel" not in encoded
        assert "local_only" not in encoded
        assert '"visibility": "confidential"' not in encoded


def test_v1_hosted_delivery_blocks_local_source_and_get_even_with_private_flags(tmp_path):
    conn, ids = _seed_delivery_v1(tmp_path)
    arguments = {
        "query": "Delivery target sentinel",
        "context": {"project": "ocbrain", "runtime": "codex"},
    }
    local_context = _payload(handle_request(conn, _tool_call("brain.context", arguments)))
    local_source_by_id = {item["id"]: item["sources"][0]["id"] for item in local_context["items"]}
    hosted_context = _payload(
        handle_request(
            conn,
            _tool_call("brain.context", arguments, request_id=2),
            delivery_target="hosted_model",
        )
    )
    hosted_source_id = hosted_context["items"][0]["sources"][0]["id"]

    denied_source = handle_request(
        conn,
        _tool_call(
            "brain.source",
            {
                "id": local_source_by_id[ids["local"]["belief"]],
                "context": {"project": "ocbrain", "runtime": "codex"},
            },
            request_id=3,
        ),
        delivery_target="hosted_model",
    )
    assert denied_source["error"]["code"] == -32001
    assert "not eligible for hosted_model delivery" in denied_source["error"]["message"]

    hosted_source = _payload(
        handle_request(
            conn,
            _tool_call(
                "brain.source",
                {
                    "id": hosted_source_id,
                    "context": {"project": "ocbrain", "runtime": "codex"},
                },
                request_id=4,
            ),
            delivery_target="hosted_model",
        )
    )
    assert hosted_source["delivery_target"] == "hosted_model"
    assert "hosted-safe sentinel" in hosted_source["content"]

    for request_id, key in enumerate(("local", "confidential"), start=5):
        denied_get = handle_request(
            conn,
            _tool_call(
                "brain.get",
                {
                    "id": ids[key]["belief"],
                    "context": {"project": "ocbrain", "runtime": "codex"},
                    "include_private": True,
                    "cross_scope": True,
                },
                request_id=request_id,
            ),
            delivery_target="hosted_model",
        )
        assert denied_get["error"]["code"] == -32001
        assert "not eligible for hosted_model delivery" in denied_get["error"]["message"]

    hosted_get = _payload(
        handle_request(
            conn,
            _tool_call(
                "brain.get",
                {
                    "id": ids["hosted"]["belief"],
                    "context": {"project": "ocbrain", "runtime": "codex"},
                },
                request_id=7,
            ),
            delivery_target="hosted_model",
        )
    )
    assert hosted_get["delivery_target"] == "hosted_model"
    assert hosted_get["canonical_id"] == ids["hosted"]["belief"]


def test_v1_hosted_delivery_redacts_local_source_paths_and_raw_metadata(tmp_path):
    conn = connect(tmp_path / "redaction-v1.sqlite")
    init_core_v1(conn)
    scope = ScopeTag(
        "project",
        "project:bountiful",
        visibility="internal",
        egress_policy="hosted_ok",
    )
    local_path = "/Users/example/.ocbrain/private/current-truth.md"
    evidence_id, _event_id = record_core_v1_evidence(
        conn,
        body="Hosted-safe verified Bountiful source body.",
        kind="curated_source_attestation",
        scope=scope,
        writer="test",
        artifact_ref=local_path,
    )
    proposal_id = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": "curated:bountiful:path-redaction",
            "belief_type": "curated_fact",
            "body": "Hosted-safe verified Bountiful source body.",
            "evidence_ids": [evidence_id],
            "scope": scope.to_dict(),
            "confidence": 0.95,
            "attributes": {
                "title": "Safe fact",
                "manifest_path": local_path,
                "source_attestations": [{"ref": "S1", "path": local_path, "sha256": "a" * 64}],
            },
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
        },
        writer="test",
        project=True,
    )
    conn.commit()
    arguments = {
        "query": "verified Bountiful source",
        "context": {"project": "bountiful", "runtime": "codex"},
    }
    context = _payload(
        handle_request(
            conn,
            _tool_call("brain.context", arguments),
            delivery_target="hosted_model",
        )
    )
    source = context["items"][0]["sources"][0]
    assert source["uri"].startswith("ocbrain://evidence/")
    assert "/Users/" not in json.dumps(context)

    local_context = _payload(handle_request(conn, _tool_call("brain.context", arguments)))
    local_source_id = local_context["items"][0]["sources"][0]["id"]
    replayed_source = _payload(
        handle_request(
            conn,
            _tool_call(
                "brain.source",
                {
                    "id": local_source_id,
                    "context": {"project": "bountiful", "runtime": "codex"},
                },
                request_id=4,
            ),
            delivery_target="hosted_model",
        )
    )
    assert replayed_source["uri"].startswith("ocbrain://evidence/")
    assert "/Users/" not in json.dumps(replayed_source)

    belief = _payload(
        handle_request(
            conn,
            _tool_call(
                "brain.get",
                {
                    "id": "curated:bountiful:path-redaction",
                    "context": {"project": "bountiful"},
                },
                request_id=2,
            ),
            delivery_target="hosted_model",
        )
    )
    assert belief["attributes"]["source_attestations"] == [{"ref": "S1", "sha256": "a" * 64}]
    assert "manifest_path" not in belief["attributes"]
    assert "/Users/" not in json.dumps(belief)

    evidence = _payload(
        handle_request(
            conn,
            _tool_call(
                "brain.get",
                {"id": evidence_id, "context": {"project": "bountiful"}},
                request_id=3,
            ),
            delivery_target="hosted_model",
        )
    )
    assert "source_uri" not in evidence
    assert "artifact_uri" not in evidence
    assert "metadata" not in evidence
    assert "/Users/" not in json.dumps(evidence)


def test_v1_source_handles_follow_current_linkage_and_policy_revocation(tmp_path):
    conn = connect(tmp_path / "revocation-v1.sqlite")
    init_core_v1(conn)
    hosted_scope = ScopeTag(
        "project",
        "project:bountiful",
        visibility="internal",
        egress_policy="hosted_ok",
    )
    local_scope = ScopeTag(
        "project",
        "project:bountiful",
        visibility="internal",
        egress_policy="local_only",
    )
    belief_id = "curated:bountiful:revocable"

    def approve(body, scope):
        evidence_id, _event_id = record_core_v1_evidence(
            conn,
            body=body,
            kind="curated_source_attestation",
            scope=scope,
            writer="test",
        )
        proposal_id = append_core_event(
            conn,
            "compilation_proposed",
            {
                "belief_id": belief_id,
                "belief_type": "curated_fact",
                "body": body,
                "evidence_ids": [evidence_id],
                "scope": scope.to_dict(),
                "confidence": 0.9,
            },
            writer="test",
        )
        append_core_event(
            conn,
            "compilation_decided",
            {"proposal_event_id": proposal_id, "decision": "approve", "actor": "test"},
            writer="test",
            project=True,
        )
        conn.commit()
        return evidence_id

    first_evidence = approve("First revocable Bountiful fact.", hosted_scope)
    first_context = _payload(
        handle_request(
            conn,
            _tool_call(
                "brain.context",
                {
                    "query": "revocable Bountiful fact",
                    "context": {"project": "bountiful"},
                },
            ),
            delivery_target="hosted_model",
        )
    )
    first_source = first_context["items"][0]["sources"][0]["id"]

    second_evidence = approve("Replacement revocable Bountiful fact.", hosted_scope)
    assert second_evidence != first_evidence
    obsolete = handle_request(
        conn,
        _tool_call(
            "brain.source",
            {"id": first_source, "context": {"project": "bountiful"}},
            request_id=2,
        ),
        delivery_target="hosted_model",
    )
    assert obsolete["error"]["code"] == -32001
    assert "no longer current support" in obsolete["error"]["message"]

    replacement_context = _payload(
        handle_request(
            conn,
            _tool_call(
                "brain.context",
                {
                    "query": "replacement revocable fact",
                    "context": {"project": "bountiful"},
                },
                request_id=3,
            ),
            delivery_target="hosted_model",
        )
    )
    replacement_source = replacement_context["items"][0]["sources"][0]["id"]
    assert approve("Replacement revocable Bountiful fact.", local_scope) == second_evidence
    evidence_scope = conn.execute(
        "SELECT egress_policy FROM evidence_objects WHERE evidence_id=?",
        (second_evidence,),
    ).fetchone()
    assert evidence_scope["egress_policy"] == "local_only"
    revoked = handle_request(
        conn,
        _tool_call(
            "brain.source",
            {"id": replacement_source, "context": {"project": "bountiful"}},
            request_id=4,
        ),
        delivery_target="hosted_model",
    )
    assert revoked["error"]["code"] == -32001
    assert "not eligible for hosted_model" in revoked["error"]["message"]


def test_v1_stdio_is_hosted_and_tool_arguments_cannot_override_delivery(tmp_path, monkeypatch):
    conn, _ids = _seed_delivery_v1(tmp_path)
    conn.close()
    frames = "\n".join(
        [
            '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}',
            json.dumps(
                _tool_call(
                    "brain.context",
                    {
                        "query": "Delivery target sentinel",
                        "context": {"project": "ocbrain"},
                        "delivery_target": "local_model",
                    },
                    request_id=2,
                )
            ),
        ]
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(frames + "\n"))
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    assert serve(tmp_path / "delivery-v1.sqlite") == 0

    initialize, override = [json.loads(line) for line in output.getvalue().splitlines()]
    assert initialize["result"]["serverInfo"]["deliveryTarget"] == "hosted_model"
    assert override["error"]["code"] == -32602
    assert "server-controlled" in override["error"]["message"]
