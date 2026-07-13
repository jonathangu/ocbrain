from __future__ import annotations

import json

from ocbrain.core_v1 import (
    append_core_event,
    init_core_v1,
    record_core_v1_evidence,
)
from ocbrain.db import connect
from ocbrain.mcp import handle_request
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
    assert conn.execute(
        "SELECT affected_decision FROM retrieval_uses WHERE id=?",
        (context["retrieval_use_id"],),
    ).fetchone()[0] == 1


def test_v1_runtime_and_admin_profiles_are_distinct(tmp_path):
    conn = _seed_v1(tmp_path)
    runtime = handle_request(
        conn, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    )
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
    assert "brain.teacher_request" not in admin_names
    assert "brain.mark_stale" not in admin_names


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
