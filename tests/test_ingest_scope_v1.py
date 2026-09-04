"""D2: brain.ingest honors an explicit scope that narrows; widenings propose.

The v1 MCP tool schema has advertised a `scope` argument for a long time, and
the v1 dispatcher dropped it on the floor: `ingest_v1` computed the inferred
write scope and never looked at the client's. These tests pin the replacement
behavior at both layers — the operation and the wire dispatcher.
"""

from __future__ import annotations

import json

import pytest

from ocbrain.core_v1 import get_core_v1_evidence, init_core_v1
from ocbrain.db import connect
from ocbrain.mcp import handle_request, tool_list
from ocbrain.mcp_v1 import ingest_v1
from ocbrain.scope import ScopeContext, ScopeTag


def _conn(tmp_path):
    conn = connect(tmp_path / "ingest-scope.sqlite")
    init_core_v1(conn)
    return conn


def _task_context() -> ScopeContext:
    return ScopeContext(task="scope-honor", runtime="test")


def test_ingest_v1_still_infers_when_no_scope_requested(tmp_path):
    conn = _conn(tmp_path)
    result = ingest_v1(
        conn,
        body="Plain observation with no explicit scope.",
        kind="observation",
        context=_task_context(),
        writer="test",
        session_id=None,
        artifact_ref=None,
    )
    stored = get_core_v1_evidence(conn, result["evidence_id"])
    assert stored["scope"]["scope_id"] == "task:scope-honor"
    assert stored["scope"]["egress_policy"] == "approval_required"
    assert stored["scope"]["provenance"] == "inferred"
    assert result["scope_decision"] == "inferred"
    assert "hosted_egress_proposal_event_id" not in result


def test_ingest_v1_honors_narrowing_scope_with_explicit_provenance(tmp_path):
    conn = _conn(tmp_path)
    requested = ScopeTag(
        "task",
        "task:scope-honor",
        visibility="secret",
        egress_policy="prohibited",
        provenance="inferred",
    )
    result = ingest_v1(
        conn,
        body="Credential-adjacent observation the client narrows itself.",
        kind="observation",
        context=_task_context(),
        writer="test",
        session_id=None,
        artifact_ref=None,
        requested_scope=requested,
    )
    stored = get_core_v1_evidence(conn, result["evidence_id"])
    assert stored["scope"]["scope_id"] == "task:scope-honor"
    assert stored["scope"]["visibility"] == "secret"
    assert stored["scope"]["egress_policy"] == "prohibited"
    assert stored["scope"]["provenance"] == "explicit"
    assert result["scope_decision"] == "explicit"
    assert "hosted_egress_proposal_event_id" not in result


def test_ingest_v1_equal_scope_counts_as_narrowing_without_proposal(tmp_path):
    conn = _conn(tmp_path)
    inferred = ScopeTag(
        "task",
        "task:scope-honor",
        visibility="internal",
        egress_policy="approval_required",
        provenance="inferred",
    )
    result = ingest_v1(
        conn,
        body="Same scope, spelled by the client.",
        kind="observation",
        context=_task_context(),
        writer="test",
        session_id=None,
        artifact_ref=None,
        requested_scope=inferred,
    )
    stored = get_core_v1_evidence(conn, result["evidence_id"])
    assert stored["scope"]["provenance"] == "explicit"
    assert result["scope_decision"] == "inferred"
    assert "hosted_egress_proposal_event_id" not in result


def test_ingest_v1_sibling_scope_identity_is_proposed_not_applied(tmp_path):
    conn = _conn(tmp_path)
    requested = ScopeTag(
        "task",
        "task:unrelated-sibling",
        visibility="internal",
        egress_policy="approval_required",
        provenance="explicit",
    )
    result = ingest_v1(
        conn,
        body="A lateral task retarget must never be called narrowing.",
        kind="observation",
        context=_task_context(),
        writer="test",
        session_id=None,
        artifact_ref=None,
        requested_scope=requested,
    )
    stored = get_core_v1_evidence(conn, result["evidence_id"])
    assert stored is not None
    assert stored["scope"]["scope_id"] == "task:scope-honor"
    assert result["scope_decision"] == "hosted_egress_proposal"
    assert result["widened"] == ["scope_identity"]


def test_scope_tag_rejects_mismatched_nonlegacy_prefix():
    with pytest.raises(ValueError, match="task: prefix"):
        ScopeTag("task", "project:not-a-task")


def test_unprovable_cross_family_retarget_is_proposed(tmp_path):
    conn = _conn(tmp_path)
    result = ingest_v1(
        conn,
        body="A task id is not provably contained by a project id.",
        kind="observation",
        context=ScopeContext(project="scope-honor", runtime="test"),
        writer="test",
        session_id=None,
        artifact_ref=None,
        requested_scope=ScopeTag(
            "task",
            "task:scope-honor",
            visibility="internal",
            egress_policy="approval_required",
        ),
    )
    stored = get_core_v1_evidence(conn, result["evidence_id"])
    assert stored is not None
    assert stored["scope"]["scope_id"] == "project:scope-honor"
    assert result["scope_decision"] == "hosted_egress_proposal"
    assert result["widened"] == ["scope_identity"]


def test_brain_ingest_schema_includes_session_scope():
    ingest = next(
        tool
        for tool in tool_list(profile="runtime", core_v1=True)
        if tool["name"] == "brain.ingest"
    )
    scope_types = (
        ingest["inputSchema"]["properties"]["scope"]["properties"]["scope_type"]["enum"]
    )
    assert "session" in scope_types


def test_ingest_v1_records_widening_request_as_proposal(tmp_path):
    conn = _conn(tmp_path)
    requested = ScopeTag(
        "project",
        "project:coframe",
        visibility="internal",
        egress_policy="hosted_ok",
        provenance="inferred",
    )
    result = ingest_v1(
        conn,
        body="Please host-deliver this, said the unattended agent.",
        kind="observation",
        context=_task_context(),
        writer="test",
        session_id=None,
        artifact_ref=None,
        requested_scope=requested,
    )
    # The evidence itself stays under the inferred scope, never the requested one.
    stored = get_core_v1_evidence(conn, result["evidence_id"])
    assert stored["scope"]["scope_id"] == "task:scope-honor"
    assert stored["scope"]["egress_policy"] == "approval_required"
    assert stored["scope"]["provenance"] == "inferred"
    assert result["scope_decision"] == "hosted_egress_proposal"
    # The request is on the ledger, named by the ingest receipt.
    proposal_event_id = result["hosted_egress_proposal_event_id"]
    proposal_row = conn.execute(
        "SELECT * FROM brain_events WHERE id=?", (proposal_event_id,)
    ).fetchone()
    assert proposal_row is not None
    assert proposal_row["kind"] == "hosted_egress_proposal"
    body = json.loads(proposal_row["body_json"])
    assert body["evidence_id"] == result["evidence_id"]
    assert body["requested_scope"]["egress_policy"] == "hosted_ok"
    assert body["inferred_scope"]["scope_id"] == "task:scope-honor"
    assert "host-deliver" in body["body_head"]
    # The receipt names the widening itself: which ladders the request exceeded.
    assert result["requested_scope"]["scope_id"] == "project:coframe"
    assert result["inferred_scope"]["scope_id"] == "task:scope-honor"
    assert result["widened"] == ["scope_type", "egress_policy"]
    # No belief may exist yet: the widening was proposed, not applied.
    beliefs = conn.execute("SELECT COUNT(*) FROM current_beliefs").fetchone()[0]
    assert beliefs == 0


def test_ingest_v1_narrowing_result_has_no_widened_field(tmp_path):
    """A narrowing (or equal) request is applied outright: no `widened` key may
    appear on the receipt, because nothing was exceeded."""
    conn = _conn(tmp_path)
    requested = ScopeTag(
        "task",
        "task:scope-honor",
        visibility="secret",
        egress_policy="prohibited",
        provenance="inferred",
    )
    result = ingest_v1(
        conn,
        body="Narrowed by the client itself, nothing exceeded.",
        kind="observation",
        context=_task_context(),
        writer="test",
        session_id=None,
        artifact_ref=None,
        requested_scope=requested,
    )
    assert result["scope_decision"] == "explicit"
    assert "widened" not in result
    assert "requested_scope" not in result
    assert "inferred_scope" not in result


def test_ingest_v1_widening_request_is_listed_by_proposals_v1(tmp_path):
    conn = _conn(tmp_path)
    requested = ScopeTag(
        "global",
        "global:doctrine",
        visibility="internal",
        egress_policy="hosted_ok",
        provenance="inferred",
    )
    ingest_v1(
        conn,
        body="Asking for global hosted doctrine without a human.",
        kind="observation",
        context=_task_context(),
        writer="test",
        session_id=None,
        artifact_ref=None,
        requested_scope=requested,
    )
    from ocbrain.mcp_v1 import proposals_v1

    listed = proposals_v1(conn, limit=50, include_decided=False)
    kinds = {
        proposal["proposal_event_id"]: proposal for proposal in listed["proposals"]
    }
    pending = [
        proposal
        for proposal in listed["proposals"]
        if proposal.get("requested_scope", {}).get("scope_id") == "global:doctrine"
    ]
    assert pending, "the widening request must appear in event-proposals output"
    proposal = pending[0]
    assert proposal["proposal_event_id"] in kinds
    assert proposal["decided"] is False


def test_mcp_dispatcher_passes_requested_scope_to_ingest_v1(tmp_path):
    conn = _conn(tmp_path)
    narrowing = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "brain.ingest",
                "arguments": {
                    "body": "Narrowed through the wire dispatcher.",
                    "kind": "observation",
                    "context": {"task": "scope-honor", "runtime": "test"},
                    "scope": {
                        "scope_type": "task",
                        "scope_id": "task:scope-honor",
                        "visibility": "secret",
                        "egress_policy": "prohibited",
                    },
                },
            },
        },
    )
    payload = narrowing["result"]["content"][0]["text"]
    result = json.loads(payload)
    assert result["scope_decision"] == "explicit"
    stored = get_core_v1_evidence(conn, result["evidence_id"])
    assert stored["scope"]["egress_policy"] == "prohibited"

    widening = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "brain.ingest",
                "arguments": {
                    "body": "A widening request through the wire dispatcher.",
                    "kind": "observation",
                    "context": {"task": "scope-honor", "runtime": "test"},
                    "scope": {
                        "scope_type": "project",
                        "scope_id": "project:coframe",
                        "visibility": "internal",
                        "egress_policy": "hosted_ok",
                    },
                },
            },
        },
    )
    widened = json.loads(widening["result"]["content"][0]["text"])
    assert widened["scope_decision"] == "hosted_egress_proposal"
    stored = get_core_v1_evidence(conn, widened["evidence_id"])
    assert stored["scope"]["egress_policy"] == "approval_required"
    assert widened["hosted_egress_proposal_event_id"]
