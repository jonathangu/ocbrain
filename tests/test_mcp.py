import io
import json
import sys

from ocbrain.db import (
    connect,
    init_db,
    link_knowledge_evidence,
    upsert_evidence,
    upsert_knowledge,
)
from ocbrain.events import decide_compilation, propose_compilation, record_tombstone
from ocbrain.mcp import handle_request, serve
from ocbrain.scope import ScopeTag


def seed_belief(conn, belief_id, body, scope, confidence=0.8):
    proposal = propose_compilation(
        conn,
        belief_id=belief_id,
        body=body,
        evidence_ids=[f"evd:{belief_id}"],
        scope=scope,
        confidence=confidence,
    )
    decide_compilation(conn, proposal_event_id=proposal, decision="approve")
    conn.commit()


def call_tool_request(conn, name, arguments, *, request_id=1, allow_writes=False):
    return handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        allow_writes=allow_writes,
    )


def test_mcp_initialize_includes_agent_conduct_guardrails(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    response = handle_request(
        conn,
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
    )

    instructions = response["result"]["instructions"]
    assert "Surface assumptions or ambiguity before acting" in instructions
    assert "smallest change that satisfies the verified goal" in instructions
    assert "do not refactor unrelated code" in instructions
    assert "record the evidence" in instructions


def test_mcp_tools_are_knowledge_first(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    response = handle_request(conn, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = {tool["name"] for tool in response["result"]["tools"]}

    assert {"brain.search", "brain.get", "brain.digest", "brain.feedback"} <= names
    assert "brain.propose" not in names
    assert "brain.mark_stale" not in names


def test_mcp_write_tools_are_opt_in(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    response = handle_request(
        conn,
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        allow_writes=True,
    )
    names = {tool["name"] for tool in response["result"]["tools"]}

    assert "brain.propose" in names
    assert "brain.mark_stale" in names


def test_mcp_get_current_knowledge_by_default_and_candidate_with_flag(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    current_id = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime:codex",
        predicate="shared_brain",
        value_bool=True,
        status="current",
        inject=True,
    )
    candidate_id = upsert_knowledge(
        conn,
        knowledge_type="doc",
        gate="auto",
        slug="candidate-doc",
        title="Candidate doc",
        body_uri="/tmp/doc.md",
        doc_kind="wiki",
        status="candidate",
    )
    conn.commit()

    current = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain.get", "arguments": {"id": current_id}},
        },
    )
    payload = json.loads(current["result"]["content"][0]["text"])
    assert payload["object_kind"] == "knowledge"
    assert payload["retrieval_use_id"].startswith("ret_")

    denied = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "brain.get", "arguments": {"id": candidate_id}},
        },
    )
    assert denied["error"]["code"] == -32001
    assert "include_candidate" in denied["error"]["message"]

    allowed = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "brain.get",
                "arguments": {"id": candidate_id, "include_candidate": True},
            },
        },
    )
    assert "result" in allowed


def test_mcp_digest_search_feedback_and_filters(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="loop:repo-quality-loop",
        predicate="typecheck_errors",
        value_numeric=9,
        status="current",
        inject=True,
        loop_tags={"loop_id": "repo-quality-loop", "family": "typecheck_narrowing"},
    )
    conn.commit()

    digest = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain.digest", "arguments": {}},
        },
    )
    digest_payload = json.loads(digest["result"]["content"][0]["text"])
    assert digest_payload["memory"][0]["predicate"] == "typecheck_errors"

    search = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "brain.search",
                "arguments": {
                    "query": "typecheck errors",
                    "filters": {"loop_id": "repo-quality-loop"},
                },
            },
        },
    )
    search_payload = json.loads(search["result"]["content"][0]["text"])
    retrieval_use_id = search_payload[0]["retrieval_use_id"]

    feedback = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "brain.feedback",
                "arguments": {"retrieval_use_id": retrieval_use_id, "outcome": "helpful"},
            },
        },
    )
    assert "result" in feedback
    row = conn.execute("SELECT outcome FROM retrieval_uses WHERE id = ?", (retrieval_use_id,))
    assert row.fetchone()["outcome"] == "helpful"


def test_mcp_wiki_resource_renders_evidence(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    evidence_id = upsert_evidence(
        conn,
        source_type="closeout",
        source_uri="/tmp/wiki-proof.md",
        content_hash="hash-wiki-proof",
        claim="Runtime integration docs were verified.",
        verifier_status="passed",
    )
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="doc",
        gate="auto",
        slug="runtime-integration",
        title="Runtime integration",
        body_uri="/tmp/wiki-proof.md",
        doc_kind="wiki",
        status="current",
        confidence=0.87,
    )
    link_knowledge_evidence(conn, knowledge_id, evidence_id)
    conn.commit()

    listed = handle_request(conn, {"jsonrpc": "2.0", "id": 1, "method": "resources/list"})
    uris = {resource["uri"] for resource in listed["result"]["resources"]}
    assert "brain://wiki/runtime-integration" in uris

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "resources/read",
            "params": {"uri": "brain://wiki/runtime-integration"},
        },
    )

    content = response["result"]["contents"][0]
    assert content["mimeType"] == "text/markdown"
    assert "# Runtime integration" in content["text"]
    assert "Runtime integration docs were verified." in content["text"]


def test_mcp_propose_and_mark_stale_are_write_gated(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    evidence_id = upsert_evidence(
        conn,
        source_type="loop_iteration",
        source_uri="/tmp/result.json",
        content_hash="hash-capability-result",
        claim="Repeated verified success suggests a reusable test workflow.",
        verifier_status="passed",
    )
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="capability",
        gate="human",
        slug="verified-test-workflow",
        title="Verified test workflow",
        body_uri="/tmp/result.json",
        status="candidate",
        risk="high",
        confidence=0.82,
    )
    link_knowledge_evidence(conn, knowledge_id, evidence_id, relation="derived_from")
    conn.commit()

    denied = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain.propose", "arguments": {"id": knowledge_id}},
        },
    )
    assert denied["error"]["code"] == -32001

    proposed = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "brain.propose",
                "arguments": {"id": knowledge_id, "output_dir": str(tmp_path / "proposals")},
            },
        },
        allow_writes=True,
    )
    payload = json.loads(proposed["result"]["content"][0]["text"])
    assert payload["proposal"].endswith(f"knowledge-capability-{knowledge_id}.md")

    stale = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "brain.mark_stale", "arguments": {"id": knowledge_id}},
        },
        allow_writes=True,
    )
    assert "result" in stale


def test_mcp_mark_stale_denied_without_allow_writes(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="capability",
        gate="human",
        slug="stale-candidate-workflow",
        title="Stale candidate workflow",
        status="candidate",
        risk="high",
    )
    conn.commit()

    denied = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain.mark_stale", "arguments": {"id": knowledge_id}},
        },
    )
    assert denied["error"]["code"] == -32001
    assert "--allow-writes" in denied["error"]["message"]


def test_mcp_feedback_approves_or_rejects_human_gated_knowledge(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    approve_id = upsert_knowledge(
        conn,
        knowledge_type="capability",
        gate="human",
        slug="approved-workflow",
        title="Approved workflow",
        status="candidate",
        risk="high",
    )
    reject_id = upsert_knowledge(
        conn,
        knowledge_type="capability",
        gate="human",
        slug="rejected-workflow",
        title="Rejected workflow",
        status="candidate",
        risk="high",
    )
    conn.commit()

    denied = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "brain.feedback",
                "arguments": {"id": approve_id, "decision": "approve", "actor": "jon"},
            },
        },
    )
    assert denied["error"]["code"] == -32001

    approved = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "brain.feedback",
                "arguments": {"id": approve_id, "decision": "approve", "actor": "jon"},
            },
        },
        allow_writes=True,
    )
    approved_payload = json.loads(approved["result"]["content"][0]["text"])
    approved_row = conn.execute(
        "SELECT status, approved_by FROM knowledge WHERE id = ?",
        (approve_id,),
    ).fetchone()

    assert approved_payload["status"] == "current"
    assert approved_row["status"] == "current"
    assert approved_row["approved_by"] == "jon"

    rejected = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "brain.feedback",
                "arguments": {
                    "id": reject_id,
                    "decision": "reject",
                    "reason": "not ready",
                },
            },
        },
        allow_writes=True,
    )
    rejected_payload = json.loads(rejected["result"]["content"][0]["text"])
    rejected_row = conn.execute(
        "SELECT status, invalidation_reason FROM knowledge WHERE id = ?",
        (reject_id,),
    ).fetchone()

    assert rejected_payload["status"] == "archived"
    assert rejected_row["status"] == "archived"
    assert rejected_row["invalidation_reason"] == "not ready"


def test_mcp_get_confidential_belief_requires_include_private(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    seed_belief(
        conn,
        "belief:acme-secret",
        "Acme lane registry holds the confidential client onboarding token.",
        ScopeTag(
            "client",
            "client:acme",
            visibility="confidential",
            egress_policy="local_only",
        ),
    )

    denied = call_tool_request(conn, "brain.get", {"id": "belief:acme-secret"})
    assert denied["error"]["code"] == -32001
    assert "include_private" in denied["error"]["message"]
    assert "onboarding token" not in json.dumps(denied)

    allowed = call_tool_request(
        conn,
        "brain.get",
        {"id": "belief:acme-secret", "include_private": True},
        request_id=2,
    )
    payload = json.loads(allowed["result"]["content"][0]["text"])
    assert payload["object_kind"] == "belief"
    assert "onboarding token" in payload["body"]


def test_mcp_get_tombstoned_belief_gated_and_shredded_body_stays_shredded(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    seed_belief(
        conn,
        "belief:old-stack",
        "Bountiful stack uses Express with Neon.",
        ScopeTag("project", "project:bountiful", egress_policy="hosted_ok"),
    )
    record_tombstone(conn, target="belief:old-stack", mode="shred", reason="cleanup")
    conn.commit()

    denied = call_tool_request(conn, "brain.get", {"id": "belief:old-stack"})
    assert denied["error"]["code"] == -32001
    assert "include_candidate" in denied["error"]["message"]

    allowed = call_tool_request(
        conn,
        "brain.get",
        {"id": "belief:old-stack", "include_candidate": True},
        request_id=2,
    )
    payload = json.loads(allowed["result"]["content"][0]["text"])
    assert payload["status"] == "tombstoned"
    assert payload["body"] == "[shredded by tombstone]"
    assert "Express with Neon" not in allowed["result"]["content"][0]["text"]


def test_mcp_non_object_frames_answered_with_invalid_request(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    for frame in ([], [{"jsonrpc": "2.0", "id": 1, "method": "ping"}], 5, "x", True, None):
        response = handle_request(conn, frame)
        assert response["error"]["code"] == -32600
        assert response["id"] is None

    follow_up = handle_request(conn, {"jsonrpc": "2.0", "id": 9, "method": "ping"})
    assert follow_up == {"jsonrpc": "2.0", "id": 9, "result": {}}


def test_mcp_serve_loop_survives_non_object_frames(tmp_path, monkeypatch):
    frames = "\n".join(
        ["[]", "5", '"x"', "null", "{not json", '{"jsonrpc":"2.0","id":7,"method":"ping"}']
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(frames + "\n"))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)

    exit_code = serve(tmp_path / "ocbrain.sqlite")

    assert exit_code == 0
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [line["error"]["code"] for line in lines[:4]] == [-32600] * 4
    assert lines[4]["error"]["code"] == -32700
    assert lines[5] == {"jsonrpc": "2.0", "id": 7, "result": {}}


def test_mcp_notifications_are_never_answered(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    assert handle_request(conn, {"jsonrpc": "2.0", "method": "bogus"}) is None
    assert (
        handle_request(
            conn,
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "brain.get", "arguments": {}},
            },
        )
        is None
    )
    assert (
        handle_request(
            conn,
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "brain.mark_stale", "arguments": {"id": "nope"}},
            },
        )
        is None
    )

    answered = handle_request(conn, {"jsonrpc": "2.0", "id": 4, "method": "bogus"})
    assert answered["error"]["code"] == -32601


def test_mcp_preview_excluded_reports_buckets_without_scope_metadata(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    seed_belief(
        conn,
        "belief:acme-secret",
        "Acme lane registry is confidential.",
        ScopeTag(
            "client",
            "client:acme",
            visibility="confidential",
            egress_policy="local_only",
        ),
    )
    seed_belief(
        conn,
        "belief:foreign-stack",
        "Foreign project uses Express.",
        ScopeTag("project", "project:foreign", egress_policy="hosted_ok"),
    )
    seed_belief(
        conn,
        "belief:bountiful-stack",
        "Bountiful stack uses React 19.",
        ScopeTag("project", "project:bountiful", egress_policy="hosted_ok"),
    )

    response = call_tool_request(
        conn,
        "brain.preview",
        {"query": "stack React Express registry", "context": {"project": "bountiful"}},
    )
    text = response["result"]["content"][0]["text"]
    payload = json.loads(text)

    assert payload["excluded_count"] == 2
    assert payload["excluded_reasons"] == {
        "confidential_scope_mismatch": 1,
        "scope_mismatch": 1,
    }
    assert "excluded" not in payload
    assert "belief:acme-secret" not in text
    assert "client:acme" not in text
    assert "belief:foreign-stack" not in text
    assert "project:foreign" not in text


def test_mcp_egress_recording_requires_allow_writes(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    def audit_count():
        return conn.execute("SELECT COUNT(*) AS n FROM egress_audits").fetchone()["n"]

    denied = call_tool_request(conn, "brain.egress_preview", {"record": True})
    denied_payload = json.loads(denied["result"]["content"][0]["text"])
    assert denied_payload["recorded"] is False
    assert "--allow-writes" in denied_payload["record_denied_reason"]
    assert "audit_id" not in denied_payload
    assert audit_count() == 0

    recorded = call_tool_request(
        conn, "brain.egress_preview", {"record": True}, request_id=2, allow_writes=True
    )
    recorded_payload = json.loads(recorded["result"]["content"][0]["text"])
    assert recorded_payload["recorded"] is True
    assert recorded_payload["audit_id"]
    assert audit_count() == 1

    teacher_denied = call_tool_request(conn, "brain.teacher_request", {}, request_id=3)
    teacher_denied_payload = json.loads(teacher_denied["result"]["content"][0]["text"])
    assert teacher_denied_payload["recorded"] is False
    assert "--allow-writes" in teacher_denied_payload["record_denied_reason"]
    assert audit_count() == 1

    teacher_recorded = call_tool_request(
        conn, "brain.teacher_request", {}, request_id=4, allow_writes=True
    )
    teacher_recorded_payload = json.loads(teacher_recorded["result"]["content"][0]["text"])
    assert teacher_recorded_payload["recorded"] is True
    assert audit_count() == 2

    dry = call_tool_request(
        conn, "brain.teacher_request", {"dry_run": True}, request_id=5, allow_writes=True
    )
    dry_payload = json.loads(dry["result"]["content"][0]["text"])
    assert dry_payload["recorded"] is False
    assert "record_denied_reason" not in dry_payload
    assert audit_count() == 2


def test_mcp_feedback_rejects_evidence_layer(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    listed = handle_request(
        conn, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, allow_writes=True
    )
    feedback_tool = next(
        tool for tool in listed["result"]["tools"] if tool["name"] == "brain.feedback"
    )
    layer_enum = feedback_tool["inputSchema"]["properties"]["layer"]["enum"]
    assert layer_enum == ["knowledge", "belief"]

    rejected = call_tool_request(
        conn,
        "brain.feedback",
        {"target": "belief:x", "layer": "evidence", "op": "mark_wrong"},
        request_id=2,
        allow_writes=True,
    )
    assert rejected["error"]["code"] == -32602
    assert "knowledge or belief" in rejected["error"]["message"]
    assert conn.execute("SELECT COUNT(*) AS n FROM brain_events").fetchone()["n"] == 0
