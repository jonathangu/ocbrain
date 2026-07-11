import json
import sqlite3

from ocbrain import __version__
from ocbrain.db import (
    connect,
    init_db,
    link_knowledge_evidence,
    upsert_evidence,
    upsert_knowledge,
)
from ocbrain.mcp import handle_request


def test_mcp_initialize_includes_agent_conduct_guardrails(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    response = handle_request(
        conn,
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
    )

    instructions = response["result"]["instructions"]
    assert response["result"]["serverInfo"]["version"] == __version__ == "0.4.0"
    assert "Surface assumptions or ambiguity before acting" in instructions
    assert "smallest change that satisfies the verified goal" in instructions
    assert "do not refactor unrelated code" in instructions
    assert "record the evidence" in instructions


def test_mcp_tools_are_knowledge_first(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    response = handle_request(conn, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = {tool["name"] for tool in response["result"]["tools"]}
    by_name = {tool["name"]: tool for tool in response["result"]["tools"]}

    assert {"brain.search", "brain.get", "brain.digest", "brain.feedback"} <= names
    # brain.propose is deleted in v0.2 (spec §5.1-4).
    assert "brain.propose" not in names
    # v0.2 §5.1-7: write tools are ungated — always listed, no --allow-writes needed.
    assert "brain.mark_stale" in names
    assert by_name["brain.search"]["annotations"] == {
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
        "readOnlyHint": True,
    }
    assert by_name["brain.feedback"]["annotations"]["destructiveHint"] is False
    assert by_name["brain.feedback"]["annotations"]["readOnlyHint"] is False
    assert by_name["brain.forget"]["annotations"]["destructiveHint"] is True


def test_mcp_write_tools_are_opt_in(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    response = handle_request(
        conn,
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        allow_writes=True,
    )
    names = {tool["name"] for tool in response["result"]["tools"]}

    # brain.propose is deleted in v0.2 (spec §5.1-4) — gone from every tool list.
    assert "brain.propose" not in names
    assert "brain.mark_stale" in names
    assert {"brain.ingest", "brain.forget", "brain.proposals"} <= names


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
    assert payload["retrieval_use_status"] == "recorded"

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
    assert digest_payload["retrieval_use_id"].startswith("ret_")
    assert digest_payload["retrieval_use_status"] == "recorded"

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


def test_mcp_contextual_search_returns_feedback_handle(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    search = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "brain.search",
                "arguments": {
                    "query": "connector acceptance",
                    "context": {
                        "runtime": "codex",
                        "project": "ocbrain",
                        "repo": "ocbrain",
                    },
                    "limit": 1,
                },
            },
        },
    )
    payload = json.loads(search["result"]["content"][0]["text"])
    retrieval_use_id = payload["retrieval_use_id"]
    assert retrieval_use_id.startswith("ret_")
    assert payload["retrieval_use_status"] == "recorded"

    feedback = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "brain.feedback",
                "arguments": {
                    "retrieval_use_id": retrieval_use_id,
                    "outcome": "irrelevant",
                },
            },
        },
    )
    feedback_payload = json.loads(feedback["result"]["content"][0]["text"])
    assert feedback_payload == {
        "outcome": "irrelevant",
        "retrieval_use_id": retrieval_use_id,
    }


def test_mcp_contextual_search_survives_busy_retrieval_log(tmp_path):
    path = tmp_path / "ocbrain.sqlite"
    reader = connect(path)
    init_db(reader)
    reader.execute("PRAGMA busy_timeout=1")
    locker = sqlite3.connect(path)
    locker.execute("PRAGMA busy_timeout=1")
    locker.execute("BEGIN IMMEDIATE")
    try:
        search = handle_request(
            reader,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "brain.search",
                    "arguments": {
                        "query": "available while writer is active",
                        "context": {"runtime": "codex", "project": "ocbrain"},
                        "limit": 1,
                    },
                },
            },
        )
    finally:
        locker.rollback()
        locker.close()

    payload = json.loads(search["result"]["content"][0]["text"])
    assert payload["query"] == "available while writer is active"
    assert payload["retrieval_use_id"] is None
    assert payload["retrieval_use_status"] == "database_busy"


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


def test_mcp_propose_tool_removed_and_mark_stale_ungated(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="capability",
        gate="auto",
        origin="loop",
        slug="verified-test-workflow",
        title="Verified test workflow",
        body_uri="/tmp/result.json",
        status="candidate",
        risk="high",
        confidence=0.82,
    )
    conn.commit()

    # brain.propose no longer exists — dispatch fails as an unknown tool (spec §5.1-4).
    removed = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain.propose", "arguments": {"id": knowledge_id}},
        },
        allow_writes=True,
    )
    assert "error" in removed
    assert "unknown tool" in removed["error"]["message"]

    # v0.2 §5.1-7: mark_stale is ungated — it succeeds with no allow_writes argument.
    stale = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "brain.mark_stale", "arguments": {"id": knowledge_id}},
        },
    )
    assert "result" in stale
    assert json.loads(stale["result"]["content"][0]["text"])["status"] == "stale"


def test_mcp_mark_stale_ungated_without_allow_writes(tmp_path):
    # v0.2 §5.1-7: the knowledge gate is removed. Write tools are always
    # available; --allow-writes is a deprecated no-op. mark_stale succeeds with
    # no allow_writes argument at all.
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

    stale = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain.mark_stale", "arguments": {"id": knowledge_id}},
        },
    )
    assert "error" not in stale
    payload = json.loads(stale["result"]["content"][0]["text"])
    assert payload == {"id": knowledge_id, "status": "stale"}


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

    # v0.2 §5.1-7: approval feedback is ungated — succeeds with no allow_writes argument.
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
