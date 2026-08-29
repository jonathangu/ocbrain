import json
import sqlite3

import pytest

from ocbrain import __version__
from ocbrain.db import (
    connect,
    init_db,
    link_knowledge_evidence,
    upsert_evidence,
    upsert_knowledge,
)
from ocbrain.mcp import canonical_tool_name, handle_request


def test_mcp_initialize_includes_agent_conduct_guardrails(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    response = handle_request(
        conn,
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
    )

    instructions = response["result"]["instructions"]
    assert response["result"]["serverInfo"]["version"] == __version__
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

    assert names == {
        "brain.context",
        "brain.source",
        "brain.search",
        "brain.get",
        "brain.digest",
        "brain.feedback",
        "brain.ingest",
        "brain.closeout",
    }
    # brain.propose is deleted in v0.2 (spec §5.1-4).
    assert "brain.propose" not in names
    assert "brain.teacher_request" not in names
    assert by_name["brain.search"]["annotations"] == {
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
        "readOnlyHint": True,
    }
    assert by_name["brain.feedback"]["annotations"]["destructiveHint"] is False
    assert by_name["brain.feedback"]["annotations"]["readOnlyHint"] is False

    # Plain dialect for unnamed clients: only semantically required fields are
    # required, and optional properties keep their declared types. The strict
    # all-keys-required shape is served only to clients that name a strict
    # harness at initialize (see test_mcp_schema_contract_v1).
    search_schema = by_name["brain.search"]["inputSchema"]
    assert search_schema.get("additionalProperties") is None
    assert search_schema["required"] == ["query"]
    assert search_schema["properties"]["limit"]["type"] == "integer"
    context_object = search_schema["properties"]["context"]
    assert context_object["type"] == "object"
    assert "required" not in context_object


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
    assert {
        "brain.ingest",
        "brain.forget",
        "brain.proposals",
        "brain.correct",
        "brain.proposal_decide",
    } <= names


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

    nullable_search = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "brain.search",
                "arguments": {
                    "query": "typecheck errors",
                    "limit": None,
                    "filters": None,
                    "context": {
                        "project": None,
                        "repo": None,
                        "client": None,
                        "task": None,
                        "session": None,
                        "runtime": None,
                    },
                    "cross_scope": None,
                    "at_ts": None,
                },
            },
        },
    )
    nullable_payload = json.loads(nullable_search["result"]["content"][0]["text"])
    assert nullable_payload

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


def test_mcp_propose_tool_is_removed(tmp_path):
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
    assert "not available in admin profile" in removed["error"]["message"]


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

    # Deprecated approval compatibility remains available only to admin clients.
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


def test_canonical_tool_name_maps_dot_free_cursor_names():
    allowed = {"brain.context", "brain.search", "brain.get"}
    assert canonical_tool_name("brain_context", allowed) == "brain.context"
    assert canonical_tool_name("brain_search", allowed) == "brain.search"
    # Canonical dotted names pass through untouched.
    assert canonical_tool_name("brain.context", allowed) == "brain.context"
    # Unknown names are returned unchanged so the profile gate rejects them.
    assert canonical_tool_name("brain_nope", allowed) == "brain_nope"
    # A dot-free spelling matching exactly one allowed tool still resolves.
    assert canonical_tool_name("brain_get", {"brain_get_now", "brain.get"}) == "brain.get"
    # A genuinely ambiguous spelling maps to nothing rather than guessing.
    ambiguous = canonical_tool_name("a_b_c", {"a.b_c", "a.b.c"})
    assert ambiguous == "a_b_c"


def test_mcp_dot_free_tool_name_reaches_dotted_tool(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "brain_context",
                "arguments": {"query": "dot-free client rewrite"},
            },
        },
    )

    assert "result" in response
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["query"] == "dot-free client rewrite"


def test_mcp_unknown_dot_free_tool_name_is_still_gated(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain_hack", "arguments": {}},
        },
    )

    assert response["error"]["code"] == -32001
    assert "not available" in response["error"]["message"]


def test_mcp_admin_only_tool_stays_gated_for_dot_free_runtime_name(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain_forget", "arguments": {"target": "x"}},
        },
    )

    assert response["error"]["code"] == -32001
    assert "not available in runtime profile" in response["error"]["message"]


# --- mcp lock-retry patch ------------------------------------------------------
class _FakeConn:
    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


def test_mcp_retries_on_database_locked(monkeypatch):
    from ocbrain import mcp

    calls = {"n": 0}

    def flaky_call_tool(conn, params, *, profile="runtime", provenance=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return {"ok": True}

    sleeps: list[float] = []
    monkeypatch.setattr(mcp, "call_tool", flaky_call_tool)
    monkeypatch.setattr(mcp.time, "sleep", lambda s: sleeps.append(s))

    conn = _FakeConn()
    result = mcp._call_tool_with_lock_retry(conn, {"name": "brain.ingest"})
    assert result == {"ok": True}
    assert calls["n"] == 3
    assert sleeps == [0.25, 0.25]
    assert conn.rollbacks == 2  # rolled back before each retry


def test_mcp_reraises_after_exhausting_retries(monkeypatch):
    from ocbrain import mcp

    def always_locked(conn, params, *, profile="runtime", provenance=None):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(mcp, "call_tool", always_locked)
    monkeypatch.setattr(mcp.time, "sleep", lambda s: None)
    with pytest.raises(sqlite3.OperationalError):
        mcp._call_tool_with_lock_retry(_FakeConn(), {"name": "brain.ingest"})


def test_mcp_non_lock_error_not_retried(monkeypatch):
    from ocbrain import mcp

    calls = {"n": 0}

    def other_error(conn, params, *, profile="runtime", provenance=None):
        calls["n"] += 1
        raise sqlite3.OperationalError("no such table: widgets")

    monkeypatch.setattr(mcp, "call_tool", other_error)
    with pytest.raises(sqlite3.OperationalError):
        mcp._call_tool_with_lock_retry(_FakeConn(), {"name": "brain.ingest"})
    assert calls["n"] == 1  # not retried


def test_request_exception_rolls_back_failed_closeout_before_next_commit(tmp_path, monkeypatch):
    """A serialized MCP error must not leave its partial writes pending."""
    from ocbrain import mcp_v1
    from ocbrain.core_v1 import init_core_v1

    db_path = tmp_path / "core.sqlite"
    conn = connect(db_path)
    init_core_v1(conn)
    observer = connect(db_path)
    real_record_evidence = mcp_v1.record_core_v1_evidence
    failed_once = False

    def fail_after_receipt(*args, **kwargs):
        nonlocal failed_once
        if kwargs.get("kind") == "task_closeout_summary" and not failed_once:
            failed_once = True
            raise RuntimeError("fault after closeout receipt insert")
        return real_record_evidence(*args, **kwargs)

    monkeypatch.setattr(mcp_v1, "record_core_v1_evidence", fail_after_receipt)

    def closeout(task_ref: str):
        return handle_request(
            conn,
            {
                "jsonrpc": "2.0",
                "id": task_ref,
                "method": "tools/call",
                "params": {
                    "name": "brain.closeout",
                    "arguments": {
                        "task_ref": task_ref,
                        "status": "completed",
                        "summary": f"{task_ref} closeout completed.",
                        "decision_impact": "none",
                        "context": {"project": "test", "task": task_ref},
                    },
                },
            },
        )

    failed = closeout("FIRST-FAILED")
    assert failed["error"]["code"] == -32000
    assert conn.execute("SELECT COUNT(*) FROM task_closeouts").fetchone()[0] == 0
    assert observer.execute("SELECT COUNT(*) FROM task_closeouts").fetchone()[0] == 0

    succeeded = closeout("SECOND-SUCCEEDS")
    assert "result" in succeeded
    assert [
        row["task_ref"]
        for row in observer.execute("SELECT task_ref FROM task_closeouts ORDER BY task_ref")
    ] == ["SECOND-SUCCEEDS"]
