import json

from ocbrain.db import (
    EventInput,
    connect,
    init_db,
    insert_candidate,
    link_knowledge_evidence,
    upsert_event,
    upsert_evidence,
    upsert_knowledge,
)
from ocbrain.mcp import handle_request
from ocbrain.schema import Candidate, Scope, Target


def test_mcp_initialize(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    response = handle_request(conn, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})

    assert response["result"]["serverInfo"]["name"] == "ocbrain"
    assert "tools" in response["result"]["capabilities"]


def test_mcp_tools_list_is_read_only_by_default(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    response = handle_request(conn, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = {tool["name"] for tool in response["result"]["tools"]}

    assert {"brain.search", "brain.get", "brain.digest", "brain.feedback"} <= names
    assert "brain.propose" not in names


def test_mcp_tools_list_can_opt_into_write_tools(tmp_path):
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


def test_mcp_initialized_notification_has_no_response(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    response = handle_request(conn, {"jsonrpc": "2.0", "method": "notifications/initialized"})

    assert response is None


def test_mcp_search_missing_query_is_invalid_params(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain.search", "arguments": {}},
        },
    )

    assert response["error"]["code"] == -32602


def test_mcp_get_private_candidate_requires_explicit_flag(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    candidate_id = insert_candidate(
        conn,
        Candidate(
            target=Target.MEMORY,
            title="Private note",
            body="Private note body",
            confidence=0.8,
            scope=Scope.PRIVATE,
        ),
    )
    conn.commit()

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain.get", "arguments": {"id": candidate_id}},
        },
    )

    assert response["error"]["code"] == -32001


def test_mcp_get_draft_candidate_requires_explicit_flag(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    candidate_id = insert_candidate(
        conn,
        Candidate(
            target=Target.WIKI,
            title="Draft note",
            body="Draft note body",
            confidence=0.8,
        ),
    )
    conn.commit()

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain.get", "arguments": {"id": candidate_id}},
        },
    )

    assert response["error"]["code"] == -32001
    assert "include_draft" in response["error"]["message"]


def test_mcp_get_approved_candidate_by_default(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    candidate_id = insert_candidate(
        conn,
        Candidate(
            target=Target.WIKI,
            title="Approved note",
            body="Approved note body",
            confidence=0.8,
        ),
    )
    conn.execute("UPDATE candidates SET status = 'approved' WHERE id = ?", (candidate_id,))
    conn.commit()

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain.get", "arguments": {"id": candidate_id}},
        },
    )

    assert response["result"]["content"][0]["type"] == "text"
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["retrieval_use_id"].startswith("ret_")
    row = conn.execute(
        "SELECT * FROM retrieval_uses WHERE artifact_or_candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    assert row["runtime"] == "mcp"
    assert row["query"] == "brain.get"


def test_mcp_get_current_knowledge_by_default(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime:codex",
        predicate="uses_shared_brain",
        value_bool=True,
        status="current",
        inject=True,
        confidence=0.9,
    )
    conn.commit()

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain.get", "arguments": {"id": knowledge_id}},
        },
    )

    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["object_kind"] == "knowledge"
    assert payload["id"] == knowledge_id
    assert payload["retrieval_use_id"].startswith("ret_")


def test_mcp_get_candidate_knowledge_requires_draft_flag(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="capability",
        gate="human",
        slug="dangerous-capability",
        title="Dangerous capability",
        status="candidate",
        risk="high",
    )
    conn.commit()

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain.get", "arguments": {"id": knowledge_id}},
        },
    )

    assert response["error"]["code"] == -32001
    assert "include_draft" in response["error"]["message"]


def test_mcp_digest_returns_current_knowledge_not_just_counts(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime:codex",
        predicate="uses_shared_brain",
        value_bool=True,
        status="current",
        inject=True,
        confidence=0.9,
    )
    upsert_knowledge(
        conn,
        knowledge_type="capability",
        gate="human",
        slug="needs-approval",
        title="Needs approval",
        status="candidate",
        risk="high",
    )
    conn.commit()

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain.digest", "arguments": {}},
        },
    )

    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["counts"]["knowledge"] == 2
    assert payload["memory"][0]["subject"] == "runtime:codex"
    assert payload["values"][0]["value"] is True
    assert payload["capabilities"] == []


def test_mcp_lists_and_renders_current_wiki_resources(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    evidence_id = upsert_evidence(
        conn,
        source_type="closeout",
        source_runtime="codex",
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


def test_mcp_mark_stale_is_write_gated_and_updates_knowledge(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime:codex",
        predicate="old_fact",
        value_text="old",
        status="current",
    )
    conn.commit()
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "brain.mark_stale", "arguments": {"id": knowledge_id}},
    }

    denied = handle_request(conn, request)
    assert denied["error"]["code"] == -32001

    response = handle_request(conn, request, allow_writes=True)
    assert "result" in response
    row = conn.execute(
        "SELECT status, invalidation_reason FROM knowledge WHERE id = ?",
        (knowledge_id,),
    ).fetchone()
    assert row["status"] == "stale"
    assert row["invalidation_reason"] == "user_request"


def test_mcp_search_records_retrieval_use(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    event = EventInput(
        id="evt_search",
        source_type="doc",
        source_uri="/tmp/search.md",
        content_hash="hash-search",
        title="MCP search",
        summary="Architecture uses MCP search.",
        body="Architecture uses MCP search.",
    )
    assert upsert_event(conn, event)
    conn.commit()

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain.search", "arguments": {"query": "MCP search"}},
        },
    )

    assert "result" in response
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload[0]["retrieval_use_id"].startswith("ret_")
    row = conn.execute(
        "SELECT * FROM retrieval_uses WHERE artifact_or_candidate_id = ?",
        (event.id,),
    ).fetchone()
    assert row["runtime"] == "mcp"
    assert row["query"] == "brain.search:MCP search"


def test_mcp_feedback_updates_retrieval_use(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    event = EventInput(
        id="evt_feedback",
        source_type="doc",
        source_uri="/tmp/feedback.md",
        content_hash="hash-feedback",
        title="Feedback search",
        summary="Claude and Codex share ocbrain context.",
        body="Claude and Codex share ocbrain context.",
    )
    assert upsert_event(conn, event)
    conn.commit()
    search_response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain.search", "arguments": {"query": "Codex Claude"}},
        },
    )
    payload = json.loads(search_response["result"]["content"][0]["text"])
    retrieval_use_id = payload[0]["retrieval_use_id"]

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "brain.feedback",
                "arguments": {
                    "retrieval_use_id": retrieval_use_id,
                    "outcome": "helpful",
                    "note": "used in answer",
                },
            },
        },
    )

    assert "result" in response
    row = conn.execute("SELECT * FROM retrieval_uses WHERE id = ?", (retrieval_use_id,)).fetchone()
    assert row["outcome"] == "helpful"
    assert row["note"] == "used in answer"
