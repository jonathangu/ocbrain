import json

from ocbrain.db import EventInput, connect, init_db, insert_candidate, upsert_event
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
