from ocbrain.db import connect, init_db, insert_candidate
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

    assert {"brain.search", "brain.get", "brain.digest"} <= names
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
