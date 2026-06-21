from ocbrain.db import connect, init_db
from ocbrain.mcp import handle_request


def test_mcp_initialize(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    response = handle_request(conn, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})

    assert response["result"]["serverInfo"]["name"] == "ocbrain"
    assert "tools" in response["result"]["capabilities"]


def test_mcp_tools_list_includes_get_and_propose(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    response = handle_request(conn, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = {tool["name"] for tool in response["result"]["tools"]}

    assert {"brain.search", "brain.get", "brain.digest", "brain.propose"} <= names
