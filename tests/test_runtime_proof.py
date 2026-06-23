import json

from ocbrain.db import connect, init_db
from tools.runtime_proof_fixture import build_proof


def test_runtime_proof_fixture_generates_reviewed_excerpts_and_mcp_get_gates(tmp_path):
    proof = build_proof(tmp_path)
    ids = proof["candidate_ids"]

    for path in proof["excerpts"].values():
        text = tmp_path.joinpath(path).read_text(encoding="utf-8")
        assert "Runtime excerpt reads approved workspace context" in text
        assert "MCP get serves proposed project memory" in text
        assert "Draft candidate requires opt-in" not in text
        assert "Private candidate requires opt-in" not in text

    mcp = proof["mcp"]
    assert "result" in mcp["approved_workspace_default"]
    assert "result" in mcp["proposed_project_default"]
    assert mcp["draft_workspace_default"]["error"]["code"] == -32001
    assert "include_draft" in mcp["draft_workspace_default"]["error"]["message"]
    assert "result" in mcp["draft_workspace_include_draft"]
    assert mcp["approved_private_default"]["error"]["code"] == -32001
    assert "include_private" in mcp["approved_private_default"]["error"]["message"]
    assert "result" in mcp["approved_private_include_private"]

    draft_payload = json.loads(
        mcp["draft_workspace_include_draft"]["result"]["content"][0]["text"]
    )
    private_payload = json.loads(
        mcp["approved_private_include_private"]["result"]["content"][0]["text"]
    )
    assert draft_payload["id"] == ids["draft_workspace"]
    assert private_payload["id"] == ids["approved_private"]

    conn = connect(tmp_path / "runtime-proof.sqlite")
    init_db(conn)
    rows = conn.execute("SELECT status, scope, COUNT(*) AS count FROM candidates GROUP BY 1, 2")
    distribution = {(row["status"], row["scope"]): row["count"] for row in rows}
    assert distribution == {
        ("approved", "private"): 1,
        ("approved", "workspace"): 1,
        ("draft", "workspace"): 1,
        ("proposed", "project"): 1,
    }
