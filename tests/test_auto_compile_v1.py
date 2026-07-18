"""Unattended evidence/closeout -> belief promotion (automatic_activation).

Off by default (promotion stays human-gated). When an operator enables it, an
agent's ingested evidence and closeouts become served, cross-client-recallable
beliefs without human review -- while never widening egress beyond local_only.
"""

from __future__ import annotations

import json

from ocbrain.core_v1 import (
    automatic_activation_enabled,
    init_core_v1,
    set_automatic_activation,
)
from ocbrain.db import connect
from ocbrain.mcp import handle_request
from ocbrain.mcp_v1 import auto_compile_scope
from ocbrain.scope import ScopeContext


def _core(tmp_path, *, auto: bool):
    conn = connect(tmp_path / "auto-core.sqlite")
    init_core_v1(conn)
    if auto:
        set_automatic_activation(conn, True)
    conn.commit()
    return conn


def _call(conn, name, arguments, *, request_id=1, delivery="local_model"):
    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        delivery_target=delivery,
    )
    if "error" in response:
        return {"_error": response["error"]["message"]}
    return json.loads(response["result"]["content"][0]["text"])


CC = {"client": "claude-code", "project": "demo"}


def test_default_off_ingest_is_not_served(tmp_path):
    conn = _core(tmp_path, auto=False)
    assert automatic_activation_enabled(conn) is False
    ingest = _call(conn, "brain.ingest", {"body": "alpha deploy needs migrations", "context": CC})
    assert ingest["kind"] == "evidence_recorded"
    assert "auto_compiled_belief_id" not in ingest
    search = _call(conn, "brain.search", {"query": "alpha deploy migrations", "context": CC})
    assert search["coverage"]["returned"] == 0


def test_enabled_ingest_is_promoted_and_served(tmp_path):
    conn = _core(tmp_path, auto=True)
    ingest = _call(conn, "brain.ingest", {"body": "alpha deploy needs migrations", "context": CC})
    assert ingest["kind"] == "evidence_recorded_and_compiled"
    belief_id = ingest["auto_compiled_belief_id"]
    assert belief_id
    search = _call(conn, "brain.search", {"query": "alpha deploy migrations", "context": CC})
    assert [item["id"] for item in search["items"]] == [belief_id]
    context = _call(conn, "brain.context", {"query": "alpha deploy migrations", "context": CC})
    assert context["coverage"]["returned"] == 1


def test_promotion_is_idempotent(tmp_path):
    conn = _core(tmp_path, auto=True)
    first = _call(conn, "brain.ingest", {"body": "same body here", "context": CC})
    second = _call(
        conn, "brain.ingest", {"body": "same body here", "context": CC}, request_id=2
    )
    assert first["auto_compiled_belief_id"] == second["auto_compiled_belief_id"]
    assert conn.execute("SELECT COUNT(*) FROM current_beliefs").fetchone()[0] == 1


def test_promotion_is_cross_client_on_shared_project(tmp_path):
    conn = _core(tmp_path, auto=True)
    # Claude Code writes (with a narrow task in context); promotion still scopes
    # to the shared project so another client recalls it.
    written = _call(
        conn,
        "brain.ingest",
        {"body": "shared deploy runbook fact", "context": {**CC, "task": "task:x"}},
    )
    belief_id = written["auto_compiled_belief_id"]
    cursor = _call(
        conn,
        "brain.search",
        {"query": "shared deploy runbook", "context": {"client": "cursor", "project": "demo"}},
        request_id=2,
    )
    assert belief_id in [item["id"] for item in cursor["items"]]


def test_promotion_never_widens_egress(tmp_path):
    conn = _core(tmp_path, auto=True)
    written = _call(conn, "brain.ingest", {"body": "local only secret-ish note", "context": CC})
    belief_id = written["auto_compiled_belief_id"]
    row = conn.execute(
        "SELECT egress_policy FROM current_beliefs WHERE belief_id=?", (belief_id,)
    ).fetchone()
    assert row["egress_policy"] == "local_only"
    # Hosted delivery must never surface an auto-compiled local_only belief.
    hosted = _call(
        conn,
        "brain.search",
        {"query": "local only secret-ish note", "context": CC},
        request_id=2,
        delivery="hosted_model",
    )
    assert hosted["coverage"]["returned"] == 0


def test_enabled_closeout_summary_is_recallable(tmp_path):
    conn = _core(tmp_path, auto=True)
    receipt = _call(
        conn,
        "brain.closeout",
        {
            "status": "completed",
            "summary": "resolved the flaky deploy by ordering migrations first",
            "context": {**CC, "task": "task:deploy"},
        },
    )
    assert receipt["auto_compiled_belief_id"]
    # A different client on the same project recalls the closeout's substance.
    recall = _call(
        conn,
        "brain.search",
        {
            "query": "flaky deploy ordering migrations",
            "context": {"client": "codex", "project": "demo"},
        },
        request_id=2,
    )
    assert receipt["auto_compiled_belief_id"] in [item["id"] for item in recall["items"]]


def test_auto_compile_scope_prefers_shared_project():
    # project wins over a narrower task, and egress stays local_only.
    scope = auto_compile_scope(ScopeContext(project="demo", client="claude-code", task="task:x"))
    assert scope.scope_type == "project"
    assert scope.scope_id == "project:demo"
    assert scope.egress_policy == "local_only"
    # client-only context falls back to a client scope, still local_only.
    client_scope = auto_compile_scope(ScopeContext(client="claude-code"))
    assert client_scope.scope_type == "client"
    assert client_scope.egress_policy == "local_only"
