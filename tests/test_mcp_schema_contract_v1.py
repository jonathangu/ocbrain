"""Contract tests reconciling the published v1 MCP schemas with the dispatcher.

Every published property must be callable, and every field the dispatcher
actually requires must be the one the schema advertises as non-nullable. These
tests lock the three schema/validator mismatches that made tools uncallable:

1. ``at_ts`` was published (and, via ``provider_safe_schema``, marked
   required-but-nullable) on a v1 core that rejects any value for it.
2. ``brain.closeout.task_ref`` is conditionally required and must be honored
   when supplied through ``context.task``.
3. A double-encoded ``context`` (a JSON string instead of an object) failed at
   the parse seam even though its fields were correct.
"""

from __future__ import annotations

import json

from test_mcp_v1 import _payload, _seed_v1, _tool_call

from ocbrain.db import connect, init_db
from ocbrain.mcp import checked_filters, handle_request, scope_from_arguments

# The semantic-required set per tool: the fields the dispatcher enforces. Any
# field NOT listed here must be published as nullable (optional) by the schema.
SEMANTIC_REQUIRED = {
    "brain.context": {"query"},
    "brain.source": {"id"},
    "brain.search": {"query"},
    "brain.digest": set(),
    "brain.get": {"id"},
    "brain.feedback": {"retrieval_use_id", "outcome"},
    "brain.ingest": {"body"},
    "brain.closeout": {"status", "summary"},
    "brain.preview": {"query"},
    "brain.egress_preview": set(),
    "brain.correct": {"target", "layer", "op"},
    "brain.proposal_decide": {"proposal_event_id", "decision"},
    "brain.proposals": set(),
    "brain.forget": {"target"},
}


def _tools_by_name(conn, *, allow_writes=False):
    response = handle_request(
        conn,
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        allow_writes=allow_writes,
    )
    return {tool["name"]: tool for tool in response["result"]["tools"]}


def _non_nullable_properties(schema):
    """Top-level properties the schema advertises as required (not …|null).

    ``provider_safe_schema`` wraps every optional field as
    ``{"anyOf": [<schema>, {"type": "null"}]}`` and leaves required fields
    unwrapped, so the unwrapped set is the real required signal on the wire.
    """
    required = set()
    for name, value in schema["properties"].items():
        branches = value.get("anyOf") if isinstance(value, dict) else None
        nullable = isinstance(branches, list) and {"type": "null"} in branches
        if not nullable:
            required.add(name)
    return required


def test_v1_schema_required_matches_dispatcher_for_every_tool(tmp_path):
    conn = _seed_v1(tmp_path)
    tools = _tools_by_name(conn, allow_writes=True)

    for name, expected in SEMANTIC_REQUIRED.items():
        assert name in tools, f"{name} missing from published surface"
        schema = tools[name]["inputSchema"]
        # The provider-safe invariant: closed shape, every property listed in
        # ``required``, real optionality carried only by the …|null wrapper.
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        assert _non_nullable_properties(schema) == expected, name


def test_closeout_schema_documents_conditional_requirements(tmp_path):
    tools = _tools_by_name(_seed_v1(tmp_path))
    properties = tools["brain.closeout"]["inputSchema"]["properties"]
    assert "Required" in properties["summary"]["description"]
    task_ref_schema = properties["task_ref"]["anyOf"][0]
    assert "Required unless context.task" in task_ref_schema["description"]


def test_v1_core_does_not_publish_at_ts(tmp_path):
    tools = _tools_by_name(_seed_v1(tmp_path), allow_writes=True)
    for name in ("brain.context", "brain.search", "brain.preview"):
        assert "at_ts" not in tools[name]["inputSchema"]["properties"], name


def test_legacy_core_still_publishes_at_ts(tmp_path):
    # The legacy v0.x core supports as-of queries, so it must keep advertising
    # the parameter. Only the v1 surface drops it.
    conn = connect(tmp_path / "legacy.sqlite")
    init_db(conn)
    tools = _tools_by_name(conn)
    assert "at_ts" in tools["brain.context"]["inputSchema"]["properties"]


def test_v1_context_accepts_omitted_null_and_blank_at_ts(tmp_path):
    conn = _seed_v1(tmp_path)
    for request_id, arguments in enumerate(
        [
            {"query": "Shared Context", "context": {"project": "ocbrain"}},
            {"query": "Shared Context", "context": {"project": "ocbrain"}, "at_ts": None},
            {"query": "Shared Context", "context": {"project": "ocbrain"}, "at_ts": ""},
            {"query": "Shared Context", "context": {"project": "ocbrain"}, "at_ts": "   "},
        ],
        start=1,
    ):
        response = handle_request(
            conn, _tool_call("brain.context", arguments, request_id=request_id)
        )
        assert "error" not in response, (arguments, response)
        assert _payload(response)["schema_version"] == "ocbrain.context.v1"


def test_v1_context_rejects_every_meaningful_at_ts(tmp_path):
    conn = _seed_v1(tmp_path)
    for name, at_ts in (
        ("brain.context", "2026-07-01T00:00:00Z"),
        ("brain.search", "2026-07-01T00:00:00Z"),
        ("brain.preview", 123),
    ):
        response = handle_request(
            conn,
            _tool_call(name, {"query": "Shared Context", "at_ts": at_ts}),
            profile="admin" if name == "brain.preview" else None,
        )
        message = response["error"]["message"]
        assert "at_ts" in message and "not supported" in message, name


def test_v1_context_accepts_double_encoded_context_string(tmp_path):
    conn = _seed_v1(tmp_path)
    payload = _payload(
        handle_request(
            conn,
            _tool_call(
                "brain.context",
                {"query": "Shared Context", "context": json.dumps({"project": "ocbrain"})},
            ),
        )
    )
    assert payload["resolved_context"]["project"] == "ocbrain"


def test_v1_get_accepts_double_encoded_context_string(tmp_path):
    conn = _seed_v1(tmp_path)
    payload = _payload(
        handle_request(
            conn,
            _tool_call(
                "brain.get",
                {
                    "id": "belief:shared-context",
                    "context": json.dumps({"project": "ocbrain", "runtime": "codex"}),
                },
            ),
        )
    )
    assert payload["canonical_id"] == "belief:shared-context"


def test_v1_context_rejects_non_object_context_string(tmp_path):
    conn = _seed_v1(tmp_path)
    for bad in ('"ocbrain"', "not json at all", "[1, 2, 3]"):
        response = handle_request(
            conn,
            _tool_call("brain.context", {"query": "Shared Context", "context": bad}),
        )
        assert response["error"]["message"] == "context must be an object", bad


def test_double_encoded_scope_and_filters_use_shared_object_seam():
    filters = checked_filters(json.dumps({"project": "ocbrain", "unknown": "ignored"}))
    assert filters == {"project": "ocbrain"}

    scope = scope_from_arguments(
        {
            "scope": json.dumps(
                {
                    "scope_type": "project",
                    "scope_id": "project:ocbrain",
                    "visibility": "internal",
                    "egress_policy": "hosted_ok",
                }
            )
        }
    )
    assert scope is not None
    assert scope.scope_id == "project:ocbrain"


def test_v1_context_reports_feedback_needed(tmp_path):
    conn = _seed_v1(tmp_path)
    served = _payload(
        handle_request(
            conn,
            _tool_call(
                "brain.context",
                {"query": "Shared Context", "context": {"project": "ocbrain"}},
            ),
        )
    )
    assert served["coverage"]["returned"] > 0
    assert served["coverage"]["feedback_needed"] is True

    empty = _payload(
        handle_request(
            conn,
            _tool_call(
                "brain.context",
                {"query": "Shared Context", "context": {"project": "no-such-project"}},
                request_id=2,
            ),
        )
    )
    assert empty["coverage"]["returned"] == 0
    assert empty["coverage"]["feedback_needed"] is False


def test_v1_closeout_requires_summary(tmp_path):
    conn = _seed_v1(tmp_path)
    response = handle_request(
        conn,
        _tool_call(
            "brain.closeout",
            {"task_ref": "task:demo", "status": "completed"},
        ),
    )
    assert "summary" in response["error"]["message"]


def test_v1_closeout_requires_task_ref_without_context_task(tmp_path):
    conn = _seed_v1(tmp_path)
    response = handle_request(
        conn,
        _tool_call(
            "brain.closeout",
            {"status": "completed", "summary": "Did the thing."},
        ),
    )
    assert "task_ref is required" in response["error"]["message"]


def test_v1_closeout_accepts_task_ref_from_context(tmp_path):
    conn = _seed_v1(tmp_path)
    payload = _payload(
        handle_request(
            conn,
            _tool_call(
                "brain.closeout",
                {
                    "status": "completed",
                    "summary": "Did the thing.",
                    "context": {"project": "ocbrain", "task": "task:from-context"},
                },
            ),
        )
    )
    assert payload["task_ref"] == "task:from-context"
