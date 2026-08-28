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
    "brain.supersede": {"target", "body", "reason"},
    # The loop-facing surface. `brain.briefing` has no required field at all and
    # deliberately no `query`: that absence is what makes it a contract rather
    # than a search, and this table is where a future edit adding one would be
    # caught.
    "brain.briefing": set(),
    "brain.ledger": set(),
    "brain.goal_open": {"objective", "finish_line", "source_path"},
    "brain.goal_close": {"goal_id", "status", "verifier_uri", "verifier_status"},
    "brain.preview": {"query"},
    "brain.egress_preview": set(),
    "brain.correct": {"target", "layer", "op"},
    "brain.proposal_decide": {"proposal_event_id", "decision"},
    "brain.proposals": set(),
    "brain.forget": {"target"},
}


def _tools_by_name(conn, *, allow_writes=False, client_name=None):
    session_state = {}
    if client_name is not None:
        handle_request(
            conn,
            {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": client_name, "version": "0"},
                },
            },
            allow_writes=allow_writes,
            session_state=session_state,
        )
    response = handle_request(
        conn,
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        allow_writes=allow_writes,
        session_state=session_state,
    )
    return {tool["name"]: tool for tool in response["result"]["tools"]}


def _is_nullable(value):
    if not isinstance(value, dict):
        return False
    type_value = value.get("type")
    if isinstance(type_value, list) and "null" in type_value:
        return True
    branches = value.get("anyOf")
    return isinstance(branches, list) and {"type": "null"} in branches


def _non_nullable_properties(schema):
    """Top-level properties the schema advertises as required (not …|null).

    ``provider_safe_schema`` marks every optional field nullable — a
    ``["<type>", "null"]`` union, or an ``anyOf`` wrapper when the field has no
    simple type — and leaves required fields alone, so the non-nullable set is
    the real required signal on the wire.
    """
    return {name for name, value in schema["properties"].items() if not _is_nullable(value)}


def test_v1_schema_required_matches_dispatcher_for_every_tool(tmp_path):
    conn = _seed_v1(tmp_path)
    tools = _tools_by_name(conn, allow_writes=True, client_name="codex")

    for name, expected in SEMANTIC_REQUIRED.items():
        assert name in tools, f"{name} missing from published surface"
        schema = tools[name]["inputSchema"]
        # The provider-safe invariant: closed shape, every property listed in
        # ``required``, real optionality carried only by the …|null wrapper.
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        assert _non_nullable_properties(schema) == expected, name


def test_v1_plain_dialect_requires_only_semantic_fields(tmp_path):
    """Default/plain dialect: ``required`` is exactly what the dispatcher enforces.

    Serving the strict all-keys-required dialect to non-strict harnesses makes
    them reject ordinary partial calls client-side before the server is reached
    (observed in Claude Code: a context of only project/repo/runtime failed
    because context.client/task/session were marked required). Unknown clients
    therefore get the plain dialect; only strict harnesses opt in by name.
    """
    conn = _seed_v1(tmp_path)
    for client_name in (None, "claude-code", "cursor", "hermes-agent"):
        tools = _tools_by_name(conn, allow_writes=True, client_name=client_name)
        for name, expected in SEMANTIC_REQUIRED.items():
            schema = tools[name]["inputSchema"]
            assert set(schema.get("required") or []) == expected, (client_name, name)
        context_schema = tools["brain.context"]["inputSchema"]["properties"]["context"]
        assert not context_schema.get("required"), client_name
        assert context_schema["type"] == "object", client_name


def test_v1_strict_dialect_is_client_gated(tmp_path):
    conn = _seed_v1(tmp_path)
    for client_name in ("codex", "Codex CLI", "openai-agents", "gpt-shell"):
        tools = _tools_by_name(conn, client_name=client_name)
        schema = tools["brain.context"]["inputSchema"]
        assert set(schema["required"]) == set(schema["properties"]), client_name
        assert schema["additionalProperties"] is False, client_name


def _denulled(value):
    """The informative half of a nullable property schema, whichever spelling."""
    if isinstance(value, dict) and isinstance(value.get("anyOf"), list):
        return value["anyOf"][0]
    return value


def test_closeout_schema_documents_conditional_requirements(tmp_path):
    tools = _tools_by_name(_seed_v1(tmp_path))
    properties = tools["brain.closeout"]["inputSchema"]["properties"]
    assert "Required" in properties["summary"]["description"]
    task_ref_schema = _denulled(properties["task_ref"])
    assert "Required unless context.task" in task_ref_schema["description"]

    for collection in ("actions", "outcomes"):
        collection_schema = _denulled(properties[collection])
        collection_description = collection_schema["description"]
        assert "Empty features objects are treated as omitted" in collection_description
        item_properties = collection_schema["items"]["properties"]
        features_schema = _denulled(item_properties["features"])
        feature_schema = _denulled(item_properties["feature_schema"])
        features_description = features_schema["description"]
        feature_schema_description = feature_schema["description"]
        assert "empty object is normalized as absent" in features_description
        assert "non-empty object requires feature_schema" in features_description
        assert "Required and nonblank when features has entries" in (feature_schema_description)
        assert "rejected when features is omitted or empty" in feature_schema_description


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


def test_v1_schema_optional_fields_keep_type_visible(tmp_path):
    """Optional fields must stay typed on the wire (strict dialect).

    A ``{"anyOf": [<schema>, {"type": "null"}]}`` wrapper is flattened to an
    untyped parameter by some client harnesses (observed in Claude Code), whose
    models then send JSON-encoded arrays and ``"false"`` strings. Nullability
    must ride the ``type`` union so every property keeps a visible type.
    """
    conn = _seed_v1(tmp_path)
    for name, tool in _tools_by_name(conn, allow_writes=True, client_name="codex").items():
        for prop, value in tool["inputSchema"]["properties"].items():
            assert "type" in value, (name, prop, value)
            if _is_nullable(value):
                enum_values = value.get("enum")
                if isinstance(enum_values, list):
                    assert None in enum_values, (name, prop)


def test_v1_closeout_accepts_double_encoded_arrays(tmp_path):
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
    response = handle_request(
        conn,
        _tool_call(
            "brain.closeout",
            {
                "task_ref": "task:double-encoded",
                "status": "completed",
                "summary": "Closed out with stringified array arguments.",
                "retrieval_use_ids": json.dumps([served["retrieval_use_id"]]),
                "artifact_refs": json.dumps([{"uri": "file:///tmp/x", "kind": "file"}]),
                "verifier_refs": json.dumps([]),
                "decision_impact": "",
                "decision_note": "",
            },
            request_id=2,
        ),
    )
    assert "error" not in response, response
    assert _payload(response)["task_ref"] == "task:double-encoded"


def test_v1_closeout_accepts_bare_string_retrieval_use_id(tmp_path):
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
    response = handle_request(
        conn,
        _tool_call(
            "brain.closeout",
            {
                "task_ref": "task:bare-string",
                "status": "completed",
                "summary": "Closed out with a single id where an array belongs.",
                "retrieval_use_ids": served["retrieval_use_id"],
            },
            request_id=2,
        ),
    )
    assert "error" not in response, response


def test_v1_feedback_treats_blank_note_as_absent(tmp_path):
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
    response = handle_request(
        conn,
        _tool_call(
            "brain.feedback",
            {
                "retrieval_use_id": served["retrieval_use_id"],
                "outcome": "used",
                "note": "   ",
            },
            request_id=2,
        ),
    )
    assert "error" not in response, response


def test_v1_context_accepts_the_deprecated_cross_scope_argument(tmp_path):
    """Five live clients still send ``cross_scope``. It must never fault.

    It no longer changes anything: local retrieval ranks every scope, so there
    is no narrower mode to widen. The packet reports the mode it actually ran
    rather than echoing the argument back.
    """
    conn = _seed_v1(tmp_path)
    baseline: list[str] | None = None
    for request_id, sent in enumerate(["false", "true", "", True, False, None], start=1):
        served = _payload(
            handle_request(
                conn,
                _tool_call(
                    "brain.context",
                    {
                        "query": "Shared Context",
                        "context": {"project": "ocbrain"},
                        "cross_scope": sent,
                    },
                    request_id=request_id,
                ),
            )
        )
        assert "cross_scope" not in served, sent
        assert served["retrieval_mode"] == "ranked", sent
        item_ids = [item["id"] for item in served["items"]]
        if baseline is None:
            baseline = item_ids
        assert item_ids == baseline, (sent, item_ids, baseline)


def test_v1_context_accepts_stringly_limit(tmp_path):
    conn = _seed_v1(tmp_path)
    response = handle_request(
        conn,
        _tool_call(
            "brain.context",
            {"query": "Shared Context", "context": {"project": "ocbrain"}, "limit": "3"},
        ),
    )
    assert "error" not in response, response


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

    # A query nothing answers, rather than a query the wrong scope cannot see:
    # an unreachable scope now retries across scopes, so it no longer produces a
    # reliably empty packet. The subject here is the flag, not scope isolation.
    empty = _payload(
        handle_request(
            conn,
            _tool_call(
                "brain.context",
                {"query": "quartz zeppelin nonsense", "context": {"project": "ocbrain"}},
                request_id=2,
            ),
        )
    )
    assert empty["coverage"]["returned"] == 0
    assert empty["coverage"]["feedback_needed"] is False


def test_v1_feedback_rejects_empty_retrieval_without_mutating_telemetry(tmp_path):
    conn = _seed_v1(tmp_path)
    empty = _payload(
        handle_request(
            conn,
            _tool_call(
                "brain.context",
                {"query": "quartz zeppelin nonsense", "context": {"project": "ocbrain"}},
            ),
        )
    )
    retrieval_use_id = empty["retrieval_use_id"]
    assert empty["coverage"]["returned"] == 0
    before = tuple(
        conn.execute(
            "SELECT outcome, note, feedback_source, feedback_at "
            "FROM retrieval_uses WHERE id=?",
            (retrieval_use_id,),
        ).fetchone()
    )

    refused = handle_request(
        conn,
        _tool_call(
            "brain.feedback",
            {"retrieval_use_id": retrieval_use_id, "outcome": "irrelevant"},
            request_id=2,
        ),
    )

    assert refused["error"]["code"] == -32602
    assert refused["error"]["message"].startswith(
        "empty_retrieval_not_feedback_eligible:"
    )
    after = tuple(
        conn.execute(
            "SELECT outcome, note, feedback_source, feedback_at "
            "FROM retrieval_uses WHERE id=?",
            (retrieval_use_id,),
        ).fetchone()
    )
    assert after == before
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM retrieval_items WHERE retrieval_use_id=?",
            (retrieval_use_id,),
        ).fetchone()[0]
        == 0
    )


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
