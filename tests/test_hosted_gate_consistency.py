"""Hosted vs local capability reporting, and closeout receipt lookup.

The harness surfaces (briefing, ledger, goals) hold local task state that never
passes the per-belief egress gate, so they are removed from the hosted catalogue
and refused on dispatch. What made that a defect rather than a boundary was the
prose: `initialize` ordered every client to call brain.briefing first, so a
hosted model was instructed into refusals it then retried. These tests pin the
whole triangle -- catalogue, dispatch, instructions -- in both delivery targets,
and pin brain.get's closeout contract on each side of the same boundary.
"""

from __future__ import annotations

import json
import re

import pytest

from ocbrain.closeout import record_closeout
from ocbrain.core_v1 import init_core_v1
from ocbrain.db import connect
from ocbrain.mcp import (
    HARNESS_TOOLS,
    HOSTED_LOCAL_ONLY_TOOLS,
    call_tool,
    handle_request,
    instructions_text,
    tool_list,
)
from ocbrain.scope import HOSTED_MODEL_TARGET, LOCAL_MODEL_TARGET, ScopeContext

_TOOL_NAME = re.compile(r"brain\.[a-z_]+")


def _core(tmp_path):
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    return conn


def _catalogue(*, delivery_target: str) -> set[str]:
    return {
        str(tool["name"])
        for tool in tool_list(profile="runtime", core_v1=True, delivery_target=delivery_target)
    }


def test_the_hosted_catalogue_omits_exactly_the_local_only_tools():
    local = _catalogue(delivery_target=LOCAL_MODEL_TARGET)
    hosted = _catalogue(delivery_target=HOSTED_MODEL_TARGET)
    # brain.proposals is admin-only and in neither runtime catalogue; it is in
    # HOSTED_LOCAL_ONLY_TOOLS for the dispatcher check, not for this one.
    assert HARNESS_TOOLS <= local
    assert HOSTED_LOCAL_ONLY_TOOLS & hosted == set()
    assert local - hosted == HARNESS_TOOLS


def test_local_instructions_order_the_harness_surfaces_and_hosted_ones_do_not():
    local = instructions_text(core_v1=True, delivery_target=LOCAL_MODEL_TARGET)
    hosted = instructions_text(core_v1=True, delivery_target=HOSTED_MODEL_TARGET)
    assert "call brain.briefing" in local
    assert "check brain.ledger" in local
    assert "call brain.briefing" not in hosted
    assert "check brain.ledger" not in hosted
    # Named once, as unavailable -- the sentence that stops the retry loop.
    assert "are not available on this transport" in hosted
    assert "do not call them and do not retry them" in hosted
    # Same contract, one delivery: the zero-item rule and the tail are shared.
    assert "feedback_needed" in hosted and "feedback_needed" in local
    assert "never start hosted judgment" in hosted


def test_every_tool_the_instructions_mention_is_in_the_catalogue_for_that_target():
    """Catalogue/instruction consistency, both directions and both targets.

    A served instruction naming a tool the same transport refuses is the
    defect: the model is told to do something that cannot succeed. The
    local-only names are the documented exception -- the hosted text must name
    them, and only to say they are unavailable.
    """
    for delivery_target in (LOCAL_MODEL_TARGET, HOSTED_MODEL_TARGET):
        mentioned = set(_TOOL_NAME.findall(instructions_text(delivery_target=delivery_target)))
        unavailable = HOSTED_LOCAL_ONLY_TOOLS if delivery_target == HOSTED_MODEL_TARGET else set()
        catalogue = _catalogue(delivery_target=delivery_target)
        assert mentioned - unavailable <= catalogue, delivery_target


def test_initialize_serves_the_delivery_matching_instructions(tmp_path):
    conn = _core(tmp_path)
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"clientInfo": {"name": "hosted-client", "version": "1"}},
    }
    local = handle_request(conn, request, delivery_target=LOCAL_MODEL_TARGET)
    hosted = handle_request(conn, request, delivery_target=HOSTED_MODEL_TARGET)
    assert local["result"]["serverInfo"]["deliveryTarget"] == LOCAL_MODEL_TARGET
    assert hosted["result"]["serverInfo"]["deliveryTarget"] == HOSTED_MODEL_TARGET
    assert "call brain.briefing" in local["result"]["instructions"]
    assert "call brain.briefing" not in hosted["result"]["instructions"]


def test_hosted_dispatch_refuses_the_harness_surface(tmp_path):
    conn = _core(tmp_path)
    for name in sorted(HOSTED_LOCAL_ONLY_TOOLS):
        with pytest.raises(PermissionError):
            call_tool(
                conn,
                {"name": name, "arguments": {}},
                profile="runtime",
                delivery_target=HOSTED_MODEL_TARGET,
            )


def test_local_get_serves_a_closeout_receipt(tmp_path):
    conn = _core(tmp_path)
    receipt = record_closeout(
        conn,
        task_ref="RECEIPT-LOOKUP-1",
        status="blocked",
        summary="Waiting on the verifier before the deploy.",
        awaiting="verifier",
        unresolved="the verifier has not been run",
        context=ScopeContext(project="coframe", runtime="hermes"),
        verifier_refs=[{"uri": "repo://ocbrain/pytest", "status": "unknown"}],
    )
    conn.commit()
    payload = call_tool(
        conn,
        {"name": "brain.get", "arguments": {"id": receipt["id"]}},
        profile="runtime",
        delivery_target=LOCAL_MODEL_TARGET,
    )
    body = payload["content"][0]
    assert body["type"] == "text"
    served = json.loads(body["text"])
    assert served["object_kind"] == "closeout"
    assert served["task_ref"] == "RECEIPT-LOOKUP-1"
    assert served["status"] == "blocked"
    assert served["id"] == receipt["id"]


def test_hosted_get_of_a_closeout_is_a_typed_local_only_refusal(tmp_path):
    """The refusal names the boundary and carries none of the receipt."""
    conn = _core(tmp_path)
    receipt = record_closeout(
        conn,
        task_ref="RECEIPT-LOOKUP-2",
        status="completed",
        summary="Local task state that must not reach a hosted model.",
        context=ScopeContext(project="coframe"),
    )
    conn.commit()
    with pytest.raises(PermissionError) as excinfo:
        call_tool(
            conn,
            {"name": "brain.get", "arguments": {"id": receipt["id"]}},
            profile="runtime",
            delivery_target=HOSTED_MODEL_TARGET,
        )
    message = str(excinfo.value)
    assert "local-only" in message
    assert "hosted_model" in message
    assert receipt["summary"] not in message
    assert receipt["task_ref"] not in message


def test_get_of_an_unknown_id_still_says_not_found(tmp_path):
    conn = _core(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        call_tool(
            conn,
            {"name": "brain.get", "arguments": {"id": "close_000000000000"}},
            profile="runtime",
            delivery_target=LOCAL_MODEL_TARGET,
        )
    assert "object not found" in str(excinfo.value)
