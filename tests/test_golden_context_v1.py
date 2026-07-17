from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ocbrain.core_v1 import (
    append_core_event,
    init_core_v1,
    record_core_v1_evidence,
    sha256_text,
)
from ocbrain.db import connect
from ocbrain.mcp import handle_request
from ocbrain.mcp_v1 import decide_proposal_v1
from ocbrain.scope import ScopeTag

DATASET_PATH = Path(__file__).parent / "fixtures" / "golden_context_v1.json"
DATASET = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
RECORDS = {record["id"]: record for record in DATASET["records"]}


def _tool_call(name: str, arguments: dict[str, Any], *, request_id: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def _payload(response: dict[str, Any]) -> dict[str, Any]:
    assert "error" not in response, response
    return json.loads(response["result"]["content"][0]["text"])


def _seed_dataset(tmp_path: Path):
    conn = connect(tmp_path / "golden-core.sqlite")
    init_core_v1(conn)
    for record in DATASET["records"]:
        scope = ScopeTag.from_dict(record["scope"])
        evidence = record["evidence"]
        evidence_id, _event_id = record_core_v1_evidence(
            conn,
            body=evidence["body"],
            kind=evidence["kind"],
            scope=scope,
            writer="golden-fixture",
        )
        proposal_id = append_core_event(
            conn,
            "compilation_proposed",
            {
                "belief_id": record["id"],
                "belief_type": "golden_fixture",
                "body": record["body"],
                "evidence_ids": [evidence_id],
                "scope": scope.to_dict(),
                "confidence": record["confidence"],
                "attributes": record.get("attributes", {}),
            },
            writer="golden-fixture",
        )
        decide_proposal_v1(
            conn,
            proposal_event_id=proposal_id,
            decision="approve",
            actor="golden-fixture",
            edited_body=None,
            reason="deterministic public contract fixture",
        )
    conn.commit()
    return conn


def test_golden_dataset_is_public_synthetic_and_self_hashing() -> None:
    assert DATASET["schema_version"] == "ocbrain.golden-context.v1"
    assert DATASET["dataset_class"] == "public_synthetic_test_fixture_not_training_data"
    assert len(RECORDS) == len(DATASET["records"])
    for record in DATASET["records"]:
        assert record["body"] == record["evidence"]["body"]
        assert sha256_text(record["evidence"]["body"]) == record["evidence"]["sha256"]
        assert record["scope"]["provenance"] == "golden_fixture"


@pytest.mark.parametrize("case", DATASET["cases"], ids=lambda case: case["id"])
def test_golden_context_and_source_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: dict[str, Any],
) -> None:
    # Force the documented deterministic lexical fallback for this contract.
    # Dense retrieval has its own sidecar tests and must not affect golden IDs.
    monkeypatch.delenv("OCBRAIN_VECTOR_DB", raising=False)
    monkeypatch.delenv("OCBRAIN_EMBED_MODEL", raising=False)
    monkeypatch.delenv("OCBRAIN_EMBED_DIMENSIONS", raising=False)
    conn = _seed_dataset(tmp_path)
    expected = case["expected"]
    context_response = handle_request(
        conn,
        _tool_call(
            "brain.context",
            {
                "query": case["query"],
                "context": case["context"],
                "cross_scope": case["cross_scope"],
                "limit": case["limit"],
            },
            request_id=f"context:{case['id']}",
        ),
        delivery_target=case["delivery_target"],
    )
    packet = _payload(context_response)
    item_ids = [item["id"] for item in packet["items"]]

    assert packet["schema_version"] == "ocbrain.context.v1"
    assert packet["core_schema"] == "ocbrain.core.v1"
    assert packet["delivery_target"] == case["delivery_target"]
    assert packet["cross_scope"] is case["cross_scope"]
    assert packet["retrieval_use_status"] == "recorded"
    assert packet["coverage"]["returned"] == len(item_ids)
    assert packet["coverage"]["excluded_scope_count"] == expected["excluded_scope_count"]
    assert (
        packet["coverage"]["excluded_delivery_count"]
        == expected["excluded_delivery_count"]
    )
    assert packet["coverage"]["exclusion_count_basis"] == "current_serving_inventory"
    assert packet["coverage"]["ranking"]["eligible_count"] == expected["eligible_count"]
    assert packet["coverage"]["serialized_bytes"] == len(
        json.dumps(packet, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    assert packet["coverage"]["serialized_bytes"] <= packet["coverage"][
        "hard_packet_limit_bytes"
    ]
    assert packet["coverage"]["source_handle_count"] == sum(
        len(item["sources"]) for item in packet["items"]
    )
    assert packet["coverage"]["unavailable_sources"] == []
    if case["delivery_target"] == "hosted_model":
        assert packet["coverage"]["excluded_sample"] == []

    if "ids_ordered" in expected:
        assert item_ids == expected["ids_ordered"]
    if "ids_unordered" in expected:
        assert set(item_ids) == set(expected["ids_unordered"])

    encoded_packet = json.dumps(packet, sort_keys=True)
    for forbidden_id in expected.get("forbidden_ids", []):
        assert forbidden_id not in encoded_packet
    for forbidden_marker in expected.get("forbidden_markers", []):
        assert forbidden_marker not in encoded_packet

    if "contradictions" in expected:
        actual_pairs = {
            (conflict["belief_id"], conflict["other_belief_id"])
            for conflict in packet["contradictions"]
        }
        expected_pairs = {
            (conflict["belief_id"], conflict["other_belief_id"])
            for conflict in expected["contradictions"]
        }
        assert actual_pairs == expected_pairs
        assert all(
            conflict["reason"] == "explicit_compiler_metadata"
            for conflict in packet["contradictions"]
        )
        item_evidence = {
            item["id"]: set(item["evidence_ids"]) for item in packet["items"]
        }
        for conflict in packet["contradictions"]:
            assert set(conflict["evidence_ids"]) == (
                item_evidence[conflict["belief_id"]]
                | item_evidence[conflict["other_belief_id"]]
            )
    else:
        assert packet["contradictions"] == []

    source_item_id = expected.get("source_item_id")
    if source_item_id is None:
        return
    item = next(item for item in packet["items"] if item["id"] == source_item_id)
    assert len(item["sources"]) == 1
    source_handle = item["sources"][0]
    record = RECORDS[source_item_id]
    assert source_handle["kind"] == "core_v1_evidence"
    assert source_handle["content_hash"] == expected["source_sha256"]
    source_project = record["scope"]["scope_id"].removeprefix("project:")
    source_context = {**case["context"], "project": source_project}
    if source_project != case["context"]["project"]:
        denied_original_scope = handle_request(
            conn,
            _tool_call(
                "brain.source",
                {"id": source_handle["id"], "context": case["context"]},
                request_id=f"source-original-scope-denied:{case['id']}",
            ),
            delivery_target=case["delivery_target"],
        )
        assert denied_original_scope["error"]["code"] == -32001
        assert "scope does not match" in denied_original_scope["error"]["message"]
    source = _payload(
        handle_request(
            conn,
            _tool_call(
                "brain.source",
                {"id": source_handle["id"], "context": source_context},
                request_id=f"source:{case['id']}",
            ),
            delivery_target=case["delivery_target"],
        )
    )
    assert source["schema_version"] == "ocbrain.source.v1"
    assert source["object_id"] == source_item_id
    assert source["kind"] == "core_v1_evidence"
    assert source["scope"] == record["scope"]
    assert source["hash_verified"] is True
    assert source["content_hash"] == expected["source_sha256"]
    assert source["content"] == record["evidence"]["body"]
    assert source["truncated"] is False
    assert source["delivery_target"] == case["delivery_target"]
    assert source["origin_retrieval_use_id"] == packet["retrieval_use_id"]
    if case["delivery_target"] == "hosted_model":
        assert source["uri"].startswith("ocbrain://evidence/")

    if expected.get("wrong_scope_denied"):
        denied = handle_request(
            conn,
            _tool_call(
                "brain.source",
                {
                    "id": source_handle["id"],
                    "context": {**case["context"], "project": "beta"},
                },
                request_id=f"source-denied:{case['id']}",
            ),
            delivery_target=case["delivery_target"],
        )
        assert denied["error"]["code"] == -32001
        assert "scope does not match" in denied["error"]["message"]
