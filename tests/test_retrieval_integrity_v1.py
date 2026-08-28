"""Two served-read-path defects, pinned by number.

1. An id-shaped query that names nothing must return nothing. ``brain.search``
   short-circuited on a locator; ``brain.context`` did not, so a well-formed
   nonexistent locator reached dense ranking, which always has a nearest
   neighbour to offer. On the reference corpus the nonexistent
   ``belief_ffffffffffffffff`` came back as two unrelated beliefs at cosine
   0.5603 and 0.6134.

2. The served packet must not carry the ``confidence`` / ``confidence_band``
   pair, and the confidence term of ``ranking_prior`` must be switchable.

Every dense fixture here uses an embedder that maps *every* text to the same
unit vector, so cosine similarity is 1.0 between any query and any belief. That
is deliberate: it is the strongest possible version of the fabrication the fix
has to refuse, and it means each "returns empty" assertion is checked against a
corpus that would otherwise have served something.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocbrain.core_v1 import (
    append_core_event,
    init_core_v1,
    looks_like_exact_locator,
    record_core_v1_evidence,
    search_core_v1,
)
from ocbrain.db import connect
from ocbrain.hybrid import build_vector_index
from ocbrain.mcp import handle_request
from ocbrain.mcp_v1 import build_context_v1
from ocbrain.scope import ScopeContext, ScopeTag

# Well-formed and, by construction, absent: 16 f's is the shape a stable object
# id has and no digest prefix the ledger mints can be.
ABSENT_BELIEF_LOCATOR = "belief_ffffffffffffffff"
ABSENT_EVIDENCE_LOCATOR = "evd_ffffffffffffffff"
ABSENT_SHA256 = "f" * 64
PRESENT_BELIEF_LOCATOR = "belief_0123456789abcdef"
PRESENT_CONFIDENTIAL_LOCATOR = "belief_fedcba9876543210"


def _seed(
    conn,
    *,
    belief_id: str,
    body: str,
    project: str = "orchard",
    visibility: str = "internal",
    confidence: float = 0.90,
    evidence_bodies: tuple[str, ...] = (),
) -> list[str]:
    scope = ScopeTag(
        "project",
        f"project:{project}",
        visibility=visibility,
        egress_policy="hosted_ok",
        provenance="test",
    )
    evidence_ids: list[str] = []
    for evidence_body in evidence_bodies:
        evidence_id, _event = record_core_v1_evidence(
            conn,
            body=evidence_body,
            kind="note",
            scope=scope,
            writer="test",
        )
        evidence_ids.append(evidence_id)
    proposal = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": belief_id,
            "belief_type": "curated_fact",
            "body": body,
            "evidence_ids": evidence_ids,
            "scope": scope.to_dict(),
            "confidence": confidence,
            "attributes": {"source_quality": 0.95},
        },
        writer="test",
    )
    from ocbrain.mcp_v1 import decide_proposal_v1

    decide_proposal_v1(
        conn,
        proposal_event_id=proposal,
        decision="approve",
        actor="test",
        edited_body=None,
        reason="test seed",
    )
    return evidence_ids


def _arm_dense_arm(conn, path: Path, monkeypatch) -> None:
    """Give every belief cosine 1.0 with every query."""
    conn.commit()
    monkeypatch.setenv("OCBRAIN_EMBED_MODEL", "test-local")
    monkeypatch.setenv("OCBRAIN_EMBED_DIMENSIONS", "2")
    monkeypatch.setattr(
        "ocbrain.hybrid.embed_texts",
        lambda texts, **_kwargs: [[1.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(
        "ocbrain.hybrid._ollama_model_metadata",
        lambda *_args, **_kwargs: {"digest": "sha256:test-model-v1"},
    )
    build_vector_index(path, model="test-local")


@pytest.fixture
def armed_core(tmp_path: Path, monkeypatch):
    path = tmp_path / "core.sqlite"
    conn = connect(path)
    init_core_v1(conn)
    _seed(
        conn,
        belief_id=PRESENT_BELIEF_LOCATOR,
        body="Meyer lemons ripen in February on the south terrace.",
        evidence_bodies=("Picked 4kg on 2026-02-11.", "Second pick 2026-02-19."),
    )
    _seed(
        conn,
        belief_id="belief_aaaaaaaaaaaaaaaa",
        body="The pump on the upper line runs at 2.1 bar.",
        evidence_bodies=("Gauge read 2.1 bar.",),
    )
    _seed(
        conn,
        belief_id=PRESENT_CONFIDENTIAL_LOCATOR,
        body="Confidential: the orchard lease renews at a 14% increase.",
        project="ledger",
        visibility="confidential",
        evidence_bodies=("Lease draft clause 7.",),
    )
    _arm_dense_arm(conn, path, monkeypatch)
    return conn


def _local(conn, query: str, *, project: str = "orchard", limit: int = 5):
    return search_core_v1(
        conn, query, context=ScopeContext(project=project), limit=limit
    )


def test_the_fixture_actually_fabricates_on_a_topical_query(armed_core) -> None:
    """The instrument first: prove this corpus DOES serve on a weak query.

    Without this, every "returns empty" assertion below could be passing because
    the fixture never had anything to offer.
    """
    served = _local(armed_core, "wholly unrelated telemetry backfill wording")
    assert served["ranking"]["dense_fallback"] is None
    assert served["ranking"]["dense_candidates"] == 3
    assert {item["belief_id"] for item in served["items"]} == {
        "belief_aaaaaaaaaaaaaaaa",
        PRESENT_BELIEF_LOCATOR,
    }


@pytest.mark.parametrize(
    "locator",
    [
        ABSENT_BELIEF_LOCATOR,
        ABSENT_EVIDENCE_LOCATOR,
        ABSENT_SHA256,
        f"ocbrain-bundle:sha256:{ABSENT_SHA256}",
        "closeout:close_ffffffffffffffff",
    ],
)
def test_an_id_shaped_miss_returns_zero_items(armed_core, locator: str) -> None:
    assert looks_like_exact_locator(locator)
    result = _local(armed_core, locator)
    assert result["ranking"]["mode"] == "exact_locator"
    assert result["ranking"]["exact_locator"] is True
    assert result["ranking"]["exact_locator_matches"] == 0
    assert result["items"] == []
    assert result["scope_mix"] == {}
    # Eligibility is still reported honestly: three beliefs were servable, none
    # of them was the record asked for.
    assert result["ranking"]["eligible_count"] == 3


def test_an_id_shaped_hit_resolves_to_exactly_one_belief(armed_core) -> None:
    result = _local(armed_core, PRESENT_BELIEF_LOCATOR)
    assert result["ranking"]["mode"] == "exact_locator"
    assert result["ranking"]["exact_locator_matches"] == 1
    assert [item["belief_id"] for item in result["items"]] == [PRESENT_BELIEF_LOCATOR]
    item = result["items"][0]
    assert item["source"] == "core_v1_exact_locator"
    assert item["ranking"]["exact_boost"] == 1.0
    assert item["evidence_count"] == 2
    # The neighbour is at cosine 1.0 and is still not served: a lookup answers
    # with the record named, not with the record plus its neighbourhood.
    assert "belief_aaaaaaaaaaaaaaaa" not in json.dumps(result)


def test_holding_a_locator_does_not_unlock_confidential_material(armed_core) -> None:
    """The visibility gate is the ranker's, applied to the lookup path too."""
    denied = _local(armed_core, PRESENT_CONFIDENTIAL_LOCATOR, project="orchard")
    assert denied["items"] == []
    assert denied["ranking"]["exact_locator_matches"] == 0
    assert "14%" not in json.dumps(denied)

    allowed = _local(armed_core, PRESENT_CONFIDENTIAL_LOCATOR, project="ledger")
    assert [item["belief_id"] for item in allowed["items"]] == [
        PRESENT_CONFIDENTIAL_LOCATOR
    ]


def test_brain_context_refuses_an_id_shaped_miss_end_to_end(armed_core) -> None:
    response = handle_request(
        armed_core,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "brain.context",
                "arguments": {
                    "query": ABSENT_BELIEF_LOCATOR,
                    "context": {"project": "orchard"},
                    "limit": 5,
                },
            },
        },
    )
    assert "error" not in response, response
    packet = json.loads(response["result"]["content"][0]["text"])
    assert packet["items"] == []
    assert packet["coverage"]["returned"] == 0
    assert packet["coverage"]["feedback_needed"] is False
    assert packet["coverage"]["source_handle_count"] == 0
    assert packet["coverage"]["ranking"]["exact_locator_matches"] == 0


def test_served_items_carry_evidence_support_and_no_confidence(armed_core) -> None:
    packet, _handles = build_context_v1(
        armed_core,
        "Meyer lemons ripen February south terrace",
        context=ScopeContext(project="orchard"),
        limit=5,
    )
    item = next(item for item in packet["items"] if item["id"] == PRESENT_BELIEF_LOCATOR)
    assert "confidence" not in item
    assert "confidence_band" not in item
    assert item["evidence_count"] == 2
    assert item["evidence_latest_at"] is not None
    # Recency is the newest of THIS belief's evidence -- not the oldest, and not
    # the newest row in the table.
    own = [
        str(row[0])
        for row in armed_core.execute(
            "SELECT recorded_at FROM evidence_objects WHERE evidence_id IN (?, ?)",
            tuple(item["evidence_ids"]),
        )
    ]
    assert len(own) == 2
    assert item["evidence_latest_at"] == max(own)
    assert item["evidence_latest_at"] != min(own)
    everything = [
        str(row[0])
        for row in armed_core.execute("SELECT recorded_at FROM evidence_objects")
    ]
    assert len(everything) == 4
    assert item["evidence_latest_at"] != max(everything)


def test_a_belief_with_no_evidence_reports_zero_not_a_confidence(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "core.sqlite"
    conn = connect(path)
    init_core_v1(conn)
    _seed(conn, belief_id="belief_bbbbbbbbbbbbbbbb", body="Unsupported claim about pears.")
    conn.commit()
    packet, _handles = build_context_v1(
        conn,
        "unsupported claim pears",
        context=ScopeContext(project="orchard"),
        limit=5,
    )
    item = packet["items"][0]
    assert item["evidence_count"] == 0
    assert item["evidence_latest_at"] is None


def test_the_confidence_prior_term_is_switchable_and_reported(
    armed_core, monkeypatch
) -> None:
    """Default ON reproduces today's packets; OFF drops exactly the one factor.

    ``ranking_prior`` is ``scope_weight * (0.85 + 0.15*confidence) *
    (0.85 + 0.15*quality) * (0.99 + 0.01*recency)``. The seeded confidence is
    0.90, so the term contributes 0.985 and switching it off must divide
    ``ranking_prior`` by exactly that and change nothing else.
    """
    query = "Meyer lemons ripen February south terrace"
    on = _local(armed_core, query)
    assert on["ranking"]["confidence_prior_enabled"] is True

    monkeypatch.setenv("OCBRAIN_RETRIEVAL_CONFIDENCE_PRIOR_ENABLED", "0")
    off = _local(armed_core, query)
    assert off["ranking"]["confidence_prior_enabled"] is False

    on_item = next(i for i in on["items"] if i["belief_id"] == PRESENT_BELIEF_LOCATOR)
    off_item = next(i for i in off["items"] if i["belief_id"] == PRESENT_BELIEF_LOCATOR)
    assert on_item["ranking"]["ranking_prior"] == pytest.approx(
        off_item["ranking"]["ranking_prior"] * 0.985, rel=1e-6
    )
    # Nothing but that factor moves.
    assert on_item["ranking"]["lexical_component"] == off_item["ranking"][
        "lexical_component"
    ]
    assert on_item["ranking"]["dense_component"] == off_item["ranking"][
        "dense_component"
    ]
    assert on_item["relevance"] == off_item["relevance"]
