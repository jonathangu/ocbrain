"""The two boundaries that did not move when scope stopped filtering.

Local retrieval now ranks across every scope instead of selecting by it. These
tests pin the gates that survived that change, because they are the ones a
widening is most likely to take with it by accident:

1. material without ``hosted_ok`` never reaches a hosted model;
2. confidential material never reaches a local model outside its own scope.

Both are written so that breaking the guard fails the test — see
``docs/`` notes in the pull request for the mutation runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from ocbrain.cli import main
from ocbrain.core_v1 import (
    append_core_event,
    init_core_v1,
    record_core_v1_evidence,
)
from ocbrain.db import connect
from ocbrain.mcp_v1 import build_context_v1, decide_proposal_v1, exact_lookup_v1
from ocbrain.scope import (
    HOSTED_MODEL_TARGET,
    LOCAL_MODEL_TARGET,
    ScopeContext,
    ScopeTag,
)

QUERY = "harbour lantern ledger reconciliation"


def _seed_belief(
    conn,
    *,
    belief_id: str,
    body: str,
    scope: ScopeTag,
) -> None:
    evidence_id, _event_id = record_core_v1_evidence(
        conn,
        body=f"Evidence for {belief_id}: {body}",
        kind="observation",
        scope=scope,
        writer="egress-invariants",
    )
    proposal_id = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": belief_id,
            "body": body,
            "evidence_ids": [evidence_id],
            "scope": scope.to_dict(),
            "confidence": 0.95,
        },
        writer="egress-invariants",
    )
    decide_proposal_v1(
        conn,
        proposal_event_id=proposal_id,
        decision="approve",
        actor="egress-invariants",
        edited_body=None,
        reason="egress invariant fixture",
    )


def _seed(tmp_path: Path):
    conn = connect(tmp_path / "egress.sqlite")
    init_core_v1(conn)
    # Same words in every body, so relevance can never be the reason one of
    # them is missing from a packet.
    _seed_belief(
        conn,
        belief_id="belief:home-internal",
        body=f"HOME_INTERNAL_MARKER {QUERY} for the caller's own project.",
        scope=ScopeTag("project", "project:home", egress_policy="local_only"),
    )
    _seed_belief(
        conn,
        belief_id="belief:neighbour-internal",
        body=f"NEIGHBOUR_INTERNAL_MARKER {QUERY} in a neighbouring project.",
        scope=ScopeTag("project", "project:neighbour", egress_policy="local_only"),
    )
    _seed_belief(
        conn,
        belief_id="belief:neighbour-confidential",
        body=f"NEIGHBOUR_CONFIDENTIAL_MARKER {QUERY} in a neighbouring project.",
        scope=ScopeTag(
            "project",
            "project:neighbour",
            visibility="confidential",
            egress_policy="local_only",
        ),
    )
    _seed_belief(
        conn,
        belief_id="belief:home-hosted-ok",
        body=f"HOME_HOSTED_OK_MARKER {QUERY} cleared for hosted delivery.",
        scope=ScopeTag("project", "project:home", egress_policy="hosted_ok"),
    )
    conn.commit()
    return conn


def test_material_without_hosted_ok_never_reaches_a_hosted_model(tmp_path: Path) -> None:
    conn = _seed(tmp_path)
    context = ScopeContext(project="home")

    packet, _handles = build_context_v1(
        conn,
        QUERY,
        context=context,
        limit=50,
        delivery_target=HOSTED_MODEL_TARGET,
    )
    encoded = json.dumps(packet)

    # The one belief cleared for hosted delivery is served, which is what makes
    # the absences below meaningful rather than a query that matched nothing.
    assert [item["id"] for item in packet["items"]] == ["belief:home-hosted-ok"]
    for marker in (
        "HOME_INTERNAL_MARKER",
        "NEIGHBOUR_INTERNAL_MARKER",
        "NEIGHBOUR_CONFIDENTIAL_MARKER",
    ):
        assert marker not in encoded, marker

    # The same holds for the exact-locator path, which bypasses ranking.
    for belief_id in (
        "belief:home-internal",
        "belief:neighbour-internal",
        "belief:neighbour-confidential",
    ):
        assert (
            exact_lookup_v1(
                conn,
                belief_id,
                context=context,
                cross_scope=True,
                delivery_target=HOSTED_MODEL_TARGET,
            )
            == []
        ), belief_id
    # The hosted-cleared belief does resolve there, so the loop above is not
    # passing because exact lookup is broken for every id.
    assert (
        exact_lookup_v1(
            conn,
            "belief:home-hosted-ok",
            context=context,
            delivery_target=HOSTED_MODEL_TARGET,
        )
        != []
    )

    # The invariant must prove the gate, not just the refusal: promoting the
    # local-only home belief to hosted_ok lifts exactly that belief into the
    # hosted packet while every wall — neighbour scope and confidentiality —
    # keeps standing.
    assert (
        main(
            [
                "--db",
                str(tmp_path / "egress.sqlite"),
                "egress-promote",
                "belief:home-internal",
                "--approved-by",
                "human:jonathan",
                "--reason",
                "invariant run: a promoted belief must be served",
            ]
        )
        == 0
    )
    packet_after, _handles_after = build_context_v1(
        conn,
        QUERY,
        context=context,
        limit=50,
        delivery_target=HOSTED_MODEL_TARGET,
    )
    served_after = [item["id"] for item in packet_after["items"]]
    assert "belief:home-internal" in served_after
    assert "belief:home-hosted-ok" in served_after
    assert "belief:neighbour-internal" not in served_after
    assert "belief:neighbour-confidential" not in served_after


def test_confidential_material_is_never_served_locally_outside_its_scope(
    tmp_path: Path,
) -> None:
    conn = _seed(tmp_path)
    packet, _handles = build_context_v1(
        conn,
        QUERY,
        context=ScopeContext(project="home"),
        limit=50,
        delivery_target=LOCAL_MODEL_TARGET,
    )
    encoded = json.dumps(packet)
    served = [item["id"] for item in packet["items"]]

    # Scope ranks rather than filters, so the neighbour's ordinary belief IS
    # served. Without this the test below would pass on an empty packet.
    assert "belief:home-internal" in served
    assert "belief:neighbour-internal" in served
    # Confidentiality is not a ranking signal. It is still a wall.
    assert "belief:neighbour-confidential" not in served
    assert "NEIGHBOUR_CONFIDENTIAL_MARKER" not in encoded

    # And the caller's own project still outranks the neighbour's.
    assert served.index("belief:home-internal") < served.index("belief:neighbour-internal")

    # Its own project reaches it; that is what makes this a scope boundary
    # rather than the belief simply being unreachable everywhere.
    owner_packet, _owner_handles = build_context_v1(
        conn,
        QUERY,
        context=ScopeContext(project="neighbour"),
        limit=50,
        delivery_target=LOCAL_MODEL_TARGET,
    )
    assert "belief:neighbour-confidential" in [
        item["id"] for item in owner_packet["items"]
    ]
