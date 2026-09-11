"""Temporal retrieval must not reopen terminal lifecycle removals."""
from __future__ import annotations

import pytest
from test_hybrid_v1 import _seed_belief

from ocbrain.core_v1 import append_core_event, init_core_v1, search_core_v1
from ocbrain.db import connect
from ocbrain.mcp_v1 import build_context_v1
from ocbrain.scope import ScopeContext


@pytest.mark.parametrize("terminal", ["tombstone", "hard_retract"])
@pytest.mark.parametrize("as_of", ["2025-01-01T00:00:00Z", "2099-01-01T00:00:00Z"])
@pytest.mark.parametrize("delivery", ["local_model", "hosted_model"])
@pytest.mark.parametrize("superseded", [False, True])
def test_as_of_respects_terminal_removals(tmp_path, terminal, as_of, delivery, superseded):
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    target = "curated:test:terminal"
    text = "Synthetic quince gate privacy marker"
    _seed_belief(conn, belief_id=target, body=text,
                 attributes={"valid_from": "2020-01-01T00:00:00+00:00"})
    if superseded:
        append_core_event(conn, "correction_recorded", {
            "target_layer": "belief", "target_id": target, "op": "supersede",
            "successor_id": "curated:test:successor", "author": "test",
        }, writer="test", project=True)
    if terminal == "tombstone":
        append_core_event(conn, "tombstone_recorded", {
            "target": target, "mode": "hide", "author": "test",
        }, writer="test", project=True)
    else:
        append_core_event(conn, "correction_recorded", {
            "target_layer": "belief", "target_id": target, "op": "retract",
            "hard": True, "author": "test",
        }, writer="test", project=True)
    conn.commit()
    context = ScopeContext(project="bountiful")
    assert not search_core_v1(conn, text, context=context)['items']
    packet, _ = build_context_v1(
        conn, query=text, context=context, limit=5, as_of=as_of, delivery_target=delivery,
    )
    assert not packet["items"]
    assert packet["coverage"]["ranking"]["eligible_count"] == 0
    conn.close()


def test_as_of_does_not_invent_an_era_for_unbounded_soft_retraction(tmp_path):
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    target = "curated:test:unbounded"
    text = "Synthetic orchard retirement without historical era"
    _seed_belief(conn, belief_id=target, body=text,
                 attributes={"valid_from": "2020-01-01T00:00:00+00:00"})
    append_core_event(conn, "correction_recorded", {
        "target_layer": "belief", "target_id": target, "op": "retract", "author": "test",
    }, writer="test", project=True)
    conn.commit()
    assert not search_core_v1(conn, text, context=ScopeContext(project="bountiful"),
                              as_of="2099-01-01T00:00:00Z")["items"]
    conn.close()
