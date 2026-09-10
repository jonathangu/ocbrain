from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ocbrain.cli import main as cli_main
from ocbrain.core_v1 import (
    append_core_event,
    init_core_v1,
    project_core_v1,
    search_core_v1,
)
from ocbrain.db import connect
from ocbrain.hygiene import (
    apply_retirements,
    plan_retirements,
    restore,
    supersede,
    verify_serving_invariants,
)
from ocbrain.mcp_v1 import decide_proposal_v1, pending_supersede_count
from ocbrain.scope import ScopeContext, ScopeTag

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
SCOPE = ScopeTag(
    "project",
    "project:bountiful",
    visibility="internal",
    egress_policy="local_only",
    provenance="test",
)


def _core(tmp_path: Path):
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    return conn


def _seed(
    conn,
    *,
    belief_id: str,
    body: str,
    attributes: dict | None = None,
    belief_type: str = "curated_fact",
    compiled_at: str | None = None,
    pinned: bool = False,
    scope: ScopeTag = SCOPE,
) -> None:
    proposal = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": belief_id,
            "belief_type": belief_type,
            "body": body,
            "evidence_ids": [],
            "scope": scope.to_dict(),
            "confidence": 0.9,
            "attributes": attributes or {},
        },
        writer="test",
        ts=compiled_at,
    )
    if compiled_at is None:
        decide_proposal_v1(
            conn,
            proposal_event_id=proposal,
            decision="approve",
            actor="test",
            edited_body=None,
            reason="seed",
        )
    else:
        # last_compiled_at comes from the *decision* event, so backdating a belief
        # means backdating the approval rather than the proposal.
        append_core_event(
            conn,
            "compilation_decided",
            {"proposal_event_id": proposal, "decision": "approve", "actor": "test"},
            writer="test",
            ts=compiled_at,
            project=True,
        )
    if pinned:
        append_core_event(
            conn,
            "correction_recorded",
            {
                "subject": {"kind": "belief", "id": belief_id},
                "target_id": belief_id,
                "target_layer": "belief",
                "op": "pin",
                "author": "test",
                "hard": False,
            },
            writer="test",
            project=True,
        )


def _judge(conn, *, belief_id: str, outcome: str, count: int, served_at: str, prefix: str) -> None:
    for index in range(count):
        use_id = f"ret_{prefix}_{index}"
        conn.execute(
            "INSERT INTO retrieval_uses (id, served_to_runtime, outcome, served_at) "
            "VALUES (?,?,?,?)",
            (use_id, "test", outcome, served_at),
        )
        conn.execute(
            "INSERT INTO retrieval_items (retrieval_use_id, object_id, object_kind, rank, score) "
            "VALUES (?,?,?,?,?)",
            (use_id, belief_id, "belief", 1, 0.5),
        )


def test_expired_class_retires_past_valid_until(tmp_path: Path) -> None:
    conn = _core(tmp_path)
    stale = "curated:bountiful:stale"
    fresh = "curated:bountiful:fresh"
    _seed(
        conn,
        belief_id=stale,
        body="The staging cluster runs three replicas this quarter.",
        attributes={"valid_until": "2026-07-01T00:00:00+00:00", "lifecycle": "current"},
    )
    _seed(
        conn,
        belief_id=fresh,
        body="The staging cluster is provisioned from Terraform.",
        attributes={"valid_until": "2027-01-01T00:00:00+00:00", "lifecycle": "current"},
    )
    conn.commit()

    plan = plan_retirements(conn, classes=("expired",), now=NOW)
    assert [target["belief_id"] for target in plan["targets"]] == [stale]
    assert plan["targets"][0]["reason"] == "expired"

    apply_retirements(conn, plan)
    rows = dict(
        conn.execute("SELECT belief_id, status FROM current_beliefs").fetchall()  # type: ignore[arg-type]
    )
    assert rows[stale] == "retracted"
    assert rows[fresh] == "current"


def test_supersede_retires_the_old_belief_immediately(tmp_path: Path) -> None:
    """An operator's supersession takes effect when they make it, not next sweep.

    This used to stamp ``superseded_by`` on the still-serving belief and leave
    the retirement to the next ``expired`` run, so until then the corpus served
    the fact the operator had just declared wrong *alongside* its replacement --
    up to a day on the scheduled cadence, with nothing telling a reader which
    was which.
    """
    conn = _core(tmp_path)
    old = "curated:bountiful:old-rule"
    new = "curated:bountiful:new-rule"
    _seed(conn, belief_id=old, body="Inventory decrements on reservation.")
    _seed(conn, belief_id=new, body="Inventory decrements on completed exchange.")
    conn.commit()

    result = supersede(conn, belief_id=old, successor_id=new)
    assert result["status"] == "retracted"
    assert result["serve"] is False
    assert result["attributes"]["superseded_by"] == new
    assert result["attributes"]["valid_until"]

    rows = dict(
        conn.execute("SELECT belief_id, serve FROM current_beliefs").fetchall()
    )
    assert rows[old] == 0
    assert rows[new] == 1
    # Nothing is left for the sweep to do; the retirement already happened.
    assert plan_retirements(conn, classes=("expired",), now=NOW)["targets"] == []


def test_expired_class_still_retires_a_belief_marked_superseded_by_hand(
    tmp_path: Path,
) -> None:
    """The attribute path predates the correction op and still has to work.

    Beliefs stamped ``superseded_by`` before this release, or by a curator
    writing the attribute directly, are still serving and still have to be
    retired by the sweep.
    """
    conn = _core(tmp_path)
    old = "curated:bountiful:hand-marked"
    _seed(
        conn,
        belief_id=old,
        body="Inventory decrements on reservation.",
        attributes={"superseded_by": "curated:bountiful:new-rule"},
    )
    conn.commit()

    plan = plan_retirements(conn, classes=("expired",), now=NOW)
    assert [target["belief_id"] for target in plan["targets"]] == [old]
    assert "superseded by" in plan["targets"][0]["detail"]


def test_supersede_refuses_self_and_unknown_targets(tmp_path: Path) -> None:
    conn = _core(tmp_path)
    known = "curated:bountiful:known"
    _seed(conn, belief_id=known, body="A known and currently served fact body.")
    conn.commit()

    with pytest.raises(ValueError, match="cannot supersede itself"):
        supersede(conn, belief_id=known, successor_id=known)
    with pytest.raises(ValueError, match="successor belief not found"):
        supersede(conn, belief_id=known, successor_id="curated:bountiful:ghost")
    with pytest.raises(ValueError, match="belief not found"):
        supersede(conn, belief_id="curated:bountiful:ghost", successor_id=known)


def test_a_swept_belief_can_be_restored(tmp_path: Path) -> None:
    """An unattended sweep is only safe if a wrong call is one command to undo."""
    conn = _core(tmp_path)
    target = "curated:bountiful:reapprovable"
    _seed(
        conn,
        belief_id=target,
        body="A fact that expires and is later put back into service.",
        attributes={"valid_until": "2026-07-01T00:00:00+00:00"},
    )
    conn.commit()
    apply_retirements(conn, plan_retirements(conn, classes=("expired",), now=NOW))
    assert (
        conn.execute(
            "SELECT status FROM current_beliefs WHERE belief_id=?", (target,)
        ).fetchone()["status"]
        == "retracted"
    )

    assert restore(conn, belief_id=target)["changed"] is True
    row = conn.execute(
        "SELECT status, serve FROM current_beliefs WHERE belief_id=?", (target,)
    ).fetchone()
    assert row["status"] == "current"
    assert bool(row["serve"]) is True
    # Serving it again means the search index came back too, not just the row.
    assert verify_serving_invariants(conn)["serving"] == 1
    # Idempotent: restoring a live belief is a no-op, not a second event.
    assert restore(conn, belief_id=target)["changed"] is False


def test_restore_survives_a_full_projection_rebuild(tmp_path: Path) -> None:
    conn = _core(tmp_path)
    target = "curated:bountiful:rebuilt"
    _seed(
        conn,
        belief_id=target,
        body="A fact retired then restored before a rebuild happens.",
        attributes={"valid_until": "2026-07-01T00:00:00+00:00"},
    )
    conn.commit()
    apply_retirements(conn, plan_retirements(conn, classes=("expired",), now=NOW))
    restore(conn, belief_id=target)

    project_core_v1(conn, full=True)
    row = conn.execute(
        "SELECT status, serve FROM current_beliefs WHERE belief_id=?", (target,)
    ).fetchone()
    assert row["status"] == "current"
    assert bool(row["serve"]) is True


def test_restore_refuses_hard_corrected_and_tombstoned_beliefs(tmp_path: Path) -> None:
    """Those were deliberate permanent decisions and must stay terminal."""
    conn = _core(tmp_path)
    hard_target = "curated:bountiful:hard-gone"
    tombstoned = "curated:bountiful:tombstoned"
    _seed(conn, belief_id=hard_target, body="A fact retired permanently on purpose.")
    _seed(conn, belief_id=tombstoned, body="A fact forgotten permanently on purpose.")
    conn.commit()

    append_core_event(
        conn,
        "correction_recorded",
        {
            "subject": {"kind": "belief", "id": hard_target},
            "target_id": hard_target,
            "target_layer": "belief",
            "op": "retract",
            "author": "test",
            "hard": True,
        },
        writer="test",
        project=True,
    )
    append_core_event(
        conn,
        "tombstone_recorded",
        {"target": tombstoned, "mode": "soft", "reason": "test"},
        writer="test",
        project=True,
    )
    conn.commit()

    with pytest.raises(PermissionError, match="hard-corrected"):
        restore(conn, belief_id=hard_target)
    with pytest.raises(PermissionError, match="tombstoned"):
        restore(conn, belief_id=tombstoned)

    # And a forged restore event must lose in the projection, not just at the API.
    for target in (hard_target, tombstoned):
        append_core_event(
            conn,
            "correction_recorded",
            {
                "subject": {"kind": "belief", "id": target},
                "target_id": target,
                "target_layer": "belief",
                "op": "restore",
                "author": "forged",
                "hard": False,
            },
            writer="forged",
            project=True,
        )
    conn.commit()
    project_core_v1(conn, full=True)
    statuses = {
        str(row["belief_id"]): str(row["status"])
        for row in conn.execute("SELECT belief_id, status FROM current_beliefs")
    }
    assert statuses[hard_target] == "retracted"
    assert statuses[tombstoned] == "tombstoned"


def test_a_second_run_retires_nothing(tmp_path: Path) -> None:
    conn = _core(tmp_path)
    for index in range(3):
        _seed(
            conn,
            belief_id=f"curated:bountiful:expired-{index}",
            body=f"An expired fact body number {index} for the sweep.",
            attributes={"valid_until": "2026-07-01T00:00:00+00:00"},
        )
    conn.commit()

    first = apply_retirements(conn, plan_retirements(conn, classes=("expired",), now=NOW))
    assert first["applied"] == 3
    second = plan_retirements(conn, classes=("expired",), now=NOW)
    assert second["targets"] == []


def test_batch_cap_bounds_a_run_and_reports_the_remainder(tmp_path: Path) -> None:
    conn = _core(tmp_path)
    for index in range(5):
        _seed(
            conn,
            belief_id=f"curated:bountiful:capped-{index}",
            body=f"An expired fact body number {index} awaiting retirement.",
            attributes={"valid_until": "2026-07-01T00:00:00+00:00"},
        )
    conn.commit()

    plan = plan_retirements(conn, classes=("expired",), now=NOW, batch_cap=2)
    assert plan["selected_total"] == 2
    assert plan["eligible_total"] == 5
    # Silent truncation would read as "nothing left to do" on the next run.
    assert plan["deferred_by_cap"] == 3


def test_retired_belief_leaves_the_search_index_and_stops_being_served(tmp_path: Path) -> None:
    conn = _core(tmp_path)
    target = "curated:bountiful:searchable"
    _seed(
        conn,
        belief_id=target,
        body="Meyer lemons ripen in winter near the coast.",
        attributes={"valid_until": "2026-07-01T00:00:00+00:00"},
    )
    conn.commit()
    before = search_core_v1(
        conn,
        "meyer lemons ripen winter coast",
        context=ScopeContext(project="bountiful"),
        limit=5,
    )
    assert [item["belief_id"] for item in before["items"]] == [target]

    apply_retirements(conn, plan_retirements(conn, classes=("expired",), now=NOW))
    after = search_core_v1(
        conn,
        "meyer lemons ripen winter coast",
        context=ScopeContext(project="bountiful"),
        limit=5,
    )
    assert after["items"] == []
    assert verify_serving_invariants(conn) == {"serving": 0, "unserved_in_search_index": 0}


def _pend_supersede(conn, *, target: str, successor: str) -> str:
    return append_core_event(
        conn,
        "compilation_proposed",
        {
            "schema_version": "ocbrain.compilation.v1",
            "subject": {"kind": "belief", "id": successor},
            "belief_id": successor,
            "belief_type": "wiki_fact",
            "body": "The replacement statement awaiting a decision.",
            "evidence_ids": [],
            "scope": SCOPE.to_dict(),
            "confidence": 0.7,
            "attributes": {"supersedes": target},
        },
        writer="wiki-curator",
    )


def test_moot_proposals_class_rejects_supersedes_of_retired_targets(tmp_path: Path) -> None:
    """A proposal whose target is already retired can never be decided.

    Nothing the decision does can serve anything differently, so it sits in the
    queue forever and ages the headline metric. The class appends a rejection
    instead of touching the belief -- append-only, nothing deleted.
    """
    conn = _core(tmp_path)
    target = "curated:bountiful:moot-target"
    successor = "belief:bountiful:moot-successor"
    _seed(
        conn,
        belief_id=target,
        body="A fact that expires while a supersession of it is undecided.",
        attributes={"valid_until": "2026-07-01T00:00:00+00:00"},
    )
    proposal = _pend_supersede(conn, target=target, successor=successor)
    conn.commit()
    assert pending_supersede_count(conn) == 1
    # Still serving: the proposal is a live decision and must be left alone.
    assert plan_retirements(conn, classes=("moot_proposals",), now=NOW)["targets"] == []

    apply_retirements(conn, plan_retirements(conn, classes=("expired",), now=NOW))
    plan = plan_retirements(conn, classes=("moot_proposals",), now=NOW)
    assert [item["belief_id"] for item in plan["targets"]] == [target]
    assert plan["targets"][0]["proposal_event_id"] == proposal
    assert plan["targets_by_reason"] == {"moot_proposals": 1}

    applied = apply_retirements(conn, plan)
    assert applied["moot_proposals_rejected"] == 1
    assert applied["applied"] == 0

    decisions = [
        json.loads(str(row["body_json"]))
        for row in conn.execute(
            "SELECT body_json FROM brain_events WHERE kind='compilation_decided' "
            "AND json_extract(body_json, '$.proposal_event_id')=?",
            (proposal,),
        )
    ]
    assert len(decisions) == 1
    assert decisions[0]["decision"] == "reject"
    assert decisions[0]["reason"] == "target_not_current"
    assert decisions[0]["actor"].startswith("maintenance:")
    assert pending_supersede_count(conn) == 0
    conn.close()


def test_moot_proposals_class_leaves_live_targets_alone(tmp_path: Path) -> None:
    """The queue is a backlog, not garbage: a decidable proposal stays decidable."""
    conn = _core(tmp_path)
    target = "curated:bountiful:live-target"
    successor = "belief:bountiful:live-successor"
    _seed(conn, belief_id=target, body="A fact still serving while its correction waits.")
    proposal = _pend_supersede(conn, target=target, successor=successor)
    conn.commit()

    plan = plan_retirements(conn, classes=("moot_proposals",), now=NOW)
    assert plan["targets"] == []
    applied = apply_retirements(conn, plan)
    assert applied["moot_proposals_rejected"] == 0
    assert pending_supersede_count(conn) == 1
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM brain_events WHERE kind='compilation_decided' "
            "AND json_extract(body_json, '$.proposal_event_id')=?",
            (proposal,),
        ).fetchone()[0]
        == 0
    )
    conn.close()


def test_unknown_class_is_rejected(tmp_path: Path) -> None:
    conn = _core(tmp_path)
    conn.commit()
    with pytest.raises(ValueError, match="unknown hygiene classes"):
        plan_retirements(conn, classes=("expired", "nonsense"), now=NOW)


def test_cli_hygiene_reports_then_applies(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "core.sqlite"
    conn = connect(db_path)
    init_core_v1(conn)
    target = "curated:bountiful:cli-expired"
    _seed(
        conn,
        belief_id=target,
        body="An expired fact reachable through the command line.",
        attributes={"valid_until": "2026-07-01T00:00:00+00:00"},
    )
    conn.commit()
    conn.close()

    assert cli_main(["--db", str(db_path), "hygiene", "--class", "expired"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["action"] == "hygiene"
    assert report["apply_requested"] is False
    assert report["selected_total"] == 1

    assert cli_main(["--db", str(db_path), "hygiene", "--class", "expired", "--apply"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["apply_requested"] is True
    assert applied["applied"] == 1
    assert applied["invariants"]["unserved_in_search_index"] == 0


def test_closeout_records_evidence_but_promotes_nothing(tmp_path: Path) -> None:
    """Recording evidence is not promotion, and must not be gated with it.

    Both used to sit behind the automatic_activation flag, so turning that flag
    off to stop unattended promotion also stopped closeout summaries becoming
    evidence -- silently removing the largest supply of curator-eligible
    evidence. The flag is gone; the separation it broke is what this pins.
    """
    from ocbrain.mcp_v1 import closeout_v1

    conn = _core(tmp_path)
    receipt = closeout_v1(
        conn,
        task_ref="task-with-activation-off",
        status="completed",
        summary="The nightly export now finishes before the morning report runs.",
        context=ScopeContext(project="bountiful"),
        retrieval_use_ids=[],
        decision_impact="changed",
        decision_note="verified locally",
        artifact_refs=[],
        verifier_refs=[],
        actions=[],
        outcomes=[],
        awaiting=None,
        actor="test",
    )
    conn.commit()

    # Evidence lands, and is the curator-eligible kind.
    assert receipt["evidence_id"]
    row = conn.execute(
        "SELECT kind, scope_id, egress_policy FROM evidence_objects WHERE evidence_id=?",
        (receipt["evidence_id"],),
    ).fetchone()
    assert row["kind"] == "task_closeout_summary"
    assert row["scope_id"] == "project:bountiful"
    # But nothing was promoted: compilation stays a human-gated curation step.
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM current_beliefs WHERE serve=1 AND status='current'"
        ).fetchone()[0]
        == 0
    )


def test_redundant_class_keeps_the_newest_restatement(tmp_path: Path) -> None:
    """A reworded fact under a new key is one fact, not two.

    The compiler keys a belief by the topic name a model chose, so a later run
    that rewords the same fact mints a second belief and exact-body dedup never
    sees it. Unchecked, every scheduled run adds a phrasing and each copy costs a
    retrieval slot.
    """
    conn = _core(tmp_path)
    old = "curated:bountiful:hermes-runtime-config"
    new = "curated:bountiful:hermes-runtime-service"
    distinct = "curated:bountiful:unrelated"
    _seed(
        conn,
        belief_id=old,
        body="Hermes runs as the launchd service ai.hermes.gateway with auto-start and restart.",
        belief_type="wiki_fact",
        compiled_at="2026-08-01T00:00:00+00:00",
    )
    _seed(
        conn,
        belief_id=new,
        body="Hermes runs as launchd service ai.hermes.gateway with auto-start and auto-restart.",
        belief_type="wiki_fact",
        compiled_at="2026-08-04T00:00:00+00:00",
    )
    _seed(
        conn,
        belief_id=distinct,
        body="Meyer lemons ripen in winter near the coast and are picked by hand.",
        belief_type="wiki_fact",
        compiled_at="2026-08-04T00:00:00+00:00",
    )
    conn.commit()

    plan = plan_retirements(conn, classes=("redundant",), now=NOW)
    assert [target["belief_id"] for target in plan["targets"]] == [old]
    assert plan["targets"][0]["detail"] == f"restates {new}"

    apply_retirements(conn, plan)
    statuses = {
        str(row["belief_id"]): str(row["status"])
        for row in conn.execute("SELECT belief_id, status FROM current_beliefs")
    }
    assert statuses[old] == "retracted"
    assert statuses[new] == "current"
    assert statuses[distinct] == "current"
    # Idempotent: with the older copy gone there is nothing left to collapse.
    assert plan_retirements(conn, classes=("redundant",), now=NOW)["targets"] == []


def test_redundant_class_spares_pinned_beliefs_and_respects_the_threshold(
    tmp_path: Path,
) -> None:
    conn = _core(tmp_path)
    pinned_old = "curated:bountiful:pinned-phrasing"
    newer = "curated:bountiful:newer-phrasing"
    _seed(
        conn,
        belief_id=pinned_old,
        body="Hermes runs as the launchd service ai.hermes.gateway with auto-start and restart.",
        belief_type="wiki_fact",
        compiled_at="2026-08-01T00:00:00+00:00",
        pinned=True,
    )
    _seed(
        conn,
        belief_id=newer,
        body="Hermes runs as launchd service ai.hermes.gateway with auto-start and auto-restart.",
        belief_type="wiki_fact",
        compiled_at="2026-08-04T00:00:00+00:00",
    )
    conn.commit()

    # A pinned belief is an operator decision and is never collapsed away.
    assert plan_retirements(conn, classes=("redundant",), now=NOW)["targets"] == []
    # And a threshold above the pair's actual overlap finds nothing at all.
    assert (
        plan_retirements(
            conn, classes=("redundant",), now=NOW, restatement_threshold=0.999
        )["targets"]
        == []
    )


def test_redundant_class_preserves_scope_and_belief_type_boundaries(tmp_path: Path) -> None:
    conn = _core(tmp_path)
    body = "OCBrain runs as an on-demand local stdio MCP service with eight tools."
    alpha = ScopeTag("project", "project:alpha", "internal", "local_only", "test")
    beta = ScopeTag("project", "project:beta", "internal", "local_only", "test")

    _seed(
        conn,
        belief_id="curated:alpha:runtime",
        body=body,
        belief_type="wiki_fact",
        compiled_at="2026-08-01T00:00:00+00:00",
        scope=alpha,
    )
    _seed(
        conn,
        belief_id="curated:beta:runtime",
        body=body,
        belief_type="wiki_fact",
        compiled_at="2026-08-04T00:00:00+00:00",
        scope=beta,
    )
    _seed(
        conn,
        belief_id="explicit:alpha:runtime",
        body=body,
        belief_type="curated_fact",
        compiled_at="2026-07-01T00:00:00+00:00",
        scope=alpha,
    )
    conn.commit()

    # Matching text does not make separately scoped or non-wiki knowledge redundant.
    assert plan_retirements(conn, classes=("redundant",), now=NOW)["targets"] == []
