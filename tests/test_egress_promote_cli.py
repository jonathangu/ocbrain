"""`ocbrain egress-promote`: the human-attributable lift to hosted egress.

`scope-promote` widens *reach* and never egress, which is why a live brain can
remember 355 things and tell a hosted agent 4 of them. These tests hold down
what an egress promotion must be: an event that survives a full refold, an act
with a named human and a reason behind it, a change to egress ONLY (scope,
visibility, body, and confidence ride through verbatim), a hard refusal for
confidential/secret material, and a dry run that writes nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from ocbrain.cli import main
from ocbrain.core_v1 import (
    append_core_event,
    init_core_v1,
    project_core_v1,
    search_core_v1,
)
from ocbrain.db import connect
from ocbrain.mcp_v1 import decide_proposal_v1
from ocbrain.scope import ScopeContext, ScopeTag

PERSONALIZATION_FACT = "The personalization packet is sealed to gs://coframe-brain-personalization-preview."


def _run(capsys, db: Path, argv: list[str], *, expected: int = 0) -> dict:
    assert main(["--db", str(db), *argv]) == expected
    output = capsys.readouterr().out
    return json.loads(output) if output else {}


def _seed_wiki_fact(
    conn,
    *,
    belief_id: str,
    body: str,
    scope_id: str = "project:coframe-personalization",
    egress_policy: str = "local_only",
    visibility: str = "internal",
) -> None:
    proposal = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": belief_id,
            "belief_type": "wiki_fact",
            "body": body,
            "evidence_ids": [],
            "scope": ScopeTag(
                "project",
                scope_id,
                visibility=visibility,
                egress_policy=egress_policy,
                provenance="wiki_curator",
            ).to_dict(),
            "confidence": 0.9,
            "attributes": {"category": "project", "lifecycle": "durable", "key": belief_id},
        },
        writer="test",
    )
    decide_proposal_v1(
        conn,
        proposal_event_id=proposal,
        decision="approve",
        actor="test",
        edited_body=None,
        reason="egress promote fixture",
    )


def _seeded(tmp_path: Path) -> Path:
    db = tmp_path / "core.sqlite"
    conn = connect(db)
    init_core_v1(conn)
    _seed_wiki_fact(
        conn,
        belief_id="wiki:personalization:packet-target",
        body=PERSONALIZATION_FACT,
    )
    conn.commit()
    conn.close()
    return db


def _event_count(db: Path, kind: str = "egress_promoted") -> int:
    conn = connect(db)
    count = conn.execute(
        f"SELECT COUNT(*) FROM brain_events WHERE kind='{kind}'"  # noqa: S608 - fixed kind
    ).fetchone()[0]
    conn.close()
    return count


def test_egress_promote_rejects_agent_approver_without_writing(tmp_path: Path, capsys) -> None:
    db = _seeded(tmp_path)
    payload = _run(
        capsys,
        db,
        [
            "egress-promote",
            "wiki:personalization:packet-target",
            "--approved-by",
            "agent:self",
            "--reason",
            "an agent must not self-approve hosted egress",
        ],
        expected=2,
    )
    assert payload["status"] == "blocked"
    assert payload["reason"] == "invalid_approved_by"
    assert _event_count(db) == 0


def test_promoting_local_only_belief_makes_it_hosted_serving(tmp_path: Path, capsys) -> None:
    """The whole point: what the delivery gate refused yesterday it serves today."""
    db = _seeded(tmp_path)

    conn = connect(db)
    context = ScopeContext(project="coframe-personalization")
    before = search_core_v1(
        conn,
        "personalization packet sealed",
        context=context,
        limit=5,
        delivery_target="hosted_model",
    )
    assert before["items"] == []
    conn.close()

    applied = _run(
        capsys,
        db,
        [
            "egress-promote",
            "wiki:personalization:packet-target",
            "--approved-by",
            "human:jonathan",
            "--reason",
            "lane knowledge must reach hosted agents",
        ],
    )
    assert applied["status"] == "applied"
    assert [entry["belief_id"] for entry in applied["promoted"]] == [
        "wiki:personalization:packet-target"
    ]
    assert applied["promoted"][0]["from_egress"] == "local_only"
    assert applied["promoted"][0]["to_egress"] == "hosted_ok"

    conn = connect(db)
    row = conn.execute(
        "SELECT egress_policy, scope_provenance, serve, status FROM current_beliefs "
        "WHERE belief_id='wiki:personalization:packet-target'"
    ).fetchone()
    assert str(row["egress_policy"]) == "hosted_ok"
    # The provenance records what lifted it, and a targeted update never
    # disturbs the belief's serving flag or its status.
    assert str(row["scope_provenance"]) == "egress_promoted"
    assert int(row["serve"]) == 1
    assert str(row["status"]) == "current"
    after = search_core_v1(
        conn,
        "personalization packet sealed",
        context=context,
        limit=5,
        delivery_target="hosted_model",
    )
    assert {item["belief_id"] for item in after["items"]} == {
        "wiki:personalization:packet-target"
    }
    conn.close()


def test_confidential_belief_is_refused_reported_and_unchanged(tmp_path: Path, capsys) -> None:
    """Refusals are reported, never silently skipped — and nothing is written."""
    db = _seeded(tmp_path)
    conn = connect(db)
    _seed_wiki_fact(
        conn,
        belief_id="wiki:personalization:client-private",
        body="Client-private incident detail that must never reach a hosted model.",
        visibility="confidential",
    )
    conn.commit()
    conn.close()

    result = _run(
        capsys,
        db,
        [
            "egress-promote",
            "wiki:personalization:client-private",
            "wiki:personalization:packet-target",
            "--approved-by",
            "human:jonathan",
            "--reason",
            "attempting a blanket lift",
        ],
    )

    assert result["status"] == "applied"
    assert [entry["belief_id"] for entry in result["refused"]] == [
        "wiki:personalization:client-private"
    ]
    assert "confidential" in result["refused"][0]["reason"]
    # The ordinary belief in the same run is still promoted.
    assert [entry["belief_id"] for entry in result["promoted"]] == [
        "wiki:personalization:packet-target"
    ]
    conn = connect(db)
    row = conn.execute(
        "SELECT egress_policy FROM current_beliefs "
        "WHERE belief_id='wiki:personalization:client-private'"
    ).fetchone()
    assert str(row["egress_policy"]) == "local_only"
    conn.close()
    assert _event_count(db) == 1


def test_dry_run_writes_no_event(tmp_path: Path, capsys) -> None:
    """A dry run reports exactly what it would do and changes nothing."""
    db = _seeded(tmp_path)

    planned = _run(
        capsys,
        db,
        [
            "egress-promote",
            "wiki:personalization:packet-target",
            "--approved-by",
            "human:jonathan",
            "--reason",
            "preview",
            "--dry-run",
        ],
    )

    assert planned["status"] == "planned"
    assert planned["dry_run"] is True
    assert [entry["belief_id"] for entry in planned["promoted"]] == [
        "wiki:personalization:packet-target"
    ]
    assert planned["promoted"][0]["from_egress"] == "local_only"
    assert planned["promoted"][0]["to_egress"] == "hosted_ok"
    assert "event_id" not in planned["promoted"][0]
    assert _event_count(db) == 0

    conn = connect(db)
    row = conn.execute(
        "SELECT egress_policy FROM current_beliefs "
        "WHERE belief_id='wiki:personalization:packet-target'"
    ).fetchone()
    assert str(row["egress_policy"]) == "local_only"
    conn.close()


def test_rebuild_from_events_reproduces_promoted_egress(tmp_path: Path, capsys) -> None:
    """The ledger is the authority: a full refold must reproduce the promotion."""
    db = _seeded(tmp_path)

    _run(
        capsys,
        db,
        [
            "egress-promote",
            "wiki:personalization:packet-target",
            "--approved-by",
            "human:jonathan",
            "--reason",
            "lane knowledge must reach hosted agents",
        ],
    )
    assert _event_count(db) == 1

    conn = connect(db)
    project_core_v1(conn, full=True)
    row = conn.execute(
        "SELECT egress_policy, scope_provenance, visibility, scope_id FROM current_beliefs "
        "WHERE belief_id='wiki:personalization:packet-target'"
    ).fetchone()
    # Egress survived the refold; scope, visibility, and provenance are exactly
    # what the promotion dictates and nothing else moved.
    assert str(row["egress_policy"]) == "hosted_ok"
    assert str(row["scope_provenance"]) == "egress_promoted"
    assert str(row["visibility"]) == "internal"
    assert str(row["scope_id"]) == "project:coframe-personalization"
    result = search_core_v1(
        conn,
        "personalization packet sealed",
        context=ScopeContext(project="coframe-personalization"),
        limit=5,
        delivery_target="hosted_model",
    )
    assert {item["belief_id"] for item in result["items"]} == {
        "wiki:personalization:packet-target"
    }
    conn.close()


def test_scope_id_selects_every_current_belief_in_scope_and_none_outside(
    tmp_path: Path, capsys
) -> None:
    db = _seeded(tmp_path)
    conn = connect(db)
    _seed_wiki_fact(
        conn,
        belief_id="wiki:personalization:headroom",
        body="Personalization headroom is published to the company dashboard.",
    )
    _seed_wiki_fact(
        conn,
        belief_id="wiki:workspace:sandbox-detail",
        body="The workspace sandbox detail stays where it is.",
        scope_id="project:workspace",
    )
    conn.commit()
    conn.close()

    result = _run(
        capsys,
        db,
        [
            "egress-promote",
            "--scope-id",
            "project:coframe-personalization",
            "--provenance",
            "wiki_curator",
            "--approved-by",
            "human:jonathan",
            "--reason",
            "make lane knowledge visible to hosted agents",
        ],
    )

    assert result["status"] == "applied"
    assert {entry["belief_id"] for entry in result["promoted"]} == {
        "wiki:personalization:packet-target",
        "wiki:personalization:headroom",
    }
    assert result["refused"] == []
    assert result["missing"] == []
    # The selection is the promotion: every selected belief actually carries
    # the new egress and the egress_promoted provenance after the fold.
    conn = connect(db)
    rows = conn.execute(
        "SELECT belief_id, egress_policy, scope_provenance FROM current_beliefs "
        "WHERE scope_id='project:coframe-personalization' AND status='current'"
    ).fetchall()
    assert {
        (str(row["belief_id"]), str(row["egress_policy"]), str(row["scope_provenance"]))
        for row in rows
    } == {
        ("wiki:personalization:packet-target", "hosted_ok", "egress_promoted"),
        ("wiki:personalization:headroom", "hosted_ok", "egress_promoted"),
    }
    # The out-of-scope belief is untouched and not even reported as refused.
    row = conn.execute(
        "SELECT egress_policy FROM current_beliefs WHERE belief_id='wiki:workspace:sandbox-detail'"
    ).fetchone()
    assert str(row["egress_policy"]) == "local_only"
    conn.close()
    assert _event_count(db) == 2