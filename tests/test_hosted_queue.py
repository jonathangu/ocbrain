"""D3: the CLI-only hosted-queue and hosted-approve verbs.

The queue is a read-only listing of approval_required evidence a human could
promote to hosted_ok project beliefs. The approve verb runs the eligibility
gauntlet (confidential/secret refusal, egress check, secret-leak scan) and
mints the same proposal + decision events `event-compile --approve` writes.
"""

from __future__ import annotations

import json

from ocbrain.cli import main
from ocbrain.core_v1 import (
    append_core_event,
    init_core_v1,
    is_core_v1,
    record_core_v1_evidence,
)
from ocbrain.db import connect
from ocbrain.mcp import tool_list
from ocbrain.mcp_v1 import build_context_v1, proposals_v1
from ocbrain.scope import ScopeContext, ScopeTag

APPROVER = "human:test-approver"


def _run(capsys, db, argv):
    rc = main(["--db", db, *argv])
    captured = capsys.readouterr()
    return rc, json.loads(captured.out)


def _seed_core(tmp_path):
    conn = connect(tmp_path / "hosted-queue.sqlite")
    init_core_v1(conn)
    return conn


def _seed_evidence(
    conn,
    *,
    body: str,
    scope: ScopeTag | None = None,
    kind: str = "observation",
    writer: str = "test-agent",
) -> str:
    scope = scope or ScopeTag(
        "project",
        "project:coframe",
        visibility="internal",
        egress_policy="approval_required",
        provenance="inferred",
    )
    evidence_id, _event_id = record_core_v1_evidence(
        conn,
        body=body,
        kind=kind,
        scope=scope,
        writer=writer,
    )
    conn.commit()
    return evidence_id


def _confidential_scope() -> ScopeTag:
    return ScopeTag(
        "client",
        "client:bihua",
        visibility="confidential",
        egress_policy="approval_required",
        provenance="inferred",
    )


def _event_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0]


def _seed_task_evidence_with_project_proposal(conn, *, project: str = "coframe"):
    """Seed the live defect shape: approval_required task evidence whose own
    widening request asks for the project's hosted_ok scope."""
    evidence_id = _seed_evidence(
        conn,
        body="Hosted-queue fixture ingested from a task context.",
        scope=ScopeTag(
            "task",
            "task:brain-daily-planner-20260904",
            visibility="internal",
            egress_policy="approval_required",
            provenance="inferred",
        ),
    )
    proposal_event_id = append_core_event(
        conn,
        "hosted_egress_proposal",
        {
            "schema_version": "ocbrain.hosted-egress-proposal.v1",
            "subject": {"kind": "evidence", "id": evidence_id},
            "evidence_id": evidence_id,
            "requested_scope": {
                "scope_type": "project",
                "scope_id": f"project:{project}",
                "visibility": "internal",
                "egress_policy": "hosted_ok",
                "provenance": "explicit",
            },
            "inferred_scope": {
                "scope_type": "task",
                "scope_id": "task:brain-daily-planner-20260904",
                "visibility": "internal",
                "egress_policy": "approval_required",
                "provenance": "inferred",
            },
            "applied_scope": {
                "scope_type": "task",
                "scope_id": "task:brain-daily-planner-20260904",
                "visibility": "internal",
                "egress_policy": "approval_required",
                "provenance": "inferred",
            },
            "writer": "test-agent",
            "body_head": "Hosted-queue fixture ingested from a task context.",
        },
        writer="test-agent",
    )
    conn.commit()
    return evidence_id, proposal_event_id


def test_hosted_queue_lists_eligible_rows_and_omits_confidential(tmp_path, capsys):
    conn = _seed_core(tmp_path)
    eligible = _seed_evidence(conn, body="Eligible approval_required observation.")
    _seed_evidence(conn, body="Confidential client observation.", scope=_confidential_scope())
    db = str(tmp_path / "hosted-queue.sqlite")
    rc, payload = _run(capsys, db, ["hosted-queue"])
    assert rc == 0
    assert payload["action"] == "hosted-queue"
    ids = [entry["evidence_id"] for entry in payload["queue"]]
    assert eligible in ids
    assert len(ids) == 1
    entry = payload["queue"][0]
    assert entry["scope_id"] == "project:coframe"
    assert entry["writer"] == "test-agent"
    assert "Eligible" in entry["body_head"]
    conn.close()


def test_hosted_queue_is_read_only(tmp_path, capsys):
    conn = _seed_core(tmp_path)
    _seed_evidence(conn, body="Read-only queue check.")
    db = str(tmp_path / "hosted-queue.sqlite")
    before = _event_count(conn)
    _run(capsys, db, ["hosted-queue", "--project", "coframe"])
    assert _event_count(conn) == before
    conn.close()


def test_hosted_queue_includes_pending_widening_proposals(tmp_path, capsys):
    conn = _seed_core(tmp_path)
    evidence_id = _seed_evidence(conn, body="Widening request fixture.")
    append_core_event(
        conn,
        "hosted_egress_proposal",
        {
            "schema_version": "ocbrain.hosted-egress-proposal.v1",
            "subject": {"kind": "evidence", "id": evidence_id},
            "evidence_id": evidence_id,
            "requested_scope": {
                "scope_type": "project",
                "scope_id": "project:coframe",
                "visibility": "internal",
                "egress_policy": "hosted_ok",
                "provenance": "explicit",
            },
            "inferred_scope": {
                "scope_type": "task",
                "scope_id": "task:t",
                "visibility": "internal",
                "egress_policy": "approval_required",
                "provenance": "inferred",
            },
            "applied_scope": {
                "scope_type": "task",
                "scope_id": "task:t",
                "visibility": "internal",
                "egress_policy": "approval_required",
                "provenance": "inferred",
            },
            "writer": "test-agent",
            "body_head": "Widening request fixture.",
        },
        writer="test-agent",
    )
    conn.commit()
    db = str(tmp_path / "hosted-queue.sqlite")
    rc, payload = _run(capsys, db, ["hosted-queue"])
    assert rc == 0
    assert payload["proposal_count"] == 1
    assert payload["proposals"][0]["evidence_id"] == evidence_id
    assert payload["proposals"][0]["requested_scope"]["egress_policy"] == "hosted_ok"
    conn.close()


def test_hosted_approve_creates_hosted_ok_belief_retrievable_by_hosted_delivery(
    tmp_path, capsys
):
    conn = _seed_core(tmp_path)
    evidence_id = _seed_evidence(
        conn,
        body="The Bountiful deploy path is documented in runbooks/deploy.md.",
    )
    conn.commit()
    db = str(tmp_path / "hosted-queue.sqlite")
    rc, payload = _run(
        capsys,
        db,
        [
            "hosted-approve",
            evidence_id,
            "--approved-by",
            APPROVER,
            "--reason",
            "operational fact, safe to serve hosted",
        ],
    )
    assert rc == 0, payload
    assert payload["status"] == "applied"
    assert payload["approved_by"] == APPROVER
    entry = payload["approved"][0]
    assert entry["scope"]["egress_policy"] == "hosted_ok"
    assert entry["scope"]["scope_id"] == "project:coframe"
    # No proposal seeded: the target falls back to the evidence row's own scope.
    assert entry["scope_source"] == "evidence_row"

    belief = conn.execute(
        "SELECT * FROM current_beliefs WHERE belief_id=?", (entry["belief_id"],)
    ).fetchone()
    assert belief is not None
    assert belief["status"] == "current"
    assert belief["serve"] == 1
    assert belief["egress_policy"] == "hosted_ok"
    assert belief["scope_provenance"] == "human_approved_hosted"
    link = conn.execute(
        "SELECT * FROM belief_evidence WHERE evidence_id=? AND belief_id=?",
        (evidence_id, entry["belief_id"]),
    ).fetchone()
    assert link is not None
    decision = conn.execute(
        "SELECT * FROM brain_events WHERE id=?", (entry["decision_event_id"],)
    ).fetchone()
    assert decision is not None
    assert decision["kind"] == "compilation_decided"
    assert decision["writer"] == APPROVER

    # The compiled belief must actually serve on the hosted delivery target.
    conn.commit()
    packet, _receipts = build_context_v1(
        conn,
        "deploy path",
        context=ScopeContext(project="coframe"),
        limit=5,
        delivery_target="hosted_model",
    )
    served_ids = {item["id"] for item in packet["items"]}
    assert entry["belief_id"] in served_ids
    conn.close()


def test_hosted_approve_lands_the_proposals_requested_project(tmp_path, capsys):
    """Approving the queue answers the row's own widening request: without
    --project the belief used to land at the evidence's task scope, silently
    dropping the reach the request named."""
    conn = _seed_core(tmp_path)
    evidence_id, proposal_event_id = _seed_task_evidence_with_project_proposal(conn)
    db = str(tmp_path / "hosted-queue.sqlite")
    rc, payload = _run(
        capsys,
        db,
        ["hosted-approve", evidence_id, "--approved-by", APPROVER],
    )
    assert rc == 0, payload
    entry = payload["approved"][0]
    assert entry["scope"]["scope_type"] == "project"
    assert entry["scope"]["scope_id"] == "project:coframe"
    assert entry["scope"]["egress_policy"] == "hosted_ok"
    assert entry["scope_source"] == "requested_by_proposal"

    belief = conn.execute(
        "SELECT * FROM current_beliefs WHERE belief_id=?", (entry["belief_id"],)
    ).fetchone()
    assert belief is not None
    assert belief["scope_type"] == "project"
    assert belief["scope_id"] == "project:coframe"
    assert belief["egress_policy"] == "hosted_ok"
    assert belief["scope_provenance"] == "human_approved_hosted"
    proposal = conn.execute(
        "SELECT body_json FROM brain_events WHERE id=?", (entry["proposal_event_id"],)
    ).fetchone()
    attributes = json.loads(proposal["body_json"])["attributes"]
    assert attributes["answers_proposal"] == proposal_event_id
    assert all(
        item["proposal_event_id"] != proposal_event_id
        for item in proposals_v1(conn, limit=50, include_decided=False)["proposals"]
    )
    decided = {
        item["proposal_event_id"]: item
        for item in proposals_v1(conn, limit=50, include_decided=True)["proposals"]
    }
    assert decided[proposal_event_id]["decided"] is True
    conn.close()


def test_all_from_project_queue_includes_task_scoped_widening_proposals(tmp_path, capsys):
    """The project filter selects a proposal by its requested scope even though
    its evidence remains task-scoped until the human approval is applied."""
    conn = _seed_core(tmp_path)
    evidence_id, proposal_event_id = _seed_task_evidence_with_project_proposal(conn)
    db = str(tmp_path / "hosted-queue.sqlite")

    rc, payload = _run(
        capsys,
        db,
        [
            "hosted-approve",
            "--all-from-queue",
            "--project",
            "coframe",
            "--approved-by",
            APPROVER,
        ],
    )

    assert rc == 0, payload
    assert payload["status"] == "applied"
    assert [entry["evidence_id"] for entry in payload["approved"]] == [evidence_id]
    entry = payload["approved"][0]
    assert entry["scope"]["scope_id"] == "project:coframe"
    assert entry["scope_source"] == "cli_project"
    compilation = conn.execute(
        "SELECT body_json FROM brain_events WHERE kind='compilation_proposed' "
        "AND json_extract(body_json, '$.attributes.answers_proposal')=?",
        (proposal_event_id,),
    ).fetchone()
    assert compilation is not None
    conn.close()


def test_hosted_approve_cli_project_overrides_the_proposal(tmp_path, capsys):
    """--project still wins: an explicit project on the CLI must not be
    second-guessed by what the evidence's own proposal asked for."""
    conn = _seed_core(tmp_path)
    evidence_id, _proposal_event_id = _seed_task_evidence_with_project_proposal(conn)
    db = str(tmp_path / "hosted-queue.sqlite")
    rc, payload = _run(
        capsys,
        db,
        [
            "hosted-approve",
            evidence_id,
            "--project",
            "other",
            "--approved-by",
            APPROVER,
        ],
    )
    assert rc == 0, payload
    entry = payload["approved"][0]
    assert entry["scope"]["scope_id"] == "project:other"
    assert entry["scope_source"] == "cli_project"
    conn.close()


def test_hosted_approve_refuses_cross_project_relabeling(tmp_path, capsys):
    conn = _seed_core(tmp_path)
    evidence_id = _seed_evidence(
        conn,
        body="Project-scoped evidence must stay in its existing project.",
    )
    db = str(tmp_path / "hosted-queue.sqlite")
    before = _event_count(conn)

    rc, payload = _run(
        capsys,
        db,
        [
            "hosted-approve",
            evidence_id,
            "--project",
            "other-project",
            "--approved-by",
            APPROVER,
        ],
    )

    assert rc == 2
    assert payload["status"] == "blocked"
    assert payload["refused"][0]["reason"] == "project_scope_mismatch"
    assert _event_count(conn) == before
    assert conn.execute("SELECT COUNT(*) FROM current_beliefs").fetchone()[0] == 0
    conn.close()


def test_hosted_approve_dry_run_plans_the_proposals_requested_project(tmp_path, capsys):
    """--dry-run plans the requested project and writes nothing: the plan must
    name where the approval would land, not the evidence row's own scope."""
    conn = _seed_core(tmp_path)
    evidence_id, _proposal_event_id = _seed_task_evidence_with_project_proposal(conn)
    db = str(tmp_path / "hosted-queue.sqlite")
    before = _event_count(conn)
    beliefs_before = conn.execute("SELECT COUNT(*) FROM current_beliefs").fetchone()[0]
    rc, payload = _run(
        capsys,
        db,
        ["hosted-approve", evidence_id, "--approved-by", APPROVER, "--dry-run"],
    )
    assert rc == 0
    assert payload["status"] == "planned"
    entry = payload["approved"][0]
    assert entry["status"] == "planned"
    assert entry["scope"]["scope_id"] == "project:coframe"
    assert entry["scope_source"] == "requested_by_proposal"
    assert _event_count(conn) == before
    assert conn.execute("SELECT COUNT(*) FROM current_beliefs").fetchone()[0] == beliefs_before
    conn.close()


def test_hosted_approve_refuses_confidential_with_no_belief_written(tmp_path, capsys):
    conn = _seed_core(tmp_path)
    confidential = _seed_evidence(
        conn,
        body="Client billing detail that must stay on this machine.",
        scope=_confidential_scope(),
    )
    conn.commit()
    db = str(tmp_path / "hosted-queue.sqlite")
    before = _event_count(conn)
    rc, payload = _run(
        capsys,
        db,
        ["hosted-approve", confidential, "--approved-by", APPROVER],
    )
    assert rc == 2
    assert payload["status"] == "blocked"
    refusal = payload["refused"][0]
    assert refusal["reason"] == "visibility_confidential"
    assert payload["approved"] == []
    # Typed refusal wrote nothing at all.
    assert _event_count(conn) == before
    beliefs = conn.execute("SELECT COUNT(*) FROM current_beliefs").fetchone()[0]
    assert beliefs == 0
    conn.close()


def test_hosted_approve_refuses_secret_leak_body(tmp_path, capsys):
    conn = _seed_core(tmp_path)
    leaky = _seed_evidence(
        conn,
        body=(
            "Use " + "api_" + "key=" + "«redacted:" + "sk-…» for the deployment."
        ),
    )
    conn.commit()
    db = str(tmp_path / "hosted-queue.sqlite")
    rc, payload = _run(
        capsys,
        db,
        ["hosted-approve", leaky, "--approved-by", APPROVER],
    )
    assert rc == 2
    refusal = payload["refused"][0]
    assert refusal["reason"] == "secret_leak_body"
    conn.close()


def test_hosted_approve_requires_human_spelled_approver(tmp_path, capsys):
    conn = _seed_core(tmp_path)
    evidence_id = _seed_evidence(conn, body="Agent may not approve its own widening.")
    conn.commit()
    db = str(tmp_path / "hosted-queue.sqlite")
    rc, payload = _run(
        capsys,
        db,
        ["hosted-approve", evidence_id, "--approved-by", "agent:self"],
    )
    assert rc == 2
    assert payload["status"] == "blocked"
    assert payload["reason"] == "invalid_approved_by"
    assert _event_count(conn) == 1  # only the seeded evidence event; no proposal
    conn.close()


def test_hosted_approve_dry_run_writes_nothing(tmp_path, capsys):
    conn = _seed_core(tmp_path)
    evidence_id = _seed_evidence(conn, body="Dry-run eligibility fixture.")
    conn.commit()
    db = str(tmp_path / "hosted-queue.sqlite")
    before = _event_count(conn)
    beliefs_before = conn.execute("SELECT COUNT(*) FROM current_beliefs").fetchone()[0]
    rc, payload = _run(
        capsys,
        db,
        ["hosted-approve", evidence_id, "--approved-by", APPROVER, "--dry-run"],
    )
    assert rc == 0
    assert payload["status"] == "planned"
    assert payload["approved"][0]["status"] == "planned"
    assert _event_count(conn) == before
    assert conn.execute("SELECT COUNT(*) FROM current_beliefs").fetchone()[0] == beliefs_before
    conn.close()


def test_hosted_approve_all_from_queue_then_queue_drains(tmp_path, capsys):
    conn = _seed_core(tmp_path)
    first = _seed_evidence(conn, body="Queue drain fixture one.")
    second = _seed_evidence(conn, body="Queue drain fixture two.", kind="task_closeout_summary")
    conn.commit()
    db = str(tmp_path / "hosted-queue.sqlite")
    rc, payload = _run(
        capsys,
        db,
        [
            "hosted-approve",
            "--all-from-queue",
            "--project",
            "coframe",
            "--approved-by",
            APPROVER,
        ],
    )
    assert rc == 0, payload
    assert payload["status"] == "applied"
    assert {entry["evidence_id"] for entry in payload["approved"]} == {first, second}
    rc, queue = _run(capsys, db, ["hosted-queue", "--project", "coframe"])
    assert rc == 0
    assert queue["count"] == 0
    conn.close()


def test_bulk_approval_preserves_supports_when_evidence_converges(tmp_path, capsys):
    conn = _seed_core(tmp_path)
    body = "Two independent observations support the same hosted claim."
    first = _seed_evidence(conn, body=body, kind="observation")
    second = _seed_evidence(conn, body=body, kind="task_closeout_summary")
    db = str(tmp_path / "hosted-queue.sqlite")

    rc, payload = _run(
        capsys,
        db,
        ["hosted-approve", "--all-from-queue", "--approved-by", APPROVER],
    )

    assert rc == 0, payload
    assert {entry["evidence_id"] for entry in payload["approved"]} == {first, second}
    belief_ids = {entry["belief_id"] for entry in payload["approved"]}
    assert len(belief_ids) == 1
    belief_id = belief_ids.pop()
    supports = {
        row["evidence_id"]
        for row in conn.execute(
            "SELECT evidence_id FROM belief_evidence WHERE belief_id=?", (belief_id,)
        )
    }
    assert supports == {first, second}
    rc, queue = _run(capsys, db, ["hosted-queue"])
    assert rc == 0
    assert queue["count"] == 0
    conn.close()


def test_hosted_approval_verbs_are_cli_only(tmp_path):
    conn = _seed_core(tmp_path)
    names = {
        tool["name"]
        for tool in (
            tool_list(profile="runtime", core_v1=True)
            + tool_list(profile="admin", core_v1=True)
        )
    }
    assert "hosted-queue" not in names
    assert "hosted-approve" not in names
    assert is_core_v1(conn)
    conn.close()
