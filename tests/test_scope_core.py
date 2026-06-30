import json
from pathlib import Path

from ocbrain.db import connect, init_db
from ocbrain.dream import dream
from ocbrain.egress import egress_preview
from ocbrain.events import (
    approval_packet,
    decide_compilation,
    event_core_digest,
    get_current_belief,
    list_compilation_proposals,
    propose_compilation,
    rebuild_projection,
    record_correction,
    record_evidence,
    record_tombstone,
)
from ocbrain.mcp import handle_request
from ocbrain.retrieve import retrieve
from ocbrain.scope import ScopeContext, ScopeTag, global_scope, resolve_write_scope
from ocbrain.teacher import hosted_teacher_request


def seeded_scoped_core(tmp_path: Path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    record_evidence(
        conn,
        body="Bountiful uses React 19, Express, and Neon for the neighbor garden app.",
        context=ScopeContext(project="bountiful"),
        writer="codex",
    )
    record_evidence(
        conn,
        body="Pelican options strategy references theta decay and position sizing.",
        scope=ScopeTag(
            "personal_finance",
            "personal_finance:pelican",
            visibility="confidential",
            egress_policy="local_only",
        ),
        writer="claude",
    )
    record_evidence(
        conn,
        body=(
            "Cormorant Bihua lane registry contains client notes and "
            "api_key=sk-123456789012345678901234."
        ),
        scope=ScopeTag(
            "client",
            "client:bihua",
            visibility="confidential",
            egress_policy="local_only",
        ),
        writer="openclaw",
    )
    record_evidence(
        conn,
        body="Never weaken rules to clear red; fix the real failing invariant.",
        scope=global_scope(),
        writer="human:jonathan",
    )

    bountiful = propose_compilation(
        conn,
        belief_id="belief:bountiful-stack",
        body="Bountiful stack is React 19, Express, and Neon.",
        evidence_ids=["evd:bountiful-stack"],
        scope=ScopeTag("project", "project:bountiful", egress_policy="hosted_ok"),
        confidence=0.85,
    )
    pelican = propose_compilation(
        conn,
        belief_id="belief:pelican-options",
        body="Pelican has options-trading beliefs about theta decay.",
        evidence_ids=["evd:pelican-options"],
        scope=ScopeTag(
            "personal_finance",
            "personal_finance:pelican",
            visibility="confidential",
            egress_policy="local_only",
        ),
        confidence=0.8,
    )
    cormorant = propose_compilation(
        conn,
        belief_id="belief:cormorant-lanes",
        body="Cormorant Bihua has a confidential lane registry.",
        evidence_ids=["evd:cormorant-lanes"],
        scope=ScopeTag(
            "client",
            "client:bihua",
            visibility="confidential",
            egress_policy="local_only",
        ),
        confidence=0.8,
    )
    doctrine = propose_compilation(
        conn,
        belief_id="belief:global-rules-red",
        body="Never weaken rules to clear red; fix the invariant.",
        evidence_ids=["evd:global-rules"],
        scope=global_scope(),
        confidence=0.9,
    )
    for proposal in (bountiful, pelican, cormorant, doctrine):
        decide_compilation(conn, proposal_event_id=proposal, decision="approve")
    conn.commit()
    return conn


def test_scoped_retrieval_excludes_foreign_project_but_keeps_global(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)

    payload = retrieve(
        conn,
        "rules red Bountiful React options Cormorant",
        context=ScopeContext(project="bountiful"),
        limit=10,
    )
    ids = {item["belief_id"] for item in payload["items"]}

    assert "belief:bountiful-stack" in ids
    assert "belief:global-rules-red" in ids
    assert "belief:pelican-options" not in ids
    assert "belief:cormorant-lanes" not in ids
    assert {item["belief_id"] for item in payload["applied_global"]} == {
        "belief:global-rules-red"
    }


def test_cross_scope_is_explicit_and_still_blocks_confidential(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)

    payload = retrieve(
        conn,
        "options theta Cormorant",
        context=ScopeContext(project="bountiful"),
        limit=10,
        cross_scope=True,
    )
    ids = {item["belief_id"] for item in payload["items"]}

    assert "belief:pelican-options" not in ids
    assert "belief:cormorant-lanes" not in ids
    assert payload["cross_scope"] is True


def test_contradiction_ranking_respects_scope_boundaries(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)
    positive = propose_compilation(
        conn,
        belief_id="belief:bountiful-fly",
        body="Bountiful deploys on Fly for production.",
        evidence_ids=["evd:bountiful-fly"],
        scope=ScopeTag("project", "project:bountiful"),
        confidence=0.9,
    )
    negative = propose_compilation(
        conn,
        belief_id="belief:bountiful-not-fly",
        body="Bountiful does not deploy on Fly for production.",
        evidence_ids=["evd:bountiful-not-fly"],
        scope=global_scope(),
        confidence=0.8,
    )
    confidential = propose_compilation(
        conn,
        belief_id="belief:pelican-not-fly",
        body="Pelican does not deploy on Fly for production.",
        evidence_ids=["evd:pelican-not-fly"],
        scope=ScopeTag(
            "personal_finance",
            "personal_finance:pelican",
            visibility="confidential",
        ),
        confidence=0.8,
    )
    for proposal in (positive, negative, confidential):
        decide_compilation(conn, proposal_event_id=proposal, decision="approve")

    payload = retrieve(
        conn,
        "Bountiful deploys Fly production",
        context=ScopeContext(project="bountiful"),
    )
    pairs = {
        frozenset({item["belief_id"], item["other_belief_id"]})
        for item in payload["contradictions"]
    }

    assert frozenset({"belief:bountiful-fly", "belief:bountiful-not-fly"}) in pairs
    assert all(
        "personal_finance:pelican" not in json.dumps(item)
        for item in payload["contradictions"]
    )
    assert payload["contradictions"][0]["score"] >= payload["contradictions"][-1]["score"]


def test_hosted_egress_excludes_confidential_and_redacts_secrets(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)

    payload = egress_preview(
        conn,
        context=ScopeContext(project="bountiful"),
        target="hosted_teacher",
        record=True,
    )
    included_text = json.dumps(payload["included"])
    rejected_ids = {item["scope"]["scope_id"] for item in payload["rejected"]}

    assert "client:bihua" in rejected_ids
    assert "personal_finance:pelican" in rejected_ids
    assert "sk-123456789012345678901234" not in included_text
    assert payload["audit_id"].startswith("egress_")
    assert conn.execute("SELECT COUNT(*) FROM egress_audits").fetchone()[0] == 1


def test_hosted_teacher_request_is_local_approval_package(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)
    record_evidence(
        conn,
        body="Bountiful hosted teacher eligible public deployment note.",
        scope=ScopeTag("project", "project:bountiful", egress_policy="hosted_ok"),
    )

    payload = hosted_teacher_request(
        conn,
        context=ScopeContext(project="bountiful"),
        query="Bountiful",
    )
    encoded = json.dumps(payload)
    included_scopes = {item["scope"]["scope_id"] for item in payload["egress"]["included"]}
    rejected_scopes = {item["scope"]["scope_id"] for item in payload["egress"]["rejected"]}

    assert payload["request_id"].startswith("teacher_request_")
    assert payload["dispatch_state"] == "approval_required"
    assert payload["call_performed"] is False
    assert payload["approval"]["required_approvals"] == [
        "hosted_teacher_calls",
        "hosted_egress",
    ]
    assert "project:bountiful" in included_scopes
    assert "client:bihua" in rejected_scopes
    assert "sk-123456789012345678901234" not in encoded
    assert payload["request"]["input"]["response_schema"]["required"] == ["compilations"]
    assert conn.execute("SELECT COUNT(*) FROM egress_audits").fetchone()[0] == 1


def test_unscoped_ingest_is_quarantined_not_global(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    scope = resolve_write_scope(ScopeContext())
    event_id = record_evidence(conn, body="An unscoped observation", scope=scope)
    body_json = conn.execute("SELECT body_json FROM brain_events WHERE id = ?", (event_id,))
    body = json.loads(body_json.fetchone()["body_json"])

    assert body["scope"]["scope_type"] == "legacy_unscoped"
    assert body["scope"]["egress_policy"] == "local_only"
    assert body["scope"]["provenance"] == "quarantined"


def test_pending_compilation_is_invisible_until_decided(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    propose_compilation(
        conn,
        belief_id="belief:pending",
        body="Pending beliefs must not be retrieved.",
        evidence_ids=["evd:pending"],
        scope=global_scope(),
        confidence=0.9,
    )
    rebuild_projection(conn)

    payload = retrieve(conn, "Pending beliefs", context=ScopeContext(project="bountiful"))

    assert payload["items"] == []


def test_compilation_requires_evidence(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    try:
        propose_compilation(
            conn,
            belief_id="belief:no-evidence",
            body="Beliefs without evidence must be rejected.",
            evidence_ids=[],
            scope=global_scope(),
            confidence=0.9,
        )
    except ValueError as exc:
        assert "at least one evidence id" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected no-evidence compilation to fail")


def test_correction_survives_projection_rebuild(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)

    record_correction(
        conn,
        target_layer="belief",
        target_id="belief:bountiful-stack",
        op="edit",
        body="Bountiful stack is React 19, Express, Neon, and Fly.",
    )
    rebuild_projection(conn)
    row = conn.execute(
        "SELECT body FROM current_beliefs WHERE belief_id = 'belief:bountiful-stack'"
    ).fetchone()

    assert row["body"] == "Bountiful stack is React 19, Express, Neon, and Fly."


def test_hard_correction_blocks_rederived_belief(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)

    record_correction(
        conn,
        target_layer="belief",
        target_id="belief:bountiful-stack",
        op="mark_wrong",
        body="Do not re-derive this stack belief.",
        hard=True,
    )

    try:
        propose_compilation(
            conn,
            belief_id="belief:bountiful-stack",
            body="Bountiful stack is React 19, Express, and Neon.",
            evidence_ids=["evd:bountiful-stack"],
            scope=ScopeTag("project", "project:bountiful"),
            confidence=0.9,
        )
    except PermissionError as exc:
        assert "hard correction" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected hard-corrected belief to be blocked")


def test_forget_survives_projection_rebuild(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)

    record_tombstone(
        conn,
        target="belief:bountiful-stack",
        mode="soft",
        reason="forget request",
    )
    rebuild_projection(conn)
    payload = retrieve(
        conn,
        "Bountiful React",
        context=ScopeContext(project="bountiful"),
    )
    row = get_current_belief(conn, "belief:bountiful-stack")

    assert "belief:bountiful-stack" not in {item["belief_id"] for item in payload["items"]}
    assert row is not None
    assert row["status"] == "tombstoned"


def test_shred_tombstone_redacts_served_projection_body(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)

    record_tombstone(
        conn,
        target="belief:bountiful-stack",
        mode="shred",
        reason="crypto shred request",
    )
    rebuild_projection(conn)
    row = get_current_belief(conn, "belief:bountiful-stack")
    payload = retrieve(
        conn,
        "Bountiful React Express Neon",
        context=ScopeContext(project="bountiful"),
    )
    tombstone_body = json.loads(
        conn.execute(
            """
            SELECT body_json
            FROM brain_events
            WHERE kind = 'tombstone_recorded'
            ORDER BY rowid DESC
            LIMIT 1
            """
        ).fetchone()["body_json"]
    )

    assert row is not None
    assert row["status"] == "tombstoned"
    assert row["body"] == "[shredded by tombstone]"
    assert row["evidence_ids"] == []
    assert tombstone_body["target_hash"]
    assert tombstone_body["serving_policy"] == "redact_projection_body_and_evidence_ids"
    assert "belief:bountiful-stack" not in {item["belief_id"] for item in payload["items"]}


def test_get_current_belief_includes_web_provenance_without_source_body(
    tmp_path: Path,
) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    evidence_event = record_evidence(
        conn,
        body="Bountiful public health endpoint returned status ok.",
        scope=ScopeTag("project", "project:bountiful", egress_policy="hosted_ok"),
        artifact_ref="https://bountiful.garden/api/health",
    )
    evidence_body = json.loads(
        conn.execute("SELECT body_json FROM brain_events WHERE id = ?", (evidence_event,))
        .fetchone()["body_json"]
    )
    proposal = propose_compilation(
        conn,
        belief_id="belief:bountiful-health",
        body="Bountiful public health endpoint is healthy.",
        evidence_ids=[evidence_body["evidence_id"]],
        scope=ScopeTag("project", "project:bountiful"),
        confidence=0.9,
    )
    decide_compilation(conn, proposal_event_id=proposal, decision="approve")

    row = get_current_belief(conn, "belief:bountiful-health")

    assert row is not None
    assert row["evidence_provenance"] == [
        {
            "evidence_id": evidence_body["evidence_id"],
            "event_id": evidence_event,
            "ts": row["evidence_provenance"][0]["ts"],
            "kind": "observation",
            "source": "https://bountiful.garden/api/health",
            "source_kind": "web",
            "body_hash": row["evidence_provenance"][0]["body_hash"],
            "scope": {
                "scope_type": "project",
                "scope_id": "project:bountiful",
                "visibility": "internal",
                "egress_policy": "hosted_ok",
                "provenance": "explicit",
            },
        }
    ]
    assert "status ok" not in json.dumps(row["evidence_provenance"])


def test_fold_to_timestamp_reproduces_prior_view_without_mutating_live_view(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)
    before_ts = conn.execute(
        """
        SELECT ts
        FROM brain_events
        WHERE id = (
          SELECT approved_event_id
          FROM current_beliefs
          WHERE belief_id = 'belief:bountiful-stack'
        )
        """
    ).fetchone()["ts"]

    record_correction(
        conn,
        target_layer="belief",
        target_id="belief:bountiful-stack",
        op="edit",
        body="Bountiful stack is React 19, Express, Neon, and Fly.",
    )

    prior = retrieve(
        conn,
        "Bountiful React",
        context=ScopeContext(project="bountiful"),
        at_ts=before_ts,
    )
    current = retrieve(
        conn,
        "Bountiful Fly",
        context=ScopeContext(project="bountiful"),
    )

    assert prior["items"][0]["body"] == "Bountiful stack is React 19, Express, and Neon."
    assert current["items"][0]["body"] == "Bountiful stack is React 19, Express, Neon, and Fly."
    live_row = get_current_belief(conn, "belief:bountiful-stack")
    assert live_row is not None
    assert live_row["body"] == "Bountiful stack is React 19, Express, Neon, and Fly."


def test_event_tables_are_insert_only_for_connector_operations(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    statements: list[str] = []
    conn.set_trace_callback(statements.append)

    evidence_event = record_evidence(
        conn,
        body="Event tables are append-only.",
        context=ScopeContext(project="ocbrain"),
    )
    proposal = propose_compilation(
        conn,
        belief_id="belief:append-only",
        body="Event tables are append-only.",
        evidence_ids=[evidence_event],
        scope=ScopeTag("project", "project:ocbrain"),
        confidence=0.9,
    )
    decide_compilation(conn, proposal_event_id=proposal, decision="approve")
    record_correction(
        conn,
        target_layer="belief",
        target_id="belief:append-only",
        op="pin",
        body=None,
    )
    record_tombstone(conn, target="belief:append-only", mode="soft", reason="test")

    forbidden = [
        sql
        for sql in statements
        if sql.strip().upper().startswith(("UPDATE BRAIN_EVENTS", "DELETE FROM BRAIN_EVENTS"))
    ]
    assert forbidden == []


def test_dream_writes_pending_scoped_proposals_not_current_beliefs(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)

    result = dream(
        conn,
        context=ScopeContext(project="bountiful"),
        target="local_model",
        record_egress=True,
    )
    rebuild_projection(conn)
    payload = retrieve(
        conn,
        "Scoped consolidation",
        context=ScopeContext(project="bountiful"),
    )

    assert result["summary"]["proposals"] >= 2
    assert result["egress"]["audit_id"].startswith("egress_")
    assert {item["reward_band"] for item in list_compilation_proposals(conn)} == {
        "moderate"
    }
    assert payload["items"] == []


def test_dream_honors_hard_corrections_as_conflicts(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)
    first = dream(conn, context=ScopeContext(project="bountiful"))
    blocked_belief_id = first["proposed"][0]["belief_id"]

    record_correction(
        conn,
        target_layer="belief",
        target_id=blocked_belief_id,
        op="mark_wrong",
        body="Do not re-derive this dream proposal.",
        hard=True,
    )
    second = dream(conn, context=ScopeContext(project="bountiful"))

    assert blocked_belief_id in {item["belief_id"] for item in second["conflicts"]}


def test_event_gate_lists_and_decides_pending_proposals(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)
    result = dream(conn, context=ScopeContext(project="bountiful"))
    proposal_id = result["proposed"][0]["proposal_event_id"]

    pending = list_compilation_proposals(conn, context=ScopeContext(project="bountiful"))
    packet = approval_packet(
        pending,
        context=ScopeContext(project="bountiful"),
        cli_prefix=["ocbrain", "--db", "data/ocbrain.sqlite"],
    )
    assert proposal_id in {item["proposal_event_id"] for item in pending}
    assert {item["status"] for item in pending} == {"pending"}
    assert packet["channel"] == "telegram"
    assert packet["send_performed"] is False
    assert f"/ocbrain_gate approve {proposal_id}" in packet["text"]
    assert packet["items"][0]["actions"]["approve"]["mcp_tool"] == "brain.feedback"
    assert packet["items"][0]["actions"]["reject"]["cli_argv"] == [
        "ocbrain",
        "--db",
        "data/ocbrain.sqlite",
        "event-decide",
        "--proposal-event-id",
        proposal_id,
        "--decision",
        "reject",
        "--actor",
        "human:jonathan",
    ]

    decision_id = decide_compilation(
        conn,
        proposal_event_id=proposal_id,
        decision="approve",
        actor="human:jonathan",
    )
    decided = list_compilation_proposals(
        conn,
        context=ScopeContext(project="bountiful"),
        include_decided=True,
    )
    active = retrieve(
        conn,
        "Scoped consolidation",
        context=ScopeContext(project="bountiful"),
    )

    assert decision_id.startswith("evt_")
    assert proposal_id not in {
        item["proposal_event_id"]
        for item in list_compilation_proposals(conn, context=ScopeContext(project="bountiful"))
    }
    assert proposal_id in {item["proposal_event_id"] for item in decided}
    assert result["proposed"][0]["belief_id"] in {item["belief_id"] for item in active["items"]}


def test_event_core_digest_reports_pending_and_current_by_scope(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)
    record_evidence(
        conn,
        body="Codex session wrote useful Bountiful evidence.",
        context=ScopeContext(project="bountiful"),
        writer="codex",
        session_id="codex-1",
    )
    dream_result = dream(conn, context=ScopeContext(project="bountiful"))

    payload = event_core_digest(conn, context=ScopeContext(project="bountiful"))

    assert payload["summary"]["pending_compilations"] >= len(dream_result["proposed"])
    assert "evidence_recorded" in payload["event_counts"]
    assert "compilation_proposed" in payload["event_counts"]
    assert any(item["belief_id"] == "belief:bountiful-stack" for item in payload["current_beliefs"])
    assert any(item["writer"] == "codex" for item in payload["runtime_health"])
    assert any("codex-1" in item["sessions"] for item in payload["runtime_health"])
    assert all("last_useful_write_at" in item for item in payload["runtime_health"])
    assert all(
        item["scope"]["scope_id"] in {"project:bountiful", "global:doctrine"}
        for item in payload["pending_compilations"]
    )


def test_event_core_digest_exposes_falsifiable_quiet_loop_surface(
    tmp_path: Path,
) -> None:
    conn = seeded_scoped_core(tmp_path)
    quiet = event_core_digest(conn, context=ScopeContext(project="bountiful"))
    dream(conn, context=ScopeContext(project="bountiful"))
    attention = event_core_digest(conn, context=ScopeContext(project="bountiful"))

    assert quiet["quiet_loop"]["state"] == "quiet"
    assert all(check["passed"] for check in quiet["quiet_loop"]["falsifiable_checks"])
    assert attention["quiet_loop"]["state"] == "attention"
    assert any(
        check["name"] == "no_pending_compilations" and not check["passed"]
        for check in attention["quiet_loop"]["falsifiable_checks"]
    )


def test_mcp_preview_uses_same_scoped_retrieval_payload(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)
    expected = retrieve(
        conn,
        "rules red Bountiful",
        context=ScopeContext(project="bountiful"),
        limit=10,
    )

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "brain.preview",
                "arguments": {
                    "query": "rules red Bountiful",
                    "limit": 10,
                    "context": {"project": "bountiful"},
                },
            },
        },
    )
    payload = json.loads(response["result"]["content"][0]["text"])

    assert payload["items"] == expected["items"]
    assert payload["applied_global"] == expected["applied_global"]


def test_mcp_connector_write_tools_are_gated_and_append_events(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)

    listed = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        },
        allow_writes=True,
    )
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert {"brain.ingest", "brain.forget"} <= names

    blocked = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "brain.ingest",
                "arguments": {
                    "body": "Runtime health is last useful write, not a green dot.",
                    "context": {"project": "ocbrain"},
                },
            },
        },
    )
    assert "error" in blocked

    ingested = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "brain.ingest",
                "arguments": {
                    "body": "Runtime health is last useful write, not a green dot.",
                    "context": {"project": "ocbrain"},
                    "writer": "codex",
                },
            },
        },
        allow_writes=True,
    )
    payload = json.loads(ingested["result"]["content"][0]["text"])
    assert payload["kind"] == "evidence_recorded"

    corrected = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "brain.feedback",
                "arguments": {
                    "layer": "belief",
                    "target": "belief:bountiful-stack",
                    "op": "edit",
                    "body": "Bountiful stack is React 19, Express, Neon, and Fly.",
                    "hard": True,
                },
            },
        },
        allow_writes=True,
    )
    correction_payload = json.loads(corrected["result"]["content"][0]["text"])
    assert correction_payload["kind"] == "correction_recorded"

    fetched = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "brain.get",
                "arguments": {"id": "belief:bountiful-stack"},
            },
        },
    )
    belief = json.loads(fetched["result"]["content"][0]["text"])
    assert belief["object_kind"] == "belief"
    assert belief["body"] == "Bountiful stack is React 19, Express, Neon, and Fly."


def test_mcp_event_gate_lists_and_decides_proposals(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)
    dream_result = dream(conn, context=ScopeContext(project="bountiful"))
    proposal_id = dream_result["proposed"][0]["proposal_event_id"]

    listed = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "brain.proposals",
                "arguments": {
                    "context": {"project": "bountiful"},
                    "approval_packet": True,
                },
            },
        },
        allow_writes=True,
    )
    listed_payload = json.loads(listed["result"]["content"][0]["text"])
    assert proposal_id in {
        item["proposal_event_id"] for item in listed_payload["proposals"]
    }
    assert listed_payload["approval_packet"]["send_performed"] is False
    assert f"/ocbrain_gate reject {proposal_id}" in listed_payload["approval_packet"]["text"]

    decided = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "brain.feedback",
                "arguments": {
                    "proposal_event_id": proposal_id,
                    "decision": "approve",
                    "actor": "human:jonathan",
                },
            },
        },
        allow_writes=True,
    )
    decision_payload = json.loads(decided["result"]["content"][0]["text"])
    assert decision_payload["kind"] == "compilation_decided"

    digest_response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "brain.digest",
                "arguments": {
                    "event_core": True,
                    "context": {"project": "bountiful"},
                },
            },
        },
    )
    digest_payload = json.loads(digest_response["result"]["content"][0]["text"])
    assert "event_core" in digest_payload
    assert any(
        item["belief_id"] == dream_result["proposed"][0]["belief_id"]
        for item in digest_payload["event_core"]["current_beliefs"]
    )


def test_mcp_teacher_request_packages_without_dispatch(tmp_path: Path) -> None:
    conn = seeded_scoped_core(tmp_path)
    record_evidence(
        conn,
        body="Bountiful hosted teacher eligible public deployment note.",
        scope=ScopeTag("project", "project:bountiful", egress_policy="hosted_ok"),
    )

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "brain.teacher_request",
                "arguments": {
                    "query": "Bountiful",
                    "context": {"project": "bountiful"},
                },
            },
        },
    )
    payload = json.loads(response["result"]["content"][0]["text"])

    assert payload["call_performed"] is False
    assert payload["dispatch_state"] == "approval_required"
    assert payload["summary"]["audit_id"].startswith("egress_")
