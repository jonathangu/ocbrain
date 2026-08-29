"""The curator's contradiction cascade: what happens when a fact changes.

Before this, a scheduled curator that found a claim on a key it already served
overwrote the body in place and replaced the confidence with whatever the hosted
model returned. Nothing marked the fact as changed, nothing recorded what it used
to say, and the run that did it looked exactly like a run that had compiled
something new. On an hourly job nobody watches, that was the single largest
source of corpus pollution.

These tests pin the four outcomes the cascade can reach -- write, supersede,
coexist, defer -- and the two guards that keep it honest: a claim cannot retire a
better-evidenced belief just by being newer, and stale evidence cannot overwrite a
fresher correction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ocbrain.curator
from ocbrain.core_v1 import (
    get_core_v1_belief,
    init_core_v1,
    record_core_v1_evidence,
)
from ocbrain.curator import (
    ELIGIBLE_KINDS,
    apply_claims,
    resolve_conflicts_with,
    select_evidence,
    validate_claims,
)
from ocbrain.db import connect
from ocbrain.mcp_v1 import (
    build_context_v1,
    correct_v1,
    decide_proposal_v1,
    pending_supersede_count,
)
from ocbrain.scope import ScopeContext, ScopeTag

PROJECT = "test"
PROJECT_SCOPE = ScopeTag(
    "project",
    f"project:{PROJECT}",
    visibility="internal",
    egress_policy="hosted_ok",
    provenance="test",
)


def _core(tmp_path: Path):
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    return conn


def _claim(
    key: str,
    body: str,
    *,
    confidence: float = 0.9,
    evidence_ids: list[str] | None = None,
    conflicts_with: list[dict[str, str]] | None = None,
) -> dict:
    claim = {
        "key": key,
        "title": key.replace("-", " "),
        "body": body,
        "category": "system",
        "lifecycle": "durable",
        "confidence": confidence,
        "evidence_ids": list(evidence_ids or []),
    }
    if conflicts_with is not None:
        claim["conflicts_with"] = conflicts_with
    return claim


def _evidence(conn, body: str, *, kind: str = "audit_finding") -> str:
    evidence_id, _event = record_core_v1_evidence(
        conn, body=body, kind=kind, scope=PROJECT_SCOPE, writer="test"
    )
    conn.commit()
    return evidence_id


def _serving(conn) -> dict[str, str]:
    return {
        str(row["belief_id"]): str(row["body"])
        for row in conn.execute(
            "SELECT belief_id, body FROM current_beliefs WHERE serve=1 AND status='current'"
        )
    }


def _corrections(conn, op: str) -> list[dict]:
    return [
        json.loads(row[0])
        for row in conn.execute(
            "SELECT body_json FROM brain_events WHERE kind='correction_recorded' "
            "AND json_extract(body_json, '$.op')=? ORDER BY event_seq",
            (op,),
        )
    ]


def _stub_neighbors(monkeypatch, neighbors, unavailable=None) -> None:
    """Answer the vector pre-filter without a local embedding server."""
    monkeypatch.setattr(
        ocbrain.curator,
        "semantic_neighbors",
        lambda conn, body, *, candidate_ids=None, limit=100: (
            [
                item
                for item in neighbors
                if candidate_ids is None or item["belief_id"] in set(candidate_ids)
            ][:limit],
            unavailable,
        ),
    )


def test_key_body_change_routes_through_supersession(tmp_path: Path) -> None:
    """A recompiled key whose statement changed is a correction, not an edit."""
    conn = _core(tmp_path)
    first = apply_claims(
        conn,
        [_claim("research-vm-live", "The live analysis VM is asa1.")],
        model="test",
        project=PROJECT,
    )
    original = first["applied"][0]

    second = apply_claims(
        conn,
        [_claim("research-vm-live", "The live analysis VM is asa2; asa1 is terminated.")],
        model="test",
        project=PROJECT,
    )

    # The claim did not land on the old belief. It stood a successor up and
    # closed the old copy's era.
    assert second["applied"] == []
    assert second["superseded"] and second["superseded"] != [original]
    successor_id = second["superseded"][0]

    retired = get_core_v1_belief(conn, original)
    assert retired["status"] == "retracted"
    assert not retired["serve"]
    assert retired["attributes"]["superseded_by"] == successor_id
    assert retired["attributes"]["valid_until"]

    successor = get_core_v1_belief(conn, successor_id)
    assert successor["serve"]
    assert successor["body"] == "The live analysis VM is asa2; asa1 is terminated."
    # The key travels with the fact, so the next cycle recognises it.
    assert successor["attributes"]["key"] == "research-vm-live"
    assert successor["attributes"]["supersedes"] == original
    assert successor["attributes"]["valid_from"]
    # Recency is not authority: the model's own number never lands unexamined.
    # A same-key curator refresh holds the fact's standing rather than taking
    # the 0.7 contested-correction ceiling, so it neither gains nor loses.
    assert successor["confidence"] == get_core_v1_belief(conn, original)["confidence"] == 0.9

    # Exactly one paired correction, written by the decision, not by the curator.
    supersessions = _corrections(conn, "supersede")
    assert len(supersessions) == 1
    assert supersessions[0]["target_id"] == original
    assert supersessions[0]["successor_id"] == successor_id
    assert list(_serving(conn)) == [successor_id]
    conn.close()


def test_unchanged_body_is_still_a_cheap_noop(tmp_path: Path) -> None:
    """Re-compiling an unchanged fact must cost nothing and write nothing."""
    conn = _core(tmp_path)
    claim = _claim("research-vm-live", "The live analysis VM is asa2; asa1 is terminated.")
    first = apply_claims(conn, [claim], model="test", project=PROJECT)
    belief_id = first["applied"][0]
    before = conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0]

    second = apply_claims(conn, [dict(claim)], model="test", project=PROJECT)

    assert second["unchanged"] == [belief_id]
    assert second["applied"] == []
    assert second["superseded"] == []
    assert second["deferred"] == []
    assert conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0] == before
    conn.close()


def test_new_key_below_cosine_floor_mints_without_cascade(
    tmp_path: Path, monkeypatch
) -> None:
    """The cheap stage ends the cascade for a claim about something else.

    Almost every claim is about a subject nothing else in the corpus mentions.
    Paying for a subsumption test, let alone an adjudication, on all of them to
    find the handful that matter is what makes a cascade too expensive to run.
    """
    conn = _core(tmp_path)
    standing = apply_claims(
        conn,
        [_claim("research-vm-live", "The live analysis VM is asa2; asa1 is terminated.")],
        model="test",
        project=PROJECT,
    )["applied"][0]

    # Near the floor but under it: related enough to rank first, not enough to
    # be treated as a claim about the same thing.
    _stub_neighbors(monkeypatch, [{"belief_id": standing, "similarity": 0.59}])
    result = apply_claims(
        conn,
        [_claim("clickhouse-access", "Production ClickHouse credentials are read-only.")],
        model="test",
        project=PROJECT,
    )

    assert len(result["applied"]) == 1
    assert result["superseded"] == []
    assert result["deferred"] == []
    assert result["coexist_marked"] == []
    assert standing in _serving(conn)
    conn.close()


def test_above_the_cosine_floor_a_non_restatement_supersedes(
    tmp_path: Path, monkeypatch
) -> None:
    """The stage the floor exists to protect still fires when it should."""
    conn = _core(tmp_path)
    standing = apply_claims(
        conn,
        [_claim("research-vm-live", "The live analysis VM is asa1.")],
        model="test",
        project=PROJECT,
    )["applied"][0]

    _stub_neighbors(monkeypatch, [{"belief_id": standing, "similarity": 0.91}])
    result = apply_claims(
        conn,
        [
            _claim(
                "applied-science-box",
                "Applied-science work now runs on a replacement host; the earlier "
                "primary was decommissioned in August.",
            )
        ],
        model="test",
        project=PROJECT,
    )

    assert result["applied"] == []
    assert len(result["superseded"]) == 1
    assert get_core_v1_belief(conn, standing)["status"] == "retracted"
    conn.close()


def test_conflicts_with_index_validated_against_advisory_list() -> None:
    """An index the model invented names nothing, so it does nothing."""
    advisory = [
        {"belief_id": "belief:one", "key": "one", "body": "first"},
        {"belief_id": "belief:two", "key": "two", "body": "second"},
    ]

    assert resolve_conflicts_with([{"index": 1, "resolution": "coexist"}], advisory) == [
        {"belief_id": "belief:two", "resolution": "coexist"}
    ]
    # Out of range, negative, non-integer, unknown resolution, and an id the
    # model typed itself instead of selecting: none of them produce an action.
    for invented in (
        {"index": 7, "resolution": "supersede"},
        {"index": -1, "resolution": "supersede"},
        {"index": "1", "resolution": "supersede"},
        {"index": True, "resolution": "supersede"},
        {"index": 1, "resolution": "delete"},
        {"belief_id": "belief:two", "resolution": "supersede"},
    ):
        assert resolve_conflicts_with([invented], advisory) == [], invented
    # No advisory list means nothing was offered to select from.
    assert resolve_conflicts_with([{"index": 0, "resolution": "coexist"}], []) == []

    # And the whole way through validation: a claim survives its own bad index.
    response = {
        "beliefs": [
            {
                "key": "clickhouse-access",
                "title": "ClickHouse access",
                "body": "Production ClickHouse credentials are read-only for analysts.",
                "category": "system",
                "lifecycle": "durable",
                "confidence": 0.9,
                "supports": [{"evidence_id": "evd_1", "quote": "read-only for analysts"}],
                "conflicts_with": [{"index": 9, "resolution": "supersede"}],
            }
        ]
    }
    evidence = [
        {
            "evidence_id": "evd_1",
            "body": "Production ClickHouse credentials are read-only for analysts.",
        }
    ]
    claims, rejected = validate_claims(
        response, evidence=evidence, max_beliefs=4, existing=advisory
    )
    assert rejected == []
    assert claims[0]["conflicts_with"] == []


def test_coexist_writes_contradicts_on_both(tmp_path: Path) -> None:
    """Conflict preservation: both beliefs keep serving, and both say so."""
    conn = _core(tmp_path)
    standing = apply_claims(
        conn,
        [_claim("holdout-share", "The project holdout share is five percent.")],
        model="test",
        project=PROJECT,
    )["applied"][0]

    result = apply_claims(
        conn,
        [
            _claim(
                "experiment-holdout-share",
                "The experiment-level holdout share is ten percent.",
                conflicts_with=[{"belief_id": standing, "resolution": "coexist"}],
            )
        ],
        model="test",
        project=PROJECT,
    )
    minted = result["applied"][0]

    assert result["coexist_marked"] == [
        {"belief_id": minted, "other_belief_id": standing}
    ]
    assert result["superseded"] == []
    assert set(_serving(conn)) == {standing, minted}
    assert get_core_v1_belief(conn, minted)["attributes"]["contradicts"] == [standing]
    assert get_core_v1_belief(conn, standing)["attributes"]["contradicts"] == [minted]
    # Metadata only. An annotation must not touch what a belief says or how
    # confident the brain is in it.
    assert get_core_v1_belief(conn, standing)["body"] == (
        "The project holdout share is five percent."
    )
    assert len(_corrections(conn, "annotate")) == 2
    conn.close()


def test_declared_coexist_overrides_mechanical_supersede(
    tmp_path: Path, monkeypatch
) -> None:
    """The model's explicit coexist verdict outranks the cosine heuristic."""
    conn = _core(tmp_path)
    standing = apply_claims(
        conn,
        [_claim("holdout-share", "The project holdout share is five percent.")],
        model="test",
        project=PROJECT,
    )["applied"][0]
    _stub_neighbors(monkeypatch, [{"belief_id": standing, "similarity": 0.91}])

    result = apply_claims(
        conn,
        [
            _claim(
                "experiment-holdout-share",
                "The experiment-level holdout share is ten percent.",
                conflicts_with=[{"belief_id": standing, "resolution": "coexist"}],
            )
        ],
        model="test",
        project=PROJECT,
    )

    assert result["superseded"] == []
    assert len(result["applied"]) == 1
    minted = result["applied"][0]
    assert set(_serving(conn)) == {standing, minted}
    assert result["coexist_marked"] == [
        {"belief_id": minted, "other_belief_id": standing}
    ]
    conn.close()


def test_low_confidence_claim_defers_instead_of_killing_high_confidence_belief(
    tmp_path: Path,
) -> None:
    """Temporal order breaks the direction of a conflict, never whether."""
    conn = _core(tmp_path)
    standing = apply_claims(
        conn,
        [_claim("research-vm-live", "The live analysis VM is asa2.", confidence=0.95)],
        model="test",
        project=PROJECT,
    )["applied"][0]

    result = apply_claims(
        conn,
        [
            _claim(
                "research-vm-live",
                "The live analysis VM might have moved to asa3.",
                confidence=0.6,
            )
        ],
        model="test",
        project=PROJECT,
    )

    # Recorded, not enacted: the correction is in the ledger as an undecided
    # proposal, and the well-evidenced belief is still the one being served.
    assert result["deferred"] == [standing]
    assert result["applied"] == []
    assert result["superseded"] == []
    assert pending_supersede_count(conn) == 1
    assert _serving(conn) == {standing: "The live analysis VM is asa2."}

    # The same claim at a confidence inside the margin does land.
    landed = apply_claims(
        conn,
        [
            _claim(
                "research-vm-live",
                "The live analysis VM might have moved to asa3.",
                confidence=0.91,
            )
        ],
        model="test",
        project=PROJECT,
    )
    assert landed["deferred"] == []
    assert len(landed["superseded"]) == 1
    conn.close()


def test_correction_kind_is_curator_eligible(tmp_path: Path) -> None:
    """A write-only hole: agents could ingest these, the curator never read them."""
    assert {"correction", "gotcha"} <= ELIGIBLE_KINDS

    conn = _core(tmp_path)
    correction_id = _evidence(
        conn, "The asa1 host was terminated; asa2 is the live box.", kind="correction"
    )
    gotcha_id = _evidence(
        conn, "BSD grep goes silent on a file containing a NUL byte.", kind="gotcha"
    )
    selected = {row["evidence_id"] for row in select_evidence(conn, limit=20, project=PROJECT)}

    assert {correction_id, gotcha_id} <= selected
    conn.close()


def test_stale_evidence_cannot_overwrite_fresher_correction(tmp_path: Path) -> None:
    """A scheduled curator reads a window, not a diff, so Monday comes back."""
    conn = _core(tmp_path)
    evidence_id = _evidence(conn, "The live analysis VM is asa1 for all research work.")
    claim = _claim(
        "research-vm-live",
        "The live analysis VM is asa1.",
        evidence_ids=[evidence_id],
    )
    belief_id = apply_claims(conn, [claim], model="test", project=PROJECT)["applied"][0]

    # A human fixes it after that evidence was recorded.
    correct_v1(
        conn,
        layer="belief",
        target=belief_id,
        op="edit",
        body="The live analysis VM is asa2; asa1 is terminated.",
        actor="human:jonathan",
        hard=False,
    )
    conn.commit()

    # The next cycle sees the same old evidence and recompiles the same key.
    result = apply_claims(conn, [dict(claim)], model="test", project=PROJECT)

    assert result["blocked"] == [belief_id]
    assert result["applied"] == []
    assert result["superseded"] == []
    assert result["deferred"] == []
    assert _serving(conn) == {belief_id: "The live analysis VM is asa2; asa1 is terminated."}

    # Evidence recorded after the correction is not stale, and lands.
    fresher = _evidence(conn, "Research work has moved again, to the asa3 host.")
    landed = apply_claims(
        conn,
        [
            _claim(
                "research-vm-live",
                "The live analysis VM is asa3.",
                evidence_ids=[fresher],
            )
        ],
        model="test",
        project=PROJECT,
    )
    assert landed["blocked"] == []
    assert len(landed["superseded"]) == 1
    conn.close()


def test_curated_contradiction_reaches_the_context_packet(tmp_path: Path) -> None:
    """End to end: `contradictions[]` finally has a writer, so it has content.

    `attributes.contradicts` has been read by the packet builder since the
    packet existed and written by nothing, so every packet the brain has ever
    served carried an empty array while happily handing the reader two answers
    to one question.
    """
    conn = _core(tmp_path)
    standing = apply_claims(
        conn,
        [
            _claim(
                "holdout-share",
                "The Coframe project holdout share is five percent of traffic.",
            )
        ],
        model="test",
        project=PROJECT,
    )["applied"][0]
    result = apply_claims(
        conn,
        [
            _claim(
                "experiment-holdout-share",
                "The Coframe experiment holdout share is ten percent of traffic.",
                conflicts_with=[{"belief_id": standing, "resolution": "coexist"}],
            )
        ],
        model="test",
        project=PROJECT,
    )
    minted = result["applied"][0]
    conn.commit()

    packet, _handles = build_context_v1(
        conn,
        "Coframe holdout share",
        context=ScopeContext(project=PROJECT),
        limit=12,
    )

    assert {standing, minted} <= {str(item["id"]) for item in packet["items"]}
    declared = [
        conflict
        for conflict in packet["contradictions"]
        if conflict["reason"] == "explicit_compiler_metadata"
    ]
    assert len(declared) == 1
    assert {declared[0]["belief_id"], declared[0]["other_belief_id"]} == {standing, minted}
    conn.close()


def test_a_deferred_supersession_leaves_the_pending_ledger_decidable(
    tmp_path: Path,
) -> None:
    """Deferring is postponement, not refusal: an operator can still finish it."""
    conn = _core(tmp_path)
    standing = apply_claims(
        conn,
        [_claim("research-vm-live", "The live analysis VM is asa2.", confidence=0.95)],
        model="test",
        project=PROJECT,
    )["applied"][0]
    apply_claims(
        conn,
        [_claim("research-vm-live", "The live analysis VM is now asa3.", confidence=0.6)],
        model="test",
        project=PROJECT,
    )
    proposal = conn.execute(
        "SELECT id FROM brain_events WHERE kind='compilation_proposed' "
        "AND json_extract(body_json, '$.attributes.supersedes') IS NOT NULL"
    ).fetchone()

    decide_proposal_v1(
        conn,
        proposal_event_id=str(proposal["id"]),
        decision="approve",
        actor="human:jonathan",
        edited_body=None,
        reason="checked the host list; asa3 is right",
    )
    conn.commit()

    assert pending_supersede_count(conn) == 0
    assert get_core_v1_belief(conn, standing)["status"] == "retracted"
    assert list(_serving(conn).values()) == ["The live analysis VM is now asa3."]
    conn.close()


def test_a_blocked_key_update_does_not_stop_the_rest_of_the_run(tmp_path: Path) -> None:
    """One poisoned claim must not cost a project its whole cycle."""
    conn = _core(tmp_path)
    evidence_id = _evidence(conn, "The live analysis VM is asa1 for all research work.")
    stale = _claim("research-vm-live", "The live analysis VM is asa1.", evidence_ids=[evidence_id])
    belief_id = apply_claims(conn, [stale], model="test", project=PROJECT)["applied"][0]
    correct_v1(
        conn,
        layer="belief",
        target=belief_id,
        op="edit",
        body="The live analysis VM is asa2; asa1 is terminated.",
        actor="human:jonathan",
        hard=False,
    )
    conn.commit()

    result = apply_claims(
        conn,
        [
            dict(stale),
            _claim("nul-byte-grep", "BSD grep goes silent on a file containing a NUL byte."),
        ],
        model="test",
        project=PROJECT,
    )

    assert result["blocked"] == [belief_id]
    assert len(result["applied"]) == 1
    conn.close()


def test_a_pinned_belief_is_never_recompiled_over_unattended(tmp_path: Path) -> None:
    """A pin is a standing operator decision; the curator is not attended."""
    conn = _core(tmp_path)
    belief_id = apply_claims(
        conn,
        [_claim("research-vm-live", "The live analysis VM is asa2.")],
        model="test",
        project=PROJECT,
    )["applied"][0]
    correct_v1(
        conn,
        layer="belief",
        target=belief_id,
        op="pin",
        body=None,
        actor="human:jonathan",
        hard=False,
    )
    conn.commit()
    assert get_core_v1_belief(conn, belief_id)["pinned"]

    result = apply_claims(
        conn,
        [_claim("research-vm-live", "The live analysis VM is asa3.")],
        model="test",
        project=PROJECT,
    )

    assert result["deferred"] == [belief_id]
    assert _serving(conn) == {belief_id: "The live analysis VM is asa2."}
    conn.close()


def test_a_malformed_operator_config_is_not_reported_as_blocked_claims(
    tmp_path: Path, monkeypatch
) -> None:
    """The original outage shape: a typo'd config masquerading as tombstones.

    The supersede router lazily loads the operator config per claim, and
    json.JSONDecodeError is a ValueError -- the same type the per-claim guard
    catches to mean "previously tombstoned target". Before ConfigError existed,
    a malformed file made apply_claims report every claim blocked and the log
    said nothing about the file. The config problem must surface as itself.
    """
    from ocbrain.config import ConfigError

    conn = _core(tmp_path)
    evidence_id = _evidence(conn, "The live analysis VM is asa2 for research work.")
    apply_claims(
        conn,
        [_claim("research-vm-live", "The live analysis VM is asa2.", evidence_ids=[evidence_id])],
        model="test",
        project=PROJECT,
    )
    conn.commit()

    broken = tmp_path / "broken-config.json"
    broken.write_text('{"supersede": {', encoding="utf-8")
    monkeypatch.setenv("OCBRAIN_CONFIG", str(broken))

    # A changed body against the same key routes through supersede_transaction,
    # which is where the per-claim config load lives.
    updated = _claim(
        "research-vm-live",
        "The live analysis VM is asa3; asa2 was terminated.",
        evidence_ids=[_evidence(conn, "asa3 replaced asa2 this week.")],
    )
    with pytest.raises(ConfigError) as err:
        apply_claims(conn, [updated], model="test", project=PROJECT)
    assert str(broken) in str(err.value)
    conn.close()
