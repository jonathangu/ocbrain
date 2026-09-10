"""The pending supersede ledger: dedup, curator authority, and confidence.

The first unattended night proved the ledger had producers and no consumer. The
scheduled curator's per-caller rate cap -- sized for a runtime agent -- pended
everything past its eighth correction of the day, and because a *proposal* does
not change the input that produced it, the next hourly cycle re-derived the same
claims and pended them again. The live core reached 283 undecided proposals over
33 beliefs in eighteen hours, one pair carrying twelve identical copies, and the
only operator-facing number said "283" as though that were a backlog.

Three guards close it, and each is pinned here:

* a supersession the ledger already carries undecided is not minted twice,
* the curator supersedes an *ordinary* belief directly instead of pending it,
* and a same-key curator refresh holds the fact's confidence instead of
  ratcheting it toward the contested-correction ceiling.

The fourth test class is the one that matters most and is easiest to forget:
``brain.supersede`` reads ``actor`` straight from client arguments, so curator
authority must not be purchasable by typing the curator's name.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path

import ocbrain.curator as curator
import ocbrain.mcp_v1 as mcp_v1
from ocbrain.core_v1 import (
    append_core_event,
    get_core_v1_belief,
    init_core_v1,
    record_core_v1_evidence,
)
from ocbrain.curator import CURATOR_VERSION, apply_claims
from ocbrain.db import connect
from ocbrain.mcp import handle_request
from ocbrain.mcp_v1 import (
    CURATOR_SUPERSEDE_WRITER,
    correct_v1,
    decide_proposal_v1,
    is_curator_writer,
    pending_supersede_count,
    pending_supersede_targets,
    supersede_transaction,
)
from ocbrain.provenance import EMPTY_PROVENANCE
from ocbrain.scope import ScopeTag

PROJECT = "test"
PROJECT_SCOPE = ScopeTag(
    "project",
    f"project:{PROJECT}",
    visibility="internal",
    egress_policy="hosted_ok",
    provenance="test",
)
DOCTRINE_SCOPE = ScopeTag(
    "global",
    "global:doctrine",
    visibility="internal",
    egress_policy="local_only",
    provenance="test",
)
CURATOR_ACTOR = f"operator-approved:{CURATOR_VERSION}"


def _core(tmp_path: Path):
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    return conn


def _claim(key: str, body: str, *, confidence: float = 0.9) -> dict:
    return {
        "key": key,
        "title": key.replace("-", " "),
        "body": body,
        "category": "system",
        "lifecycle": "durable",
        "confidence": confidence,
        "evidence_ids": [],
    }


def _seed(
    conn,
    *,
    belief_id: str,
    body: str,
    scope: ScopeTag = PROJECT_SCOPE,
    attributes: dict | None = None,
    confidence: float = 0.9,
) -> dict:
    evidence_id, _event = record_core_v1_evidence(
        conn, body=f"evidence for {belief_id}", kind="observation", scope=scope, writer="test"
    )
    proposal = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": belief_id,
            "belief_type": "wiki_fact",
            "body": body,
            "evidence_ids": [evidence_id],
            "scope": scope.to_dict(),
            "confidence": confidence,
            "attributes": attributes or {},
        },
        writer="test",
    )
    append_core_event(
        conn,
        "compilation_decided",
        {"proposal_event_id": proposal, "decision": "approve", "actor": "test"},
        writer="test",
        project=True,
    )
    conn.commit()
    return get_core_v1_belief(conn, belief_id)


def _propose(conn, old: dict, statement: str, *, actor: str = "agent:one", **kwargs) -> dict:
    """Run the shared transaction directly, the way both real doors do."""
    outcome = supersede_transaction(
        conn,
        old=old,
        statement=statement,
        rationale="the stored statement is out of date",
        attributes=dict(old.get("attributes") or {}),
        actor=actor,
        provenance=EMPTY_PROVENANCE,
        **kwargs,
    )
    conn.commit()
    return outcome


def _events(conn) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0])


def _proposed(conn) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM brain_events WHERE kind='compilation_proposed'"
        ).fetchone()[0]
    )


def _serving(conn) -> dict[str, str]:
    return {
        str(row["belief_id"]): str(row["body"])
        for row in conn.execute(
            "SELECT belief_id, body FROM current_beliefs WHERE serve=1 AND status='current'"
        )
    }


# --------------------------------------------------------------------------- #
# Dedup at proposal time
# --------------------------------------------------------------------------- #
def test_re_proposing_an_identical_supersede_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    """The bug, in one test: the same pair proposed twice is one proposal.

    Nothing is written on the second call -- not the proposal, and not the
    rationale evidence row either -- so an hourly loop that re-derives the same
    claim costs the ledger nothing at all rather than costing it less.
    """
    monkeypatch.setenv("OCBRAIN_SUPERSEDE_TIER", "pending_all")
    conn = _core(tmp_path)
    old = _seed(conn, belief_id="belief:vm", body="The research VM is asa1.")

    first = _propose(conn, old, "The research VM is asa2.")
    after_first = _events(conn)
    second = _propose(conn, old, "The research VM is asa2.")

    assert first["mode"] == "pending"
    assert not first.get("deduped")
    assert second["mode"] == "pending"
    assert second["deduped"] is True
    # It points at the proposal already carrying this supersession.
    assert second["proposal_event_id"] == first["proposal_event_id"]
    assert second["successor_id"] == first["successor_id"]
    assert _events(conn) == after_first
    assert pending_supersede_count(conn) == 1
    conn.close()


def test_concurrent_identical_supersedes_create_one_pending_proposal(
    tmp_path: Path, monkeypatch
) -> None:
    """The pending lookup and append serialize under the same write lock."""
    monkeypatch.setenv("OCBRAIN_SUPERSEDE_TIER", "pending_all")
    setup = _core(tmp_path)
    belief_id = "belief:vm"
    _seed(setup, belief_id=belief_id, body="The research VM is asa1.")
    setup.close()

    # Meet both callers at the real transaction boundary. This adds only
    # synchronization: the original context manager still owns BEGIN IMMEDIATE,
    # commit/rollback semantics, and the body of the transaction.
    barrier = threading.Barrier(2)
    original_transaction = mcp_v1._one_transaction

    @contextmanager
    def synchronized_transaction(conn):
        barrier.wait(timeout=5)
        with original_transaction(conn):
            yield

    monkeypatch.setattr(mcp_v1, "_one_transaction", synchronized_transaction)

    outcomes: list[dict] = []
    errors: list[BaseException] = []
    outcome_lock = threading.Lock()
    db_path = tmp_path / "core.sqlite"

    def propose() -> None:
        conn = connect(db_path)
        try:
            old = get_core_v1_belief(conn, belief_id)
            assert old is not None
            outcome = supersede_transaction(
                conn,
                old=old,
                statement="The research VM is asa2.",
                rationale="the stored statement is out of date",
                attributes=dict(old.get("attributes") or {}),
                actor="agent:race",
                provenance=EMPTY_PROVENANCE,
            )
            conn.commit()
            with outcome_lock:
                outcomes.append(outcome)
        except BaseException as exc:  # pragma: no cover - asserted below
            conn.rollback()
            with outcome_lock:
                errors.append(exc)
        finally:
            conn.close()

    workers = [threading.Thread(target=propose) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)

    assert all(not worker.is_alive() for worker in workers)
    assert errors == []
    assert len(outcomes) == 2
    assert sorted(bool(outcome.get("deduped")) for outcome in outcomes) == [False, True]
    assert len({outcome["proposal_event_id"] for outcome in outcomes}) == 1

    check = connect(db_path)
    try:
        assert pending_supersede_count(check) == 1
        assert (
            check.execute(
                "SELECT COUNT(*) FROM brain_events AS proposal "
                "WHERE proposal.kind='compilation_proposed' "
                "AND json_extract(proposal.body_json, '$.attributes.supersedes') = ? "
                "AND json_extract(proposal.body_json, '$.belief_id') = ?",
                (belief_id, outcomes[0]["successor_id"]),
            ).fetchone()[0]
            == 1
        )
    finally:
        check.close()


def test_a_different_replacement_body_for_the_same_target_still_mints(
    tmp_path: Path, monkeypatch
) -> None:
    """Dedup is on the pair, not the target.

    Two operators proposing different replacements for one wrong belief is a
    disagreement an admin has to see, not a duplicate to swallow.
    """
    monkeypatch.setenv("OCBRAIN_SUPERSEDE_TIER", "pending_all")
    conn = _core(tmp_path)
    old = _seed(conn, belief_id="belief:vm", body="The research VM is asa1.")

    first = _propose(conn, old, "The research VM is asa2.")
    second = _propose(conn, old, "The research VM is asa3.")

    assert not second.get("deduped")
    assert second["successor_id"] != first["successor_id"]
    assert pending_supersede_count(conn) == 2
    # Two proposals, one belief: exactly the shape the headline metric must show.
    assert pending_supersede_targets(conn) == 1
    conn.close()


def test_a_decided_proposal_does_not_block_a_new_one(tmp_path: Path, monkeypatch) -> None:
    """Dedup keys on *undecided*, so a rejection cannot wedge the pair forever."""
    monkeypatch.setenv("OCBRAIN_SUPERSEDE_TIER", "pending_all")
    conn = _core(tmp_path)
    old = _seed(conn, belief_id="belief:vm", body="The research VM is asa1.")

    first = _propose(conn, old, "The research VM is asa2.")
    decide_proposal_v1(
        conn,
        proposal_event_id=str(first["proposal_event_id"]),
        decision="reject",
        actor="human:jonathan",
        edited_body=None,
        reason="checked the host list; asa1 is still right",
    )
    conn.commit()

    again = _propose(conn, old, "The research VM is asa2.")

    assert not again.get("deduped")
    assert again["proposal_event_id"] != first["proposal_event_id"]
    assert pending_supersede_count(conn) == 1
    conn.close()


def test_the_curator_reports_a_deduped_supersession_apart_from_a_deferred_one(
    tmp_path: Path,
) -> None:
    """A stable loop and a stopped loop must not look the same in the log."""
    conn = _core(tmp_path)
    standing = apply_claims(
        conn,
        [_claim("research-vm-live", "The live analysis VM is asa2.", confidence=0.95)],
        model="test",
        project=PROJECT,
    )["applied"][0]
    below_margin = _claim("research-vm-live", "The live analysis VM is asa3.", confidence=0.6)

    first = apply_claims(conn, [below_margin], model="test", project=PROJECT)
    second = apply_claims(conn, [below_margin], model="test", project=PROJECT)

    assert first["deferred"] == [standing]
    assert first["pending_deduped"] == []
    # Second cycle, same claim: recognised, not re-proposed.
    assert second["deferred"] == []
    assert second["pending_deduped"] == [standing]
    assert pending_supersede_count(conn) == 1
    conn.close()


def test_the_curator_does_not_pend_a_supersession_of_a_retracted_target(
    tmp_path: Path, monkeypatch
) -> None:
    """A target retired between selection and the write is moot, not pending.

    The curator shares the core with the wiki-curator script and with hygiene, so
    the belief it selected can be retired by another writer before it proposes.
    A supersession whose target is already gone cannot be decided by anyone --
    approving it would era-close a closed belief -- so it must never reach the
    queue and age the headline metric.
    """
    conn = _core(tmp_path)
    standing = apply_claims(
        conn,
        [_claim("research-vm-live", "The live analysis VM is asa2.", confidence=0.95)],
        model="test",
        project=PROJECT,
    )["applied"][0]
    below_margin = _claim("research-vm-live", "The live analysis VM is asa3.", confidence=0.6)
    proposed_before = _proposed(conn)

    real_serving_belief = curator._serving_belief
    retired: list[str] = []

    def _retired_by_another_writer(cnx, belief_id):
        if str(belief_id) == standing and not retired:
            retired.append(str(belief_id))
            correct_v1(
                cnx,
                layer="belief",
                target=belief_id,
                op="retract",
                body=None,
                actor="human:jonathan",
                hard=False,
            )
            cnx.commit()
        return real_serving_belief(cnx, belief_id)

    monkeypatch.setattr(curator, "_serving_belief", _retired_by_another_writer)
    result = apply_claims(conn, [below_margin], model="test", project=PROJECT)

    assert retired == [standing]
    assert result["moot"] == [standing]
    assert result["deferred"] == []
    assert result["pending_deduped"] == []
    assert _proposed(conn) == proposed_before
    assert pending_supersede_count(conn) == 0
    conn.close()


# --------------------------------------------------------------------------- #
# Curator direct authority
# --------------------------------------------------------------------------- #
def test_the_curator_supersedes_an_ordinary_belief_directly(
    tmp_path: Path, monkeypatch
) -> None:
    """The cap is pinned shut, so only curator authority can land this.

    Without that the test passes on the ordinary cap path and proves nothing --
    which is exactly what it did until a mutation run said so.
    """
    monkeypatch.setenv("OCBRAIN_SUPERSEDE_DIRECT_CAP", "0")
    conn = _core(tmp_path)
    old = _seed(conn, belief_id="belief:vm", body="The research VM is asa1.")

    outcome = _propose(conn, old, "The research VM is asa2.", actor=CURATOR_ACTOR,
                       curator_authored=True)

    assert outcome["mode"] == "direct"
    assert pending_supersede_count(conn) == 0
    assert list(_serving(conn).values()) == ["The research VM is asa2."]
    conn.close()


def test_the_curator_still_pends_a_pinned_target(tmp_path: Path) -> None:
    """A pin is a standing operator decision and outranks the schedule."""
    conn = _core(tmp_path)
    old = _seed(conn, belief_id="belief:vm", body="The research VM is asa1.")
    correct_v1(
        conn,
        layer="belief",
        target="belief:vm",
        op="pin",
        body=None,
        actor="human:jonathan",
        hard=False,
    )
    conn.commit()
    old = get_core_v1_belief(conn, "belief:vm")

    outcome = _propose(conn, old, "The research VM is asa2.", actor=CURATOR_ACTOR,
                       curator_authored=True)

    assert outcome["mode"] == "pending"
    assert "pinned" in outcome["pending_reason"]
    assert list(_serving(conn).values()) == ["The research VM is asa1."]
    conn.close()


def test_the_curator_still_pends_doctrine(tmp_path: Path) -> None:
    conn = _core(tmp_path)
    old = _seed(
        conn,
        belief_id="belief:doctrine",
        body="Agents ask permission before spending money.",
        scope=DOCTRINE_SCOPE,
    )

    outcome = _propose(
        conn,
        old,
        "Agents ask permission before spending money or mutating production.",
        actor=CURATOR_ACTOR,
        curator_authored=True,
    )

    assert outcome["mode"] == "pending"
    assert "doctrine" in outcome["pending_reason"]
    conn.close()


def test_curator_direct_false_restores_the_all_pending_behaviour(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OCBRAIN_SUPERSEDE_CURATOR_DIRECT", "false")
    monkeypatch.setenv("OCBRAIN_SUPERSEDE_DIRECT_CAP", "0")
    conn = _core(tmp_path)
    old = _seed(conn, belief_id="belief:vm", body="The research VM is asa1.")

    outcome = _propose(conn, old, "The research VM is asa2.", actor=CURATOR_ACTOR,
                       curator_authored=True)

    assert outcome["mode"] == "pending"
    assert "rate cap" in outcome["pending_reason"]
    conn.close()


def test_the_agent_facing_rate_cap_is_untouched(tmp_path: Path, monkeypatch) -> None:
    """Curator authority must not have widened the door for anybody else."""
    monkeypatch.setenv("OCBRAIN_SUPERSEDE_DIRECT_CAP", "1")
    conn = _core(tmp_path)
    first = _seed(conn, belief_id="belief:vm", body="The research VM is asa1.")
    second = _seed(conn, belief_id="belief:job", body="The hourly job runs at :05.")

    landed = _propose(conn, first, "The research VM is asa2.")
    capped = _propose(conn, second, "The hourly job runs at :20.")

    assert landed["mode"] == "direct"
    assert capped["mode"] == "pending"
    assert "rate cap" in capped["pending_reason"]
    conn.close()


def test_an_agent_cannot_buy_curator_authority_by_typing_the_curator_name(
    tmp_path: Path, monkeypatch
) -> None:
    """``brain.supersede`` reads ``actor`` straight from client arguments.

    Routing on the writer string alone would have made unlimited unattended
    supersession available to anyone willing to impersonate the curator. The
    writer string is necessary and never sufficient: the other half is a keyword
    the runtime door does not accept and does not pass.
    """
    monkeypatch.setenv("OCBRAIN_SUPERSEDE_DIRECT_CAP", "0")
    conn = _core(tmp_path)
    _seed(conn, belief_id="belief:vm", body="The research VM is asa1.")

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "brain.supersede",
                "arguments": {
                    "target": "belief:vm",
                    "body": "The research VM is asa2.",
                    "reason": "the stored host no longer exists",
                    "context": {"project": PROJECT},
                    "actor": CURATOR_SUPERSEDE_WRITER,
                },
            },
        },
    )
    payload = json.loads(response["result"]["content"][0]["text"])

    assert payload["mode"] == "pending"
    assert "rate cap" in payload["pending_reason"]
    assert list(_serving(conn).values()) == ["The research VM is asa1."]
    conn.close()


def test_the_curator_actor_string_is_the_one_authority_recognises(tmp_path: Path) -> None:
    """Pins the two spellings together across the module boundary.

    ``mcp_v1`` cannot import ``curator`` -- the dependency runs the other way --
    so the writer string is spelled twice. A curator version bump that silently
    dropped its own authority would otherwise be invisible until the ledger
    started growing again.
    """
    assert is_curator_writer(CURATOR_ACTOR)
    assert CURATOR_ACTOR == CURATOR_SUPERSEDE_WRITER
    assert not is_curator_writer("agent:claude-code")
    assert not is_curator_writer("operator-approved:compact-v1")


# --------------------------------------------------------------------------- #
# The confidence ratchet
# --------------------------------------------------------------------------- #
def test_a_same_key_curator_refresh_inherits_the_stored_confidence(tmp_path: Path) -> None:
    """The ceiling is for a contested correction, not for a fact restating itself.

    Measured on the live core before this: approving the 33 pending proposals
    as-proposed would have dropped confidence on 30 of them, mean -0.09, every
    one landing on 0.65 or 0.70. Hourly, that walks the whole corpus to 0.7.
    """
    conn = _core(tmp_path)
    old = _seed(
        conn,
        belief_id="belief:vm",
        body="The research VM is asa1.",
        attributes={"key": "research-vm-live"},
        confidence=0.85,
    )

    outcome = _propose(
        conn,
        old,
        "The research VM is asa2.",
        actor=CURATOR_ACTOR,
        curator_authored=True,
        inherit_confidence=True,
        confidence_ceiling=0.82,
    )

    assert outcome["mode"] == "direct"
    # Neither the 0.7 ceiling nor the claim's own 0.82 pulls the fact down.
    assert outcome["confidence"] == 0.85
    assert get_core_v1_belief(conn, outcome["successor_id"])["confidence"] == 0.85
    conn.close()


def test_inheriting_confidence_never_raises_a_fact(tmp_path: Path) -> None:
    """No-gain as well as no-loss: arriving later is still not evidence."""
    conn = _core(tmp_path)
    old = _seed(
        conn,
        belief_id="belief:vm",
        body="The research VM is asa1.",
        attributes={"key": "research-vm-live"},
        confidence=0.6,
    )

    outcome = _propose(
        conn,
        old,
        "The research VM is asa2.",
        actor=CURATOR_ACTOR,
        curator_authored=True,
        inherit_confidence=True,
        confidence_ceiling=0.99,
    )

    assert outcome["confidence"] == 0.6
    conn.close()


def test_a_cross_key_curator_supersession_keeps_the_ceiling(tmp_path: Path) -> None:
    """Replacing a *different* fact is a contested correction, ceiling and all."""
    conn = _core(tmp_path)
    old = _seed(
        conn,
        belief_id="belief:vm",
        body="The research VM is asa1.",
        attributes={"key": "research-vm-live"},
        confidence=0.9,
    )

    outcome = supersede_transaction(
        conn,
        old=old,
        statement="The research VM is asa2.",
        rationale="a different fact now covers this",
        attributes={"key": "research-vm-successor"},
        actor=CURATOR_ACTOR,
        provenance=EMPTY_PROVENANCE,
        curator_authored=True,
        inherit_confidence=True,
    )
    conn.commit()

    assert outcome["confidence"] == 0.7
    conn.close()


def test_an_agent_supersession_keeps_the_ceiling(tmp_path: Path) -> None:
    """Only the curator refreshing its own fact is exempt, and only via the keyword."""
    conn = _core(tmp_path)
    old = _seed(
        conn,
        belief_id="belief:vm",
        body="The research VM is asa1.",
        attributes={"key": "research-vm-live"},
        confidence=0.9,
    )

    # Even asking for inheritance: an agent is not the curator.
    outcome = _propose(
        conn,
        old,
        "The research VM is asa2.",
        actor="agent:claude-code",
        inherit_confidence=True,
    )

    assert outcome["confidence"] == 0.7
    conn.close()
