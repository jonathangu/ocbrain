"""The curator's egress gate, made falsifiable.

The gate had never refused anything, and not by luck: `select_evidence` filtered
on the allow-list inside its own SQL, so the audit could only ever be handed rows
that had already passed. Across 240 audits on the live core, spanning 2026-08-04
to 2026-08-28 and 25,106 transmitted items, `rejected_json` was the literal
string `'[]'` every single time -- `SELECT DISTINCT` returned exactly one value.
A guard whose failing input is unreachable is a transmission log.

Two things are pinned here. A disallowed item must land in the audit's rejected
list and must not be in the payload, so the refusal is *observable*. And the
selftest must report an allow-list that admits every hosted-eligible policy the
corpus contains as a defect, rather than reporting 240 clean audits. A
``local_only`` declaration is a separate, stronger failure: hosted policy
resolution rejects it before evidence selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocbrain.core_v1 import init_core_v1, record_core_v1_evidence
from ocbrain.curator import (
    allowlist_is_vacuous,
    partition_evidence,
    record_curation_egress,
    select_evidence,
)
from ocbrain.db import connect
from ocbrain.scope import ScopeTag
from ocbrain.selftest import run_selftest

PROJECT = "test"


def _scope(*, visibility: str = "internal", egress_policy: str = "hosted_ok") -> ScopeTag:
    return ScopeTag(
        "project",
        f"project:{PROJECT}",
        visibility=visibility,
        egress_policy=egress_policy,
        provenance="test",
    )


def _core(tmp_path: Path):
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    return conn


def _evidence(conn, body: str, scope: ScopeTag, *, kind: str = "audit_finding") -> str:
    evidence_id, _event = record_core_v1_evidence(
        conn, body=body, kind=kind, scope=scope, writer="test"
    )
    conn.commit()
    return evidence_id


def _seeded(tmp_path: Path):
    """One row of each disposition: admitted, floor-refused, not-declared."""
    conn = _core(tmp_path)
    allowed = _evidence(conn, "An admitted finding about the pipeline.", _scope())
    prohibited = _evidence(
        conn,
        "A prohibited finding that must never leave this machine.",
        _scope(visibility="confidential", egress_policy="prohibited"),
    )
    undeclared = _evidence(
        conn,
        "A local-only finding this run did not declare.",
        _scope(egress_policy="local_only"),
    )
    return conn, allowed, prohibited, undeclared


def _seeded_hosted_policies(tmp_path: Path):
    """Every hosted-eligible disposition plus one code-floor refusal."""
    conn = _core(tmp_path)
    hosted = _evidence(conn, "A hosted finding about the pipeline.", _scope())
    approval = _evidence(
        conn,
        "A finding whose hosted use requires operator approval.",
        _scope(egress_policy="approval_required"),
    )
    prohibited = _evidence(
        conn,
        "A prohibited finding that must never leave this machine.",
        _scope(visibility="confidential", egress_policy="prohibited"),
    )
    return conn, hosted, approval, prohibited


def test_a_disallowed_item_is_refused_and_never_transmitted(tmp_path: Path) -> None:
    conn, allowed, prohibited, undeclared = _seeded(tmp_path)

    partition = partition_evidence(
        conn, limit=50, project=PROJECT, egress_policies=["hosted_ok"], visibilities=["internal"]
    )

    included_ids = [str(row["evidence_id"]) for row in partition["included"]]
    assert included_ids == [allowed]
    assert partition["rejected_count"] == 2
    refusals = {row["evidence_id"]: row["reason"] for row in partition["rejected"]}
    assert refusals[prohibited] == "forbidden_egress_policy:prohibited"
    assert refusals[undeclared] == "egress_policy_not_declared:local_only"

    audit_id = record_curation_egress(
        conn,
        evidence=partition["included"],
        provider="anthropic",
        model="claude-sonnet-5",
        project=PROJECT,
        egress_policies=("hosted_ok",),
        rejected=partition["rejected"],
        rejected_count=partition["rejected_count"],
        visibilities=("internal",),
        present_egress_policies=partition["present_egress_policies"],
        present_visibilities=partition["present_visibilities"],
    )
    row = conn.execute(
        "SELECT included_json, rejected_json, context_json FROM egress_audits WHERE id=?",
        (audit_id,),
    ).fetchone()

    # The refusal is in the record, which is the whole point: this column has
    # been the string '[]' on every audit this brain has ever written.
    rejected_json = str(row["rejected_json"])
    assert rejected_json != "[]"
    assert len(json.loads(rejected_json)) == 2
    # And the refused bodies are not in what was sent.
    included = json.loads(str(row["included_json"]))
    assert [item["evidence_id"] for item in included] == [allowed]
    assert prohibited not in str(row["included_json"])
    assert undeclared not in str(row["included_json"])


def test_the_audit_records_what_was_declared_beside_what_was_present(tmp_path: Path) -> None:
    """"Nothing was rejected" means two different things and the row must say which."""
    conn, _hosted, _approval, _prohibited = _seeded_hosted_policies(tmp_path)
    partition = partition_evidence(
        conn,
        limit=50,
        project=PROJECT,
        egress_policies=["hosted_ok", "approval_required"],
        visibilities=["internal", "confidential"],
    )
    audit_id = record_curation_egress(
        conn,
        evidence=partition["included"],
        provider="anthropic",
        model="claude-sonnet-5",
        project=PROJECT,
        egress_policies=tuple(partition["declared_egress_policies"]),
        rejected=partition["rejected"],
        rejected_count=partition["rejected_count"],
        visibilities=partition["declared_visibilities"],
        present_egress_policies=partition["present_egress_policies"],
        present_visibilities=partition["present_visibilities"],
    )
    context = json.loads(
        str(
            conn.execute(
                "SELECT context_json FROM egress_audits WHERE id=?", (audit_id,)
            ).fetchone()[0]
        )
    )
    assert context["declared_egress_policies"] == ["approval_required", "hosted_ok"]
    assert context["present_egress_policies"] == [
        "approval_required",
        "hosted_ok",
        "prohibited",
    ]
    # `prohibited` is refused in code, not by the declaration, so declaring
    # everything an operator may declare is still vacuous.
    assert context["allowlist_vacuous"] is True
    assert context["rejected_count"] == 1


def test_select_evidence_still_returns_exactly_what_it_always_did(tmp_path: Path) -> None:
    """The policy did not change. Only the refusals became visible."""
    conn, allowed, _prohibited, _undeclared = _seeded(tmp_path)
    selected = select_evidence(
        conn, limit=50, project=PROJECT, egress_policies=["hosted_ok"], visibilities=["internal"]
    )
    assert [str(row["evidence_id"]) for row in selected] == [allowed]


def test_allowlist_is_vacuous_reads_the_reachability_not_the_outcome() -> None:
    assert allowlist_is_vacuous(
        ["hosted_ok", "approval_required"], ["hosted_ok", "approval_required"]
    ) is True
    assert allowlist_is_vacuous(["hosted_ok"], ["hosted_ok", "approval_required"]) is False
    # An empty corpus proves nothing either way, and must not be reported as a
    # vacuous gate.
    assert allowlist_is_vacuous(["hosted_ok"], []) is False


# --------------------------------------------------------------------------- #
# the selftest metric
# --------------------------------------------------------------------------- #


def _metric(conn):
    scorecard = run_selftest(conn)
    return next(m for m in scorecard["metrics"] if m["key"] == "egress_refusable_policies")


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, curator: dict) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"curator": curator}), encoding="utf-8")
    monkeypatch.setenv("OCBRAIN_CONFIG", str(path))


def test_an_allowlist_that_admits_everything_present_is_an_alarm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn, _hosted, _approval, _prohibited = _seeded_hosted_policies(tmp_path)
    _config(
        tmp_path,
        monkeypatch,
        {"egress_policies": ["hosted_ok", "approval_required"]},
    )
    metric = _metric(conn)
    assert metric["value"] == 0.0
    assert metric["status"] == "alarm"
    assert metric["detail"]["refusable_policies"] == []
    assert metric["detail"]["present_egress_policies"] == [
        "approval_required",
        "hosted_ok",
        "prohibited",
    ]


def test_a_narrower_allowlist_has_teeth_and_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn, _allowed, _prohibited, _undeclared = _seeded(tmp_path)
    _config(tmp_path, monkeypatch, {"egress_policies": ["hosted_ok"]})
    metric = _metric(conn)
    assert metric["value"] == 1.0
    assert metric["status"] == "ok"
    assert metric["detail"]["refusable_policies"] == ["local_only"]


def test_a_declared_acknowledgement_downgrades_but_never_hides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exemptions are declared; findings are not enumerated away."""
    conn, _hosted, _approval, _prohibited = _seeded_hosted_policies(tmp_path)
    _config(
        tmp_path,
        monkeypatch,
        {
            "egress_policies": ["hosted_ok", "approval_required"],
            "egress_allowlist_ack": "all hosted inputs have separate operator approval",
        },
    )
    metric = _metric(conn)
    assert metric["value"] == 0.0
    assert metric["status"] == "watch"
    assert metric["detail"]["acknowledgement"].startswith("all hosted inputs")


def test_the_payload_limit_caps_what_is_sent_without_truncating_the_denominator(
    tmp_path: Path,
) -> None:
    """The limit stops selecting; it must not stop *counting*.

    This is why the cap is a `continue` and not the `break` it used to be. A
    `break` at the limit would leave the audit reporting only the refusals it
    happened to reach before the payload filled up -- the same class of defect
    as filtering the refusals out in SQL, just with a different cause: a
    denominator that depends on where the loop stopped.
    """
    conn = _core(tmp_path)
    for index in range(6):
        record_core_v1_evidence(
            conn,
            body=f"Sendable fact {index}.",
            kind="task_closeout_summary",
            scope=_scope(),
            writer="test",
        )
    refused_ids = [
        record_core_v1_evidence(
            conn,
            body=f"Local-only fact {index}.",
            kind="task_closeout_summary",
            scope=_scope(egress_policy="local_only"),
            writer="test",
        )[0]
        for index in range(3)
    ]
    # Order the refused rows last, so a loop that stopped at the limit would
    # never reach them.
    conn.executemany(
        "UPDATE evidence_objects SET recorded_at='2000-01-01T00:00:00+00:00' "
        "WHERE evidence_id=?",
        [(evidence_id,) for evidence_id in refused_ids],
    )
    conn.commit()

    partition = partition_evidence(conn, limit=2, project=PROJECT)
    assert len(partition["included"]) == 2
    assert partition["rejected_count"] == 3
    assert {row["reason"] for row in partition["rejected"]} == {
        "egress_policy_not_declared:local_only"
    }
    assert sorted(partition["present_egress_policies"]) == ["hosted_ok", "local_only"]
    # And the two survivors are eligible ones, not the first two rows of the table.
    assert all(row["egress_policy"] == "hosted_ok" for row in partition["included"])

    # `select_evidence` is the same policy through a narrower door.
    assert len(select_evidence(conn, limit=2, project=PROJECT)) == 2
