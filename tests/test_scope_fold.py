"""Scope folding and the operator alias table.

Callers name their own scope and spell it inconsistently, while matching is exact
string equality. These tests pin the two halves of the repair: a caller's
spelling is canonicalized once at construction, and the immutable stored
inventory is matched by its canonical form rather than rewritten.

The boundary they also pin is where folding must NOT happen. ``ScopeTag`` runs
during projection replay and over stored handle scopes; folding there would make
a ledger refold depend on whatever the alias table says today.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocbrain.core_v1 import append_core_event, init_core_v1, project_core_v1, search_core_v1
from ocbrain.db import connect
from ocbrain.mcp import scope_from_arguments
from ocbrain.mcp_v1 import decide_proposal_v1
from ocbrain.scope import (
    ScopeContext,
    ScopeTag,
    fold_scope_component,
    fold_scope_id,
    matching_stored_scope_ids,
    resolve_scope_alias,
    scope_match,
)


@pytest.fixture(autouse=True)
def isolated_scopes_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read the operator's real config; these tests own the alias table."""
    monkeypatch.setenv("OCBRAIN_CONFIG", str(tmp_path / "absent-config.json"))
    monkeypatch.delenv("OCBRAIN_SCOPES_ALIASES", raising=False)
    monkeypatch.delenv("OCBRAIN_SCOPES_FOLD_ENABLED", raising=False)


def _set_aliases(monkeypatch: pytest.MonkeyPatch, aliases: dict[str, str]) -> None:
    monkeypatch.setenv("OCBRAIN_SCOPES_ALIASES", json.dumps(aliases))


def _seed_belief(
    conn,
    *,
    belief_id: str,
    body: str,
    scope: ScopeTag,
) -> None:
    proposal = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": belief_id,
            "belief_type": "curated_fact",
            "body": body,
            "evidence_ids": [],
            "scope": scope.to_dict(),
            "confidence": 0.9,
            "attributes": {"source_quality": 0.95},
        },
        writer="test",
    )
    decide_proposal_v1(
        conn,
        proposal_event_id=proposal,
        decision="approve",
        actor="test",
        edited_body=None,
        reason="scope fold fixture",
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Coframe Brain", "coframe-brain"),
        (" CRO Oracle ", "cro-oracle"),
        ("coframe_brain__v2", "coframe-brain-v2"),
        ("coframe-brain", "coframe-brain"),
        ("COFRAME---BRAIN", "coframe-brain"),
        ("  spaced   out  ", "spaced-out"),
        ("-leading-and-trailing-", "leading-and-trailing"),
        ("Café Ölsen", "café-ölsen"),
        ("", None),
        ("   ", None),
        ("___", None),
        (None, None),
        # Path-shaped components are directory names, not slugs. Folding one
        # renames the directory that ``retrieve`` and ``shared_context`` resolve.
        ("/Users/dev/My_Repo", "/Users/dev/My_Repo"),
        ("~/Developer/OCBrain", "~/Developer/OCBrain"),
    ],
)
def test_fold_scope_component_table(raw, expected) -> None:
    assert fold_scope_component(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["Coframe Brain", "coframe_brain__v2", " CRO Oracle ", "café ölsen", "/tmp/A_Repo"],
)
def test_fold_scope_component_is_idempotent(raw: str) -> None:
    once = fold_scope_component(raw)
    assert fold_scope_component(once) == once


def test_fold_scope_id_keeps_the_type_prefix_verbatim() -> None:
    assert fold_scope_id("project:Coframe Brain") == "project:coframe-brain"
    # Scope types carry their own underscores; folding the prefix would produce
    # a type that matches nothing.
    assert fold_scope_id("legacy_unscoped:Pelican Fund") == "legacy_unscoped:pelican-fund"
    assert fold_scope_id("legacy_unscoped:Legacy Thing") == "legacy_unscoped:legacy-thing"
    assert fold_scope_id("global:doctrine") == "global:doctrine"
    assert fold_scope_id("project:") == "project:"
    assert fold_scope_id("") == ""


def test_scope_context_folds_project_at_construction() -> None:
    context = ScopeContext(project="Coframe Brain", repo="OCBrain_Repo", client="Acme  Corp")

    assert context.project == "coframe-brain"
    assert context.repo == "ocbrain-repo"
    assert context.client == "acme-corp"
    assert context.compatible_scope_ids() == {
        "global:doctrine",
        "project:coframe-brain",
        "repo:ocbrain-repo",
        "client:acme-corp",
    }
    # from_dict goes through the same hook, so every entry point agrees.
    assert ScopeContext.from_dict({"project": "Coframe_Brain"}).project == "coframe-brain"


def test_task_and_session_ids_are_not_folded() -> None:
    context = ScopeContext(task="  PR_3504 Review ", session="Sess_ABC-01")

    assert context.task == "PR_3504 Review"
    assert context.session == "Sess_ABC-01"
    assert "task:PR_3504 Review" in context.compatible_scope_ids()


def test_legacy_public_visibility_is_narrowed_to_internal() -> None:
    direct = ScopeTag(
        "project",
        "project:legacy",
        visibility="public",
        egress_policy="hosted_ok",
        provenance="legacy-event",
    )
    decoded = ScopeTag.from_dict(
        {
            "scope_type": "project",
            "scope_id": "project:legacy",
            "visibility": "public",
            "egress_policy": "hosted_ok",
            "provenance": "legacy-event",
        }
    )

    assert direct.visibility == "internal"
    assert decoded.visibility == "internal"
    assert direct.to_dict()["visibility"] == "internal"


def test_retired_personal_finance_scope_is_quarantined() -> None:
    direct = ScopeTag(
        "personal_finance",
        "personal_finance:pelican",
        visibility="confidential",
        egress_policy="local_only",
        provenance="legacy-event",
    )

    assert direct.scope_type == "legacy_unscoped"
    assert direct.scope_id == "personal_finance:pelican"
    assert direct.visibility == "confidential"
    assert direct.egress_policy == "local_only"


def test_full_replay_narrows_legacy_public_event_scope(tmp_path: Path) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    proposal = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": "curated:legacy:public",
            "belief_type": "curated_fact",
            "body": "Historical public visibility is served as internal.",
            "evidence_ids": [],
            "scope": {
                "scope_type": "project",
                "scope_id": "project:legacy",
                "visibility": "public",
                "egress_policy": "hosted_ok",
                "provenance": "legacy-event",
            },
            "confidence": 0.9,
            "attributes": {"source_quality": 0.95},
        },
        writer="legacy-test",
    )
    decide_proposal_v1(
        conn,
        proposal_event_id=proposal,
        decision="approve",
        actor="test",
        edited_body=None,
        reason="legacy visibility replay fixture",
    )

    before = conn.execute(
        "SELECT visibility FROM current_beliefs WHERE belief_id=?",
        ("curated:legacy:public",),
    ).fetchone()
    assert before["visibility"] == "internal"

    project_core_v1(conn, full=True)
    after = conn.execute(
        "SELECT visibility FROM current_beliefs WHERE belief_id=?",
        ("curated:legacy:public",),
    ).fetchone()
    assert after["visibility"] == "internal"
    conn.close()


def test_alias_resolution_maps_variant_to_canonical_and_unknown_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_aliases(monkeypatch, {"project:coframe-brain": "project:coframe"})

    assert resolve_scope_alias("project:Coframe Brain") == "project:coframe"
    assert resolve_scope_alias("project:coframe_brain") == "project:coframe"
    # Unknown ids pass through folded, never dropped and never invented.
    assert resolve_scope_alias("project:Some Other Thing") == "project:some-other-thing"
    assert ScopeContext(project="Coframe Brain").project == "coframe"


def test_alias_may_rename_a_scope_but_never_retype_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """An alias is not a promotion path.

    Re-typing through the table would let an operator make a project belief
    globally reachable without the ``scope_promoted`` event that records who
    approved it.
    """
    _set_aliases(
        monkeypatch,
        {
            "project:coframe-brain": "global:doctrine",
            "project:workspace-notes": "client:acme",
        },
    )

    assert resolve_scope_alias("project:coframe-brain") == "project:coframe-brain"
    assert resolve_scope_alias("project:workspace-notes") == "project:workspace-notes"
    assert ScopeContext(project="coframe brain").project == "coframe-brain"


def test_empty_alias_table_reproduces_exact_match_behavior(tmp_path: Path) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_belief(
        conn,
        belief_id="curated:coframe:fact",
        body="Trino sandbox schemas are the scratch tier for research queries.",
        scope=ScopeTag("project", "project:coframe", provenance="test"),
    )
    conn.commit()

    compatible = ScopeContext(project="coframe").compatible_scope_ids()
    assert matching_stored_scope_ids(conn, "current_beliefs", compatible) == sorted(compatible)
    conn.close()


def test_search_matches_stored_unfolded_scope_ids(tmp_path: Path) -> None:
    """A belief written under a hand-typed spelling stays reachable forever.

    Stored rows are never rewritten, so the comparison has to move instead.
    """
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_belief(
        conn,
        belief_id="curated:legacy-spelling:fact",
        body="Trino sandbox schemas are the scratch tier for research queries.",
        scope=ScopeTag("project", "project:Coframe Brain", provenance="test"),
    )
    conn.commit()

    result = search_core_v1(
        conn,
        "trino sandbox scratch tier",
        context=ScopeContext(project="coframe_brain"),
        limit=5,
    )

    assert [item["belief_id"] for item in result["items"]] == ["curated:legacy-spelling:fact"]
    # The per-row weight agrees with the widened prefilter: an admitted row must
    # not be zeroed by ``scope_match`` a few lines later.
    assert result["items"][0]["scope_weight"] == 1.25
    assert result["ranking"]["eligible_count"] == 1
    conn.close()


def test_alias_expansion_never_admits_confidential_or_client_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_aliases(monkeypatch, {"project:coframe-brain": "project:coframe"})
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_belief(
        conn,
        belief_id="curated:coframe:visible",
        body="Trino sandbox schemas are the scratch tier for research queries.",
        scope=ScopeTag("project", "project:Coframe", provenance="test"),
    )
    _seed_belief(
        conn,
        belief_id="curated:client:hidden",
        body="PRIVATE_CLIENT_SENTINEL trino sandbox scratch tier.",
        scope=ScopeTag(
            "client",
            "client:Bihua",
            visibility="confidential",
            egress_policy="local_only",
            provenance="test",
        ),
    )
    _seed_belief(
        conn,
        belief_id="curated:confidential:hidden",
        body="PRIVATE_CONFIDENTIAL_SENTINEL trino sandbox scratch tier.",
        scope=ScopeTag(
            "project",
            "project:Secret Programme",
            visibility="confidential",
            egress_policy="local_only",
            provenance="test",
        ),
    )
    conn.commit()

    context = ScopeContext(project="Coframe Brain")
    admitted = matching_stored_scope_ids(conn, "current_beliefs", context.compatible_scope_ids())
    assert "client:Bihua" not in admitted
    assert "project:Secret Programme" not in admitted
    assert "project:Coframe" in admitted

    result = search_core_v1(conn, "trino sandbox scratch tier", context=context, limit=10)
    encoded = json.dumps(result)
    assert [item["belief_id"] for item in result["items"]] == ["curated:coframe:visible"]
    assert "PRIVATE_CLIENT_SENTINEL" not in encoded
    assert "PRIVATE_CONFIDENTIAL_SENTINEL" not in encoded
    conn.close()


def test_scope_tag_never_folds_so_stored_scopes_stay_byte_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_aliases(monkeypatch, {"project:coframe-brain": "project:coframe"})

    tag = ScopeTag.from_dict({"scope_type": "project", "scope_id": "project:Coframe Brain"})

    assert tag.scope_id == "project:Coframe Brain"
    assert ScopeTag("project", "project:Coframe Brain").scope_id == "project:Coframe Brain"
    # The fold happens one layer out, at the client argument boundary.
    folded = scope_from_arguments(
        {"scope": {"scope_type": "project", "scope_id": "project:Coframe Brain"}}
    )
    assert folded is not None
    assert folded.scope_id == "project:coframe"


def test_projection_replay_is_alias_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full refold must produce the same rows whatever the alias table says.

    The ledger is the authority. If replay folded stored scopes, an operator
    editing their alias table would silently rewrite history on the next rebuild.
    """
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_belief(
        conn,
        belief_id="curated:replay:fact",
        body="Trino sandbox schemas are the scratch tier for research queries.",
        scope=ScopeTag("project", "project:Coframe Brain", provenance="test"),
    )
    conn.commit()

    before = conn.execute(
        "SELECT scope_type, scope_id FROM current_beliefs WHERE belief_id='curated:replay:fact'"
    ).fetchone()
    assert str(before["scope_id"]) == "project:Coframe Brain"

    _set_aliases(monkeypatch, {"project:coframe-brain": "project:coframe"})
    project_core_v1(conn, full=True)
    after = conn.execute(
        "SELECT scope_type, scope_id FROM current_beliefs WHERE belief_id='curated:replay:fact'"
    ).fetchone()

    assert str(after["scope_id"]) == str(before["scope_id"])
    assert str(after["scope_type"]) == str(before["scope_type"])
    # And it is still reachable, because reach comes from the comparison, not
    # from a rewrite.
    assert (
        scope_match(
            ScopeTag("project", "project:Coframe Brain"),
            ScopeContext(project="Coframe Brain"),
        )
        == 1.25
    )
    conn.close()


def test_folding_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCBRAIN_SCOPES_FOLD_ENABLED", "false")

    assert fold_scope_component("Coframe Brain") == "Coframe Brain"
    assert ScopeContext(project="Coframe Brain").project == "Coframe Brain"
