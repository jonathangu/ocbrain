"""The pre-write duplicate gate: a restatement must not become a new key.

Supersession can only ever replace a belief filed under the same key -- the
transaction copies the predecessor's key onto the successor, and the curator's own
rationale says it "recompiled key <k>". So a fact restated under a *new* slug was
uncollapsible by construction, and the live corpus proved it: 344 serving beliefs
carrying 344 distinct keys, perfect uniqueness, with 35 same-scope clusters at
cosine 0.88 covering 98 of them. Five Plane-1 recency beliefs were compiled on one
day under five keys, two of which differ by a single hyphen.

These tests pin the three arms of the gate and, more importantly, what it does when
it cannot see: an unavailable embedder pends the claim rather than admitting it,
with two declared exemptions for an install that has no sidecar at all.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import ocbrain.curator
from ocbrain.compact import DEFAULT_COSINE_FLOOR
from ocbrain.config import CuratorConfig
from ocbrain.core_v1 import init_core_v1
from ocbrain.curator import (
    DEFAULT_CURRENT_TTL_DAYS,
    DEFAULT_MEASURED_TTL_DAYS,
    DEFAULT_VOLATILE_TTL_DAYS,
    DUPLICATE_GATE_EXEMPT_REASONS,
    NEAR_DUPLICATE_COSINE,
    apply_claims,
    curator_runtime_settings,
    fold_key,
    near_duplicate_neighbor,
    serving_key_row,
)
from ocbrain.db import connect
from ocbrain.hybrid import (
    DEFAULT_DOCUMENT_EMBED_BUDGET,
    VECTOR_SCHEMA_VERSION,
    vector_db_path,
)

PROJECT = "test"


def _core(tmp_path: Path) -> sqlite3.Connection:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    return conn


def _claim(key: str, body: str, *, confidence: float = 0.9) -> dict:
    return {
        "key": key,
        "title": key.replace("-", " ")[:80],
        "body": body,
        "category": "system",
        "lifecycle": "durable",
        "confidence": confidence,
        "evidence_ids": [],
    }


def _serving(conn) -> list[str]:
    return [
        str(row["belief_id"])
        for row in conn.execute(
            "SELECT belief_id FROM current_beliefs WHERE serve=1 AND status='current'"
        )
    ]


def _keys(conn) -> list[str]:
    return sorted(
        str(row[0])
        for row in conn.execute(
            "SELECT json_extract(attributes_json, '$.key') FROM current_beliefs "
            "WHERE serve=1 AND status='current'"
        )
    )


def _proposals(conn) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM brain_events WHERE kind='compilation_proposed'"
        ).fetchone()[0]
    )


def _write_broken_sidecar(tmp_path: Path) -> Path:
    """A sidecar that exists and cannot be trusted: the state the gate must pend on.

    Distinct from "no sidecar", which is a declared exemption. The schema version
    is wrong, which is one of the eleven typed reasons the reader can return, and
    it is reached before any network call so the test needs no embedder.
    """
    path = vector_db_path(tmp_path / "core.sqlite")
    sidecar = sqlite3.connect(path)
    sidecar.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE belief_vectors(belief_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL,
          vector BLOB NOT NULL);
        """
    )
    sidecar.execute("INSERT INTO meta VALUES ('schema_version', 'ocbrain.vectors.v0')")
    sidecar.commit()
    sidecar.close()
    assert VECTOR_SCHEMA_VERSION != "ocbrain.vectors.v0"
    return path


def _stub_document_neighbors(monkeypatch, *, similarity=None, unavailable=None, uncovered=0):
    """Answer the document-side reader without a local embedding server."""

    def fake(conn, text, *, candidate_ids, limit=5, embed_budget=32, cache=None):
        coverage = {
            "candidates": len(list(candidate_ids)),
            "reused": 0,
            "embedded": 0,
            "uncovered": uncovered,
        }
        if unavailable is not None:
            return [], unavailable, coverage
        neighbors = [
            {"belief_id": belief_id, "similarity": similarity}
            for belief_id in candidate_ids
        ]
        return neighbors[:limit], None, coverage

    monkeypatch.setattr(ocbrain.curator, "document_neighbors", fake)


# --------------------------------------------------------------------------- #
# the constants
# --------------------------------------------------------------------------- #


def test_the_gate_and_the_compactor_share_one_floor() -> None:
    """The pre-write gate and the after-the-fact compactor must mean the same thing.

    A claim the gate admits and the compactor then proposes retiring is a gate
    that moved the work rather than doing it, so these two numbers are pinned
    equal rather than merely documented as similar.
    """
    assert NEAR_DUPLICATE_COSINE == DEFAULT_COSINE_FLOOR == 0.88


def test_fold_key_collapses_separator_noise_and_nothing_else() -> None:
    """Measured on the live corpus: 344 exact keys fold to 343, collapsing one pair."""
    assert fold_key("plane1-recency-gate-result") == fold_key("plane-1-recency-gate-result")
    assert fold_key("lakehouse-org-id-gap") == fold_key("lakehouse-orgid-gap")
    # Different facts stay different. A fold that merged these would silently
    # overwrite one belief with another.
    assert fold_key("plane1-recency-gate-result") != fold_key("plane1-recency-blend-gated")
    assert fold_key("hermes-model-config") != fold_key("hermes-model-routing")
    assert fold_key("") == ""


# --------------------------------------------------------------------------- #
# arm 1: the key fold, which needs no embedding at all
# --------------------------------------------------------------------------- #


def test_a_hyphen_variant_key_updates_the_belief_instead_of_minting_a_second(
    tmp_path: Path,
) -> None:
    """The smoking gun, reproduced: two keys one hyphen apart are one fact."""
    conn = _core(tmp_path)
    first = apply_claims(
        conn,
        [_claim("plane1-recency-gate-result", "The Plane-1 recency blend gate did not pass.")],
        model="test",
        project=PROJECT,
    )
    assert len(first["applied"]) == 1
    assert len(_serving(conn)) == 1

    second = apply_claims(
        conn,
        [
            _claim(
                "plane-1-recency-gate-result",
                "The Plane-1 recency blend gate failed on the held-out split.",
            )
        ],
        model="test",
        project=PROJECT,
    )

    # One fact, one serving belief. Before the fold this minted a second.
    assert second["applied"] == []
    assert len(second["superseded"]) == 1
    assert len(_serving(conn)) == 1
    assert _keys(conn) == ["plane-1-recency-gate-result"]


def test_serving_key_row_prefers_the_exact_spelling(tmp_path: Path) -> None:
    """The fold only ever answers where exact matching found nothing.

    The pair this reads is the one the live corpus already holds and the gate
    can no longer create, so it is seeded onto the projection directly: this
    helper is a reader, and what is being pinned is which of two candidate rows
    it returns.

    Each assertion below names one row, never a set of the possible answers. An
    earlier draft of this test accepted either candidate on the folded lookup,
    which made it pass under a fully reversed ORDER BY -- a vacuous assertion in
    the file added to close a vacuity defect.
    """
    conn = _core(tmp_path)
    apply_claims(
        conn,
        [_claim("plane-1-recency-gate-result", "The gate result, one spelling.")],
        model="test",
        project=PROJECT,
    )
    exact_id = _serving(conn)[0]
    last_event_id = str(
        conn.execute(
            "SELECT last_event_id FROM current_beliefs WHERE belief_id=?", (exact_id,)
        ).fetchone()[0]
    )

    def _seed_row(belief_id: str, key: str, *, scope_type: str, scope_id: str, at: str) -> None:
        conn.execute(
            "INSERT INTO current_beliefs (belief_id, body, belief_type, attributes_json, "
            "scope_type, scope_id, visibility, egress_policy, confidence, evidence_ids, "
            "status, serve, pinned, last_event_id, last_compiled_at) "
            "VALUES (?, ?, 'wiki_fact', ?, ?, ?, 'internal', "
            "'local_only', 0.9, '[]', 'current', 1, 0, ?, ?)",
            (
                belief_id,
                f"The gate result, spelled {key}.",
                json.dumps({"key": key}),
                scope_type,
                scope_id,
                last_event_id,
                at,
            ),
        )

    # A project-scoped sibling, newer than the exact row.
    _seed_row(
        "belief_folded_sibling",
        "plane1-recency-gate-result",
        scope_type="project",
        scope_id="project:test",
        at="2099-01-01T00:00:00+00:00",
    )
    conn.commit()

    assert str(serving_key_row(conn, "plane-1-recency-gate-result")["belief_id"]) == exact_id
    # ... and the folded lookup answers only because no exact row exists. Of the
    # two folded candidates it takes the most recently compiled one.
    assert (
        str(serving_key_row(conn, "plane1recencygateresult")["belief_id"])
        == "belief_folded_sibling"
    )

    # Doctrine outranks recency: a global-scoped row wins even when it is the
    # oldest of the three.
    _seed_row(
        "belief_folded_doctrine",
        "plane_1_recency_gate_result",
        scope_type="global",
        scope_id="global:doctrine",
        at="1999-01-01T00:00:00+00:00",
    )
    conn.commit()
    assert (
        str(serving_key_row(conn, "plane1recencygateresult")["belief_id"])
        == "belief_folded_doctrine"
    )

    assert serving_key_row(conn, "an-unrelated-key") is None


# --------------------------------------------------------------------------- #
# arm 2: the semantic gate
# --------------------------------------------------------------------------- #


def test_a_near_duplicate_under_a_new_key_supersedes_instead_of_minting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The corpus grew 60 of its 63 restatement copies exactly this way."""
    conn = _core(tmp_path)
    apply_claims(
        conn,
        [_claim("t1-vs-t1t2-eval-result", "T2 features were net-negative out of sample.")],
        model="test",
        project=PROJECT,
    )
    assert len(_serving(conn)) == 1

    _stub_document_neighbors(monkeypatch, similarity=0.93)
    second = apply_claims(
        conn,
        [
            _claim(
                "t1-t2-feature-eval",
                "Out of sample, the T2 feature set did not beat T1 and was net-negative.",
            )
        ],
        model="test",
        project=PROJECT,
    )

    assert second["applied"] == []
    assert len(second["superseded"]) == 1
    assert len(second["duplicate_routed"]) == 1
    assert second["duplicate_routed"][0]["similarity"] == 0.93
    assert len(_serving(conn)) == 1


def test_a_claim_below_the_floor_is_still_a_new_fact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is a duplicate test, not a topic test. Two facts about one subject stay two."""
    conn = _core(tmp_path)
    apply_claims(
        conn,
        [_claim("t1-vs-t1t2-eval-result", "T2 features were net-negative out of sample.")],
        model="test",
        project=PROJECT,
    )
    _stub_document_neighbors(monkeypatch, similarity=NEAR_DUPLICATE_COSINE - 0.01)
    second = apply_claims(
        conn,
        [_claim("t1-t2-runtime-cost", "The T2 feature build adds 40 minutes per refit.")],
        model="test",
        project=PROJECT,
    )
    assert len(second["applied"]) == 1
    assert second["duplicate_routed"] == []
    assert len(_serving(conn)) == 2


def test_an_uncovered_candidate_is_unavailability_not_a_clean_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A comparison that skipped a candidate cannot report "no duplicate"."""
    conn = _core(tmp_path)
    apply_claims(
        conn,
        [_claim("first-fact", "The first fact about the pipeline.")],
        model="test",
        project=PROJECT,
    )
    _stub_document_neighbors(monkeypatch, similarity=0.10, uncovered=1)
    match, unavailable = near_duplicate_neighbor(
        conn, body="An unrelated second fact.", candidates=_serving(conn)
    )
    assert match is None
    assert unavailable == "candidates_uncovered:1"


# --------------------------------------------------------------------------- #
# arm 3: what happens when the gate cannot see
# --------------------------------------------------------------------------- #


def test_an_unreadable_sidecar_pends_the_claim_rather_than_admitting_it(
    tmp_path: Path,
) -> None:
    """Fail-closed. The gate that disappears when its instrument dies is the defect."""
    conn = _core(tmp_path)
    apply_claims(
        conn,
        [_claim("first-fact", "The first fact about the pipeline.")],
        model="test",
        project=PROJECT,
    )
    _write_broken_sidecar(tmp_path)

    result = apply_claims(
        conn,
        [_claim("second-fact", "A second fact the gate cannot check.")],
        model="test",
        project=PROJECT,
    )

    assert result["applied"] == []
    assert len(result["pended_unverified"]) == 1
    assert result["pended_unverified"][0]["reason"] == "vector_schema_mismatch"
    # Nothing is lost: the claim is in the ledger as an undecided proposal, and
    # it is not serving.
    assert len(_serving(conn)) == 1
    assert _proposals(conn) == 2
    pended = json.loads(
        conn.execute(
            "SELECT body_json FROM brain_events WHERE kind='compilation_proposed' "
            "ORDER BY event_seq DESC LIMIT 1"
        ).fetchone()[0]
    )
    assert pended["attributes"]["duplicate_gate"] == "vector_schema_mismatch"
    assert (
        conn.execute("SELECT COUNT(*) FROM brain_events WHERE kind='compilation_decided'")
        .fetchone()[0]
        == 1
    )


def test_an_identical_re_derivation_of_a_pended_claim_writes_nothing(tmp_path: Path) -> None:
    """A proposal does not change the input that produced it, so it must not re-pend.

    This is the loop that took the live pending ledger to 312 proposals over 33
    beliefs in nineteen hours. The pend path is a producer too, and it gets the
    same dedup.
    """
    conn = _core(tmp_path)
    apply_claims(
        conn, [_claim("first-fact", "The first fact.")], model="test", project=PROJECT
    )
    _write_broken_sidecar(tmp_path)
    claim = _claim("second-fact", "A second fact the gate cannot check.")

    first = apply_claims(conn, [claim], model="test", project=PROJECT)
    assert len(first["pended_unverified"]) == 1
    proposals_after_first = _proposals(conn)

    second = apply_claims(conn, [claim], model="test", project=PROJECT)
    assert second["pended_unverified"] == []
    assert len(second["pending_deduped"]) == 1
    assert _proposals(conn) == proposals_after_first


def test_the_admit_fallback_restores_the_previous_behaviour_exactly(tmp_path: Path) -> None:
    """An operator who would rather grow the corpus dirty than stall gets that choice."""
    conn = _core(tmp_path)
    apply_claims(
        conn, [_claim("first-fact", "The first fact.")], model="test", project=PROJECT
    )
    _write_broken_sidecar(tmp_path)

    result = apply_claims(
        conn,
        [_claim("second-fact", "A second fact the gate cannot check.")],
        model="test",
        project=PROJECT,
        duplicate_gate_fallback="admit",
    )
    assert len(result["applied"]) == 1
    assert result["pended_unverified"] == []
    assert len(_serving(conn)) == 2


def test_no_sidecar_at_all_is_a_declared_exemption(tmp_path: Path) -> None:
    """An install that never opted into semantic dedup still compiles.

    The exemptions are declared and there are two of them, rather than the
    fail-open list being everything nobody thought to enumerate.
    """
    assert DUPLICATE_GATE_EXEMPT_REASONS == {"core_path_unavailable", "vector_sidecar_missing"}
    conn = _core(tmp_path)
    apply_claims(
        conn, [_claim("first-fact", "The first fact.")], model="test", project=PROJECT
    )
    result = apply_claims(
        conn,
        [_claim("second-fact", "A second, genuinely different fact.")],
        model="test",
        project=PROJECT,
    )
    assert len(result["applied"]) == 1
    assert result["pended_unverified"] == []


def test_the_first_claim_in_an_empty_scope_has_nothing_to_be_a_duplicate_of(
    tmp_path: Path,
) -> None:
    """No candidates is not an unavailable gate. A brand-new scope must compile."""
    conn = _core(tmp_path)
    _write_broken_sidecar(tmp_path)
    result = apply_claims(
        conn, [_claim("first-fact", "The first fact.")], model="test", project=PROJECT
    )
    assert len(result["applied"]) == 1
    assert result["pended_unverified"] == []


# --------------------------------------------------------------------------- #
# the settings in front of the gate, and the budget that decides whether it can
# answer at all
# --------------------------------------------------------------------------- #


def test_an_unreadable_config_falls_back_to_the_shipped_settings_not_to_fail_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact case `curator_runtime_settings` cites, and the one it must survive.

    `load_config` raising is what the broad `except` is for -- "config problems
    must not break curation". The literal it returns on that path is a second,
    independent copy of every default, and it was untested: flipping the
    duplicate-gate fallback in it to `admit` made a broken config file fail OPEN
    with nothing in the suite noticing. Asserting the whole dict, rather than
    the one key, is what stops the next default drifting there unseen.
    """
    import ocbrain.config

    def explode(*_args, **_kwargs):
        raise OSError("config file is not readable")

    monkeypatch.setattr(ocbrain.config, "load_config", explode)
    assert curator_runtime_settings() == {
        "duplicate_gate_fallback": "pend",
        "volatility_ttl": True,
        "volatile_ttl_days": DEFAULT_VOLATILE_TTL_DAYS,
        "measured_ttl_days": DEFAULT_MEASURED_TTL_DAYS,
        "document_embed_budget": DEFAULT_DOCUMENT_EMBED_BUDGET,
        "current_ttl_days": DEFAULT_CURRENT_TTL_DAYS,
    }


def test_a_broken_config_still_pends_an_unverifiable_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the consequence of that literal, measured rather than read."""
    import ocbrain.config

    conn = _core(tmp_path)
    apply_claims(
        conn, [_claim("first-fact", "The first fact.")], model="test", project=PROJECT
    )
    _write_broken_sidecar(tmp_path)

    def explode(*_args, **_kwargs):
        raise OSError("config file is not readable")

    monkeypatch.setattr(ocbrain.config, "load_config", explode)
    result = apply_claims(
        conn,
        [_claim("second-fact", "A second fact the gate cannot check.")],
        model="test",
        project=PROJECT,
    )
    assert result["applied"] == []
    assert len(result["pended_unverified"]) == 1


def test_the_configured_embed_budget_reaches_the_readers_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`curator.document_embed_budget` is the knob on the availability cliff.

    Past the budget the reader reports candidates as `uncovered`, the gate
    returns `candidates_uncovered:N`, and that reason is deliberately NOT an
    exemption -- so on the shipped `pend` fallback a cycle larger than the
    budget pends the rest of its claims. An operator whose cycles are larger
    needs to be able to raise it, which means the number has to travel from the
    config all the way to the reader.
    """
    seen: list[int] = []

    def fake(conn, text, *, candidate_ids, limit=5, embed_budget=None, cache=None):
        seen.append(embed_budget)
        ids = list(candidate_ids)
        return (
            [],
            None,
            {"candidates": len(ids), "reused": 0, "embedded": 0, "uncovered": len(ids)},
        )

    monkeypatch.setattr(ocbrain.curator, "document_neighbors", fake)
    monkeypatch.setattr(
        ocbrain.curator,
        "curator_runtime_settings",
        lambda: {
            "duplicate_gate_fallback": "pend",
            "volatility_ttl": True,
            "volatile_ttl_days": 14,
            "measured_ttl_days": 45,
            "document_embed_budget": 7,
            "current_ttl_days": 90,
        },
    )

    conn = _core(tmp_path)
    apply_claims(
        conn, [_claim("first-fact", "The first fact.")], model="test", project=PROJECT
    )
    result = apply_claims(
        conn,
        [_claim("second-fact", "A second, genuinely different fact.")],
        model="test",
        project=PROJECT,
    )

    assert seen == [7]
    # And the cliff itself: an uncovered candidate pends rather than compiles.
    assert result["applied"] == []
    assert result["pended_unverified"][0]["reason"] == "candidates_uncovered:1"


def test_the_shipped_embed_budget_covers_a_full_max_beliefs_cycle() -> None:
    """The budget has to exceed what one cycle can write, or the gate stalls itself.

    `--max-beliefs` is capped at 40 in the script and defaults to 24; the
    candidates needing an on-demand embedding are exactly the beliefs written
    since the last sidecar build, and `brain-promote.sh` rebuilds the sidecar
    only after curation. A budget below the cycle's own output is a gate that
    switches itself off partway through every run.
    """
    assert DEFAULT_DOCUMENT_EMBED_BUDGET >= 24
    assert CuratorConfig().document_embed_budget == DEFAULT_DOCUMENT_EMBED_BUDGET


def test_the_embed_budget_config_field_actually_reaches_the_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A knob is only a knob if turning it moves something.

    `curator_runtime_settings` is where the config file becomes a number the
    gate uses; a field the loader never reads is a config entry that documents a
    control the operator does not have.
    """
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"curator": {"document_embed_budget": 9}}), encoding="utf-8"
    )
    monkeypatch.setenv("OCBRAIN_CONFIG", str(config_path))
    assert curator_runtime_settings()["document_embed_budget"] == 9

    config_path.write_text(
        json.dumps({"curator": {"document_embed_budget": -4}}), encoding="utf-8"
    )
    assert curator_runtime_settings()["document_embed_budget"] == 0
