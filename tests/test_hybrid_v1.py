from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ocbrain.core_v1 import (
    append_core_event,
    get_core_v1_belief,
    init_core_v1,
    search_core_v1,
)
from ocbrain.curation import apply_curated_manifest
from ocbrain.db import connect
from ocbrain.hybrid import build_vector_index, vector_status
from ocbrain.mcp_v1 import (
    build_context_v1,
    decide_proposal_v1,
    prepare_retrieval_packet_v1,
    search_v1,
)
from ocbrain.scope import ScopeContext, ScopeTag


def _seed_belief(
    conn,
    *,
    belief_id: str,
    body: str,
    egress_policy: str = "hosted_ok",
    attributes: dict | None = None,
) -> None:
    scope = ScopeTag(
        "project",
        "project:bountiful",
        visibility="internal",
        egress_policy=egress_policy,
        provenance="test",
    )
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
            "attributes": attributes or {"source_quality": 0.95},
        },
        writer="test",
    )
    decide_proposal_v1(
        conn,
        proposal_event_id=proposal,
        decision="approve",
        actor="test",
        edited_body=None,
        reason="test seed",
    )


def test_hybrid_dense_recall_and_stale_sidecar_fallback(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "core.sqlite"
    conn = connect(path)
    init_core_v1(conn)
    _seed_belief(conn, belief_id="curated:bountiful:citrus", body="Meyer lemons are ready.")
    _seed_belief(conn, belief_id="curated:bountiful:tomato", body="Tomatoes are available.")
    conn.commit()

    def fake_embed(texts, **_kwargs):
        result = []
        for text in texts:
            lowered = text.lower()
            if "citrus" in lowered or "lemon" in lowered:
                result.append([1.0, 0.0])
            else:
                result.append([0.0, 1.0])
        return result

    monkeypatch.setenv("OCBRAIN_EMBED_MODEL", "test-local")
    monkeypatch.setenv("OCBRAIN_EMBED_DIMENSIONS", "2")
    monkeypatch.setattr("ocbrain.hybrid.embed_texts", fake_embed)
    installed_digest = ["sha256:test-model-v1"]
    monkeypatch.setattr(
        "ocbrain.hybrid._ollama_model_metadata",
        lambda *_args, **_kwargs: {"digest": installed_digest[0]},
    )
    built = build_vector_index(path, model="test-local")
    assert built["rows"] == 2
    assert vector_status(path)["healthy"] is True

    result = search_core_v1(
        conn,
        "citrus harvest",
        context=ScopeContext(project="bountiful"),
        limit=2,
        delivery_target="hosted_model",
    )
    assert result["ranking"]["mode"] == "hybrid_rrf"
    assert result["items"][0]["belief_id"] == "curated:bountiful:citrus"

    monkeypatch.setenv("OCBRAIN_EMBED_DIMENSIONS", "1")
    dimension_drift = search_core_v1(
        conn,
        "citrus harvest",
        context=ScopeContext(project="bountiful"),
        limit=2,
        delivery_target="hosted_model",
    )
    assert dimension_drift["ranking"]["mode"] == "lexical"
    assert dimension_drift["ranking"]["dense_fallback"] == "vector_dimension_config_mismatch"
    monkeypatch.setenv("OCBRAIN_EMBED_DIMENSIONS", "2")

    installed_digest[0] = "sha256:test-model-v2"
    digest_drift = search_core_v1(
        conn,
        "citrus harvest",
        context=ScopeContext(project="bountiful"),
        limit=2,
        delivery_target="hosted_model",
    )
    assert digest_drift["ranking"]["mode"] == "lexical"
    assert digest_drift["ranking"]["dense_fallback"] == "vector_model_digest_mismatch"
    installed_digest[0] = "sha256:test-model-v1"

    _seed_belief(conn, belief_id="curated:bountiful:pear", body="Pears are ready.")
    conn.commit()
    stale = search_core_v1(
        conn,
        "pears",
        context=ScopeContext(project="bountiful"),
        limit=2,
        delivery_target="hosted_model",
    )
    assert stale["ranking"]["mode"] == "lexical"
    assert stale["ranking"]["dense_fallback"] == "vector_sidecar_stale"
    assert stale["items"][0]["belief_id"] == "curated:bountiful:pear"


def test_irrelevant_fresh_dense_candidate_cannot_outrank_exact_lexical_match(
    tmp_path: Path, monkeypatch
) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    relevant = "curated:bountiful:relevant"
    irrelevant = "curated:bountiful:irrelevant"
    _seed_belief(conn, belief_id=relevant, body="Verified orchard truth for matching.")
    _seed_belief(conn, belief_id=irrelevant, body="Completely unrelated recent note.")
    conn.execute(
        "UPDATE current_beliefs SET last_compiled_at='2010-01-01T00:00:00+00:00' WHERE belief_id=?",
        (relevant,),
    )
    conn.commit()

    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, **_kwargs: (
            [
                {"belief_id": irrelevant, "similarity": -1.0},
                {"belief_id": relevant, "similarity": 1.0},
            ],
            None,
        ),
    )
    result = search_core_v1(
        conn,
        "verified orchard truth",
        context=ScopeContext(project="bountiful"),
        limit=10,
        delivery_target="hosted_model",
    )
    assert result["items"][0]["belief_id"] == relevant
    assert irrelevant not in [item["belief_id"] for item in result["items"]]


def test_curated_manifest_is_hash_verified_and_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "truth.md"
    source.write_text("verified truth\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "ocbrain.curated-memory.v1",
                "project": "bountiful",
                "sources": [{"ref": "S1", "path": source.name, "sha256": digest}],
                "facts": [
                    {
                        "id": "B01",
                        "body": "Bountiful shares neighborhood food.",
                        "source_refs": ["S1"],
                        "visibility": "internal",
                        "egress_policy": "hosted_ok",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    first = apply_curated_manifest(conn, manifest_path)
    second = apply_curated_manifest(conn, manifest_path)
    assert first["applied"] == ["curated:bountiful:B01"]
    assert second["unchanged"] == ["curated:bountiful:B01"]
    assert conn.execute("SELECT COUNT(*) FROM current_beliefs WHERE serve=1").fetchone()[0] == 1
    assert (
        conn.execute("SELECT writer FROM brain_events ORDER BY rowid LIMIT 1").fetchone()[0]
        == "human-curated:operator"
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["facts"][0]["source_quality"] = 0.72
    manifest["facts"][0]["confidence"] = 0.83
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    changed = apply_curated_manifest(conn, manifest_path)
    assert changed["applied"] == ["curated:bountiful:B01"]
    current = get_core_v1_belief(conn, "curated:bountiful:B01")
    assert current is not None
    assert current["attributes"]["source_quality"] == 0.72
    assert current["confidence"] == 0.83
    assert apply_curated_manifest(conn, manifest_path)["unchanged"] == ["curated:bountiful:B01"]

    manifest["facts"].append(dict(manifest["facts"][0]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate curated fact id"):
        apply_curated_manifest(conn, manifest_path)
    manifest["facts"].pop()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    source.write_text("replacement truth\n", encoding="utf-8")
    manifest["sources"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest["facts"][0]["body"] = "Updated Bountiful neighborhood food truth."
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    replaced = apply_curated_manifest(conn, manifest_path)
    assert replaced["applied"] == ["curated:bountiful:B01"]
    linked = conn.execute(
        "SELECT eo.body FROM belief_evidence be "
        "JOIN evidence_objects eo ON eo.evidence_id=be.evidence_id "
        "WHERE be.belief_id='curated:bountiful:B01'"
    ).fetchall()
    assert [row["body"] for row in linked] == ["Updated Bountiful neighborhood food truth."]
    assert conn.execute("SELECT count(*) FROM evidence_objects").fetchone()[0] == 2

    source.write_text("changed truth\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        apply_curated_manifest(conn, manifest_path)


def test_hosted_delivery_excludes_local_only_before_ranking(tmp_path: Path) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_belief(
        conn,
        belief_id="curated:bountiful:private",
        body="Secret tomato phrase.",
        egress_policy="local_only",
    )
    _seed_belief(
        conn,
        belief_id="curated:bountiful:safe",
        body="Safe tomato phrase.",
        egress_policy="hosted_ok",
    )
    conn.commit()
    result = search_core_v1(
        conn,
        "tomato phrase",
        context=ScopeContext(project="bountiful"),
        limit=10,
        delivery_target="hosted_model",
    )
    assert [item["belief_id"] for item in result["items"]] == ["curated:bountiful:safe"]


def test_context_packet_has_real_serialized_budget_and_no_guessed_conflicts(
    tmp_path: Path,
) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    for index in range(30):
        _seed_belief(
            conn,
            belief_id=f"curated:bountiful:long-{index:02d}",
            body=f"matching orchard fact {index} " + ("verified detail " * 400),
        )
    _seed_belief(
        conn,
        belief_id="curated:bountiful:positive",
        body="Matching exchange state is ready for neighbors.",
    )
    _seed_belief(
        conn,
        belief_id="curated:bountiful:negative",
        body="Matching exchange state is not ready for neighbors.",
    )
    conn.commit()
    packet, handles = build_context_v1(
        conn,
        "matching exchange orchard state neighbors",
        context=ScopeContext(project="bountiful"),
        limit=50,
        cross_scope=False,
        delivery_target="hosted_model",
    )
    encoded = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()
    assert len(encoded) <= 32_000
    assert packet["coverage"]["serialized_bytes"] <= 32_000
    assert packet["coverage"]["trimmed_for_packet_limit"] > 0
    assert all(len(item["excerpt"]) <= 1_600 for item in packet["items"])
    assert packet["contradictions"] == []
    first_trimmed = packet["coverage"]["trimmed_for_packet_limit"]
    prepared, _prepared_handles = prepare_retrieval_packet_v1(packet, handles)
    assert prepared["coverage"]["trimmed_for_packet_limit"] >= first_trimmed

    search = search_v1(
        conn,
        "matching exchange orchard state neighbors " * 5_000,
        context=ScopeContext(project="bountiful", runtime="test"),
        limit=50,
        cross_scope=False,
        delivery_target="hosted_model",
    )
    search_encoded = json.dumps(search, sort_keys=True, separators=(",", ":")).encode()
    assert len(search["query"]) == 4_000
    assert len(search_encoded) <= 32_000
    assert search["coverage"]["serialized_bytes"] == len(search_encoded)


def test_context_packages_only_explicit_contradictions(tmp_path: Path) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_belief(
        conn,
        belief_id="curated:bountiful:old-rule",
        body="Garden inventory reserves on basket add.",
        attributes={
            "source_quality": 0.9,
            "contradicts": ["curated:bountiful:new-rule"],
        },
    )
    _seed_belief(
        conn,
        belief_id="curated:bountiful:new-rule",
        body="Garden inventory decrements only on successful exchange completion.",
    )
    conn.commit()
    packet, _handles = build_context_v1(
        conn,
        "garden inventory exchange",
        context=ScopeContext(project="bountiful"),
        limit=10,
        cross_scope=False,
        delivery_target="hosted_model",
    )
    assert packet["contradictions"] == [
        {
            "belief_id": "curated:bountiful:old-rule",
            "other_belief_id": "curated:bountiful:new-rule",
            "reason": "explicit_compiler_metadata",
            "evidence_ids": [],
        }
    ]
