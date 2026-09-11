from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ocbrain.core_v1 import (
    KEYWORD_QUERY_HINT,
    _retrieval_feedback_scores,
    append_core_event,
    get_core_v1_belief,
    init_core_v1,
    search_core_v1,
)
from ocbrain.curation import apply_curated_manifest
from ocbrain.db import connect
from ocbrain.hybrid import (
    DEFAULT_EMBED_DOCUMENT_BYTES,
    LocalEmbeddingUnavailable,
    _bounded_embedding_text,
    _corpus_fingerprint,
    _document_text,
    _serving_rows,
    build_vector_index,
    embed_missing_beliefs,
    semantic_neighbors,
    vector_db_path,
    vector_status,
)
from ocbrain.mcp import handle_request
from ocbrain.mcp_v1 import (
    build_context_v1,
    decide_proposal_v1,
    finish_vector_refresh,
    prepare_retrieval_packet_v1,
    search_v1,
    supersede_v1,
)
from ocbrain.scope import ScopeContext, ScopeTag


def _seed_belief(
    conn,
    *,
    belief_id: str,
    body: str,
    egress_policy: str = "hosted_ok",
    project: str = "bountiful",
    visibility: str = "internal",
    scope: ScopeTag | None = None,
    attributes: dict | None = None,
    belief_type: str = "curated_fact",
) -> None:
    scope = scope or ScopeTag(
        "project",
        f"project:{project}",
        visibility=visibility,
        egress_policy=egress_policy,
        provenance="test",
    )
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


def _mcp_call(name: str, arguments: dict, *, request_id: int) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def _mcp_payload(response: dict) -> dict:
    assert "error" not in response, response
    return json.loads(response["result"]["content"][0]["text"])


def test_dense_document_text_is_bounded_and_preserves_head_and_tail() -> None:
    short = "short belief"
    assert _document_text({"body": short}) == short

    long = "HEAD-" + ("🧠" * 2_000) + "-TAIL"
    bounded = _document_text({"body": long})
    assert len(bounded.encode("utf-8")) <= DEFAULT_EMBED_DOCUMENT_BYTES
    assert bounded.startswith("HEAD-")
    assert bounded.endswith("-TAIL")
    assert "middle omitted for local embedding" in bounded
    bounded_query = _bounded_embedding_text("Instruct: retrieve\nQuery: " + ("漢" * 2_000))
    assert len(bounded_query.encode("utf-8")) <= DEFAULT_EMBED_DOCUMENT_BYTES


def test_vector_build_cleans_temporary_sidecar_when_interrupted(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "core.sqlite"
    conn = connect(path)
    init_core_v1(conn)
    _seed_belief(conn, belief_id="curated:test:interrupt", body="bounded belief")
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        "ocbrain.hybrid._ollama_model_metadata",
        lambda *_args, **_kwargs: {"digest": "sha256:test-model"},
    )

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("ocbrain.hybrid.embed_texts", interrupted)
    with pytest.raises(KeyboardInterrupt):
        build_vector_index(path, model="test-local")
    assert not list(tmp_path.glob(".core-vectors.sqlite.*.tmp"))


def _local_dense_arm(monkeypatch) -> list[str]:
    monkeypatch.setenv("OCBRAIN_EMBED_MODEL", "test-local")
    monkeypatch.setenv("OCBRAIN_EMBED_DIMENSIONS", "2")
    monkeypatch.setattr(
        "ocbrain.hybrid.embed_texts",
        lambda texts, **_kwargs: [
            [1.0, 0.0]
            if "citrus" in str(text).lower() or "lemon" in str(text).lower()
            else [0.0, 1.0]
            for text in texts
        ],
    )
    installed_digest = ["sha256:test-model-v1"]
    monkeypatch.setattr(
        "ocbrain.hybrid._ollama_model_metadata",
        lambda *_args, **_kwargs: {"digest": installed_digest[0]},
    )
    return installed_digest


def _search(conn, query: str, *, limit: int = 2) -> dict:
    return search_core_v1(
        conn,
        query,
        context=ScopeContext(project="bountiful"),
        limit=limit,
        delivery_target="hosted_model",
    )


def test_stale_fingerprint_with_high_coverage_still_serves_dense(
    tmp_path: Path, monkeypatch
) -> None:
    """A corpus that moved on must not switch the dense arm off.

    The sidecar rebuilds hourly and the corpus changes every few minutes, so a
    whole-corpus fingerprint mismatch was the normal state, not the exception:
    ranking refused and every retrieval ran lexical-only. What the ranker needs
    is the share of serving beliefs it can still answer for.
    """
    path = tmp_path / "core.sqlite"
    conn = connect(path)
    init_core_v1(conn)
    _seed_belief(conn, belief_id="curated:bountiful:citrus", body="Citrus lemons are ready.")
    _seed_belief(conn, belief_id="curated:bountiful:tomato", body="Tomatoes are available.")
    _seed_belief(conn, belief_id="curated:bountiful:pear", body="Pears are ready.")
    conn.commit()

    installed_digest = _local_dense_arm(monkeypatch)
    built = build_vector_index(path, model="test-local")
    assert built["rows"] == 3
    assert built["embedded_rows"] == 3
    assert built["reused_rows"] == 0
    assert vector_status(path)["healthy"] is True

    append_core_event(
        conn,
        "retrieval_used",
        {"retrieval_id": "retrieval:test", "outcome": "used"},
        writer="test",
    )
    conn.commit()
    after_ledger_only_event = vector_status(path)
    assert after_ledger_only_event["event_fresh"] is False
    assert after_ledger_only_event["corpus_fresh"] is True
    assert after_ledger_only_event["healthy"] is True

    result = _search(conn, "citrus harvest")
    assert result["ranking"]["mode"] == "hybrid_rrf"
    assert result["items"][0]["belief_id"] == "curated:bountiful:citrus"
    assert result["ranking"]["dense_coverage"] == 1.0
    assert result["ranking"]["dense_stale_rows"] == 0

    monkeypatch.setenv("OCBRAIN_EMBED_DIMENSIONS", "1")
    dimension_drift = _search(conn, "citrus harvest")
    assert dimension_drift["ranking"]["mode"] == "hybrid_rrf"
    assert dimension_drift["ranking"]["dense_fallback"] is None
    dimension_status = vector_status(path)
    assert dimension_status["configured_dimensions_differ"] is True
    assert dimension_status["healthy"] is True
    monkeypatch.setenv("OCBRAIN_EMBED_DIMENSIONS", "2")

    installed_digest[0] = "sha256:test-model-v2"
    digest_drift = _search(conn, "citrus harvest")
    assert digest_drift["ranking"]["mode"] == "lexical"
    assert digest_drift["ranking"]["dense_fallback"] == "vector_model_digest_mismatch"
    installed_digest[0] = "sha256:test-model-v1"

    conn.execute(
        "UPDATE current_beliefs SET body=? WHERE belief_id=?",
        ("Tomatoes are ripe.", "curated:bountiful:tomato"),
    )
    conn.commit()
    drifted = _search(conn, "citrus harvest")
    assert drifted["ranking"]["mode"] == "hybrid_rrf"
    assert drifted["ranking"]["dense_fallback"] is None
    assert drifted["ranking"]["dense_serving_rows"] == 3
    assert drifted["ranking"]["dense_usable_rows"] == 2
    assert drifted["ranking"]["dense_stale_rows"] == 1
    assert drifted["ranking"]["dense_coverage"] == pytest.approx(2 / 3)
    assert drifted["items"][0]["belief_id"] == "curated:bountiful:citrus"

    rebuilt = build_vector_index(path, model="test-local")
    assert rebuilt["rows"] == 3
    assert rebuilt["embedded_rows"] == 1
    assert rebuilt["reused_rows"] == 2
    assert vector_status(path)["healthy"] is True


def test_sparse_sidecar_falls_back_with_a_typed_reason(tmp_path: Path, monkeypatch) -> None:
    """Below the coverage floor the dense arm stands down and says why.

    Answering from a remnant of the corpus is worse than answering lexically:
    the nearest neighbour of a belief the sidecar never saw is whatever it did
    see, and that is a confident wrong answer rather than a missing one.
    """
    path = tmp_path / "core.sqlite"
    conn = connect(path)
    init_core_v1(conn)
    _seed_belief(conn, belief_id="curated:bountiful:citrus", body="Citrus lemons are ready.")
    _seed_belief(conn, belief_id="curated:bountiful:tomato", body="Tomatoes are available.")
    _seed_belief(conn, belief_id="curated:bountiful:pear", body="Pears are ready.")
    conn.commit()
    _local_dense_arm(monkeypatch)
    build_vector_index(path, model="test-local")

    conn.execute(
        "UPDATE current_beliefs SET body='Tomatoes are ripe.' "
        "WHERE belief_id='curated:bountiful:tomato'"
    )
    conn.execute(
        "UPDATE current_beliefs SET body='Pears are ripe.' "
        "WHERE belief_id='curated:bountiful:pear'"
    )
    conn.commit()

    sparse = _search(conn, "citrus harvest")
    assert sparse["ranking"]["mode"] == "lexical"
    assert sparse["ranking"]["dense_fallback"] == "vector_sidecar_sparse"
    assert sparse["ranking"]["dense_serving_rows"] == 3
    assert sparse["ranking"]["dense_usable_rows"] == 1
    assert sparse["ranking"]["dense_stale_rows"] == 2
    assert sparse["ranking"]["dense_coverage"] == pytest.approx(1 / 3)


def test_embed_missing_beliefs_fills_the_gap_and_updates_meta(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "core.sqlite"
    conn = connect(path)
    init_core_v1(conn)
    _seed_belief(conn, belief_id="curated:bountiful:citrus", body="Citrus lemons are ready.")
    _seed_belief(conn, belief_id="curated:bountiful:tomato", body="Tomatoes are available.")
    conn.commit()
    _local_dense_arm(monkeypatch)
    build_vector_index(path, model="test-local")

    monkeypatch.setenv("OCBRAIN_RETRIEVAL_EMBED_ON_WRITE", "0")
    _seed_belief(conn, belief_id="curated:bountiful:pear", body="Pears are ready.")
    conn.commit()
    monkeypatch.delenv("OCBRAIN_RETRIEVAL_EMBED_ON_WRITE")
    assert vector_status(path)["coverage"]["dense_stale_rows"] == 1

    refresh = embed_missing_beliefs(conn)
    assert refresh == {"embedded": 1, "remaining": 0, "skipped_reason": None}

    sidecar = sqlite3.connect(vector_db_path(path))
    sidecar.row_factory = sqlite3.Row
    try:
        stored = sidecar.execute(
            "SELECT content_hash FROM belief_vectors WHERE belief_id=?",
            ("curated:bountiful:pear",),
        ).fetchone()
        meta = {str(row[0]): str(row[1]) for row in sidecar.execute("SELECT key, value FROM meta")}
    finally:
        sidecar.close()
    assert stored is not None
    assert str(stored["content_hash"]) == hashlib.sha256(b"Pears are ready.").hexdigest()
    assert meta["corpus_sha256"] == _corpus_fingerprint(_serving_rows(conn))
    assert meta["corpus_rows"] == "3"
    assert meta["rows"] == "3"
    status = vector_status(path)
    assert status["coverage"]["dense_coverage"] == 1.0
    assert status["healthy"] is True


def test_embed_missing_beliefs_never_raises_when_ollama_is_down(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "core.sqlite"
    conn = connect(path)
    init_core_v1(conn)
    _seed_belief(conn, belief_id="curated:bountiful:citrus", body="Citrus lemons are ready.")
    conn.commit()
    _local_dense_arm(monkeypatch)
    build_vector_index(path, model="test-local")

    monkeypatch.setenv("OCBRAIN_RETRIEVAL_EMBED_ON_WRITE", "0")
    _seed_belief(conn, belief_id="curated:bountiful:pear", body="Pears are ready.")
    conn.commit()
    monkeypatch.delenv("OCBRAIN_RETRIEVAL_EMBED_ON_WRITE")

    def down(*_args, **_kwargs):
        raise LocalEmbeddingUnavailable("ollama is down")

    monkeypatch.setattr("ocbrain.hybrid.embed_texts", down)
    refresh = embed_missing_beliefs(conn)
    assert refresh["embedded"] == 0
    assert refresh["skipped_reason"] == "local_embedding_unavailable:LocalEmbeddingUnavailable"
    assert vector_status(path)["coverage"]["dense_stale_rows"] == 1


def test_supersede_v1_reports_vector_refresh(tmp_path: Path, monkeypatch) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    target = "curated:bountiful:citrus"
    _seed_belief(conn, belief_id=target, body="Citrus lemons are ready.")
    conn.commit()
    marker = {"embedded": 7, "remaining": 0, "skipped_reason": None}
    monkeypatch.setattr("ocbrain.mcp_v1.embed_missing_beliefs", lambda *_args, **_kwargs: marker)

    payload = supersede_v1(
        conn,
        target=target,
        body="Citrus lemons are ready for pickup.",
        reason="the pickup detail was missing",
        context=ScopeContext(project="bountiful"),
        actor="agent:test",
    )
    assert payload["vector_refresh"]["skipped_reason"] == "deferred_until_commit"
    assert conn.in_transaction
    conn.commit()
    assert finish_vector_refresh(conn, payload)["vector_refresh"] == marker


def test_vector_refresh_runs_once_after_the_dispatcher_commits(tmp_path: Path, monkeypatch) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    target = "curated:bountiful:citrus"
    _seed_belief(conn, belief_id=target, body="Citrus lemons are ready.")
    conn.commit()
    seen: list[bool] = []

    def fake_embed(active_conn, **_kwargs):
        seen.append(active_conn.in_transaction)
        return {"embedded": 1, "remaining": 0, "skipped_reason": None}

    monkeypatch.setattr("ocbrain.mcp_v1.embed_missing_beliefs", fake_embed)
    monkeypatch.setattr("ocbrain.mcp.connect", lambda *_args, **_kwargs: conn, raising=False)
    payload = supersede_v1(
        conn,
        target=target,
        body="Citrus lemons are ready for pickup.",
        reason="the pickup detail was missing",
        context=ScopeContext(project="bountiful"),
        actor="agent:test",
    )
    assert seen == []
    conn.commit()
    finished = finish_vector_refresh(conn, payload)
    assert seen == [False]
    assert finished["vector_refresh"]["embedded"] == 1
    assert finish_vector_refresh(conn, finished)["vector_refresh"]["embedded"] == 1
    assert seen == [False]


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
            {},
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


def test_hybrid_relevance_gate_returns_empty_instead_of_same_scope_filler(
    tmp_path: Path, monkeypatch
) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    garden_noise = "curated:ocbrain:sparse-truth-hygiene"
    upgrade_noise = "curated:bountiful:old-deploy"
    _seed_belief(
        conn,
        belief_id=garden_noise,
        project="ocbrain",
        body=(
            "Keep background history harvests in the evidence ledger with evidence-only imports."
        ),
    )
    _seed_belief(
        conn,
        belief_id=upgrade_noise,
        body="A July deployment completed and production probes passed.",
    )
    conn.commit()

    similarities = {garden_noise: 0.22, upgrade_noise: 0.395}
    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, candidate_ids=None, **_kwargs: (
            [
                {"belief_id": belief_id, "similarity": similarities[belief_id]}
                for belief_id in sorted(candidate_ids or [])
            ],
            None,
            {},
        ),
    )

    garden = search_core_v1(
        conn,
        "Which tomatoes and peppers in my garden are ready to harvest today?",
        context=ScopeContext(project="ocbrain"),
        limit=10,
        delivery_target="hosted_model",
    )
    assert garden["items"] == []
    assert garden["ranking"]["lexical_candidates"] == 0
    assert garden["ranking"]["dense_candidates"] == 0

    upgrade = search_core_v1(
        conn,
        "How are OCBrain MCP tool schemas validated after an upgrade?",
        context=ScopeContext(project="bountiful"),
        limit=10,
        delivery_target="hosted_model",
    )
    assert upgrade["items"] == []
    assert upgrade["ranking"]["lexical_candidates"] == 0
    assert upgrade["ranking"]["dense_candidates"] == 1
    assert upgrade["ranking"]["min_dense_only_cosine"] == 0.55


def test_hybrid_relevance_gate_keeps_strong_dense_only_recall(tmp_path: Path, monkeypatch) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    relevant = "curated:bountiful:semantic-recall"
    _seed_belief(
        conn,
        belief_id=relevant,
        body="Meyer lemons are ready for neighborhood pickup.",
    )
    conn.commit()
    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, **_kwargs: ([{"belief_id": relevant, "similarity": 0.72}], None, {}),
    )

    result = search_core_v1(
        conn,
        "ripe citrus available nearby",
        context=ScopeContext(project="bountiful"),
        limit=10,
        delivery_target="hosted_model",
    )
    assert [item["belief_id"] for item in result["items"]] == [relevant]
    assert result["items"][0]["ranking"]["dense_similarity"] == 0.72


def test_hybrid_dense_only_floor_includes_boundary_and_rejects_below(
    tmp_path: Path, monkeypatch
) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    boundary = "curated:bountiful:dense-boundary"
    below = "curated:bountiful:dense-below"
    _seed_belief(conn, belief_id=boundary, body="Meyer lemons are available nearby.")
    _seed_belief(conn, belief_id=below, body="Tomatoes are ready for pickup.")
    conn.commit()
    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, **_kwargs: (
            [
                {"belief_id": below, "similarity": 0.5499},
                {"belief_id": boundary, "similarity": 0.55},
            ],
            None,
            {},
        ),
    )

    result = search_core_v1(
        conn,
        "otherwise unmatched semantic probe",
        context=ScopeContext(project="bountiful"),
        limit=10,
        delivery_target="hosted_model",
    )

    assert [item["belief_id"] for item in result["items"]] == [boundary]
    assert result["ranking"]["min_dense_only_cosine"] == 0.55


def test_multi_term_lexical_query_drops_single_generic_token_filler(
    tmp_path: Path, monkeypatch
) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    relevant = "curated:bountiful:transport-recovery"
    filler = "curated:bountiful:model-fallback"
    _seed_belief(
        conn,
        belief_id=relevant,
        body=(
            "Preserve exact feedback after a closed client stdio transport with a "
            "one-shot runtime-only fallback."
        ),
    )
    _seed_belief(
        conn,
        belief_id=filler,
        body="Use a second model as an independent fallback for planning.",
    )
    conn.commit()
    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, **_kwargs: ([], "test_lexical_only", {}),
    )

    result = search_core_v1(
        conn,
        "one-shot runtime-only fallback closed client stdio transport",
        context=ScopeContext(project="bountiful"),
        limit=10,
        delivery_target="hosted_model",
    )

    assert [item["belief_id"] for item in result["items"]] == [relevant]
    assert result["ranking"]["lexical_candidates"] == 1
    assert result["ranking"]["min_lexical_query_term_matches"] == 2
    assert result["ranking"]["min_redundant_lexical_strength_ratio"] == 0.5


def test_multi_term_lexical_query_preserves_distinctive_single_term_coverage(
    tmp_path: Path, monkeypatch
) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    generic = "curated:bountiful:generic-recovery"
    distinctive = "curated:bountiful:postgres-recovery"
    _seed_belief(
        conn,
        belief_id=generic,
        body="Database recovery procedures are documented.",
    )
    _seed_belief(
        conn,
        belief_id=distinctive,
        body="Postgres uses WAL archiving for point-in-time restore.",
    )
    conn.commit()
    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, **_kwargs: ([], "test_lexical_only", {}),
    )

    result = search_core_v1(
        conn,
        "postgres database backup recovery",
        context=ScopeContext(project="bountiful"),
        limit=10,
        delivery_target="hosted_model",
    )

    assert {item["belief_id"] for item in result["items"]} == {
        generic,
        distinctive,
    }
    assert result["ranking"]["lexical_candidates"] == 2


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
    with pytest.raises(ValueError, match="--allow-hosted-egress"):
        apply_curated_manifest(conn, manifest_path)
    assert conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0] == 0

    first = apply_curated_manifest(conn, manifest_path, allow_hosted_egress=True)
    second = apply_curated_manifest(conn, manifest_path, allow_hosted_egress=True)
    assert first["hosted_egress_acknowledged"] is True
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
    changed = apply_curated_manifest(conn, manifest_path, allow_hosted_egress=True)
    assert changed["applied"] == ["curated:bountiful:B01"]
    current = get_core_v1_belief(conn, "curated:bountiful:B01")
    assert current is not None
    assert current["attributes"]["source_quality"] == 0.72
    assert current["confidence"] == 0.83
    assert apply_curated_manifest(conn, manifest_path, allow_hosted_egress=True)[
        "unchanged"
    ] == ["curated:bountiful:B01"]

    manifest["facts"].append(dict(manifest["facts"][0]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate curated fact id"):
        apply_curated_manifest(conn, manifest_path, allow_hosted_egress=True)
    manifest["facts"].pop()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    source.write_text("replacement truth\n", encoding="utf-8")
    manifest["sources"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest["facts"][0]["body"] = "Updated Bountiful neighborhood food truth."
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    replaced = apply_curated_manifest(conn, manifest_path, allow_hosted_egress=True)
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
        apply_curated_manifest(conn, manifest_path, allow_hosted_egress=True)


def test_curated_manifest_rolls_back_if_a_later_fact_is_invalid(tmp_path: Path) -> None:
    source = tmp_path / "truth.md"
    source.write_text("verified truth\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "ocbrain.curated-memory.v1",
                "project": "bountiful",
                "sources": [
                    {
                        "ref": "S1",
                        "path": source.name,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ],
                "facts": [
                    {
                        "id": "valid-first",
                        "body": "This valid fact must roll back with the manifest.",
                        "source_refs": ["S1"],
                    },
                    {
                        "id": "invalid-second",
                        "body": "This fact references a missing source.",
                        "source_refs": ["MISSING"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)

    with pytest.raises(ValueError, match="unknown source MISSING"):
        apply_curated_manifest(conn, manifest_path)

    assert conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM current_beliefs").fetchone()[0] == 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["facts"] = [
        {
            "id": "confidential-hosted",
            "body": "Confidential facts cannot be acknowledged into hosted delivery.",
            "source_refs": ["S1"],
            "visibility": "confidential",
            "egress_policy": "hosted_ok",
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot combine hosted_ok with confidential"):
        apply_curated_manifest(conn, manifest_path, allow_hosted_egress=True)
    assert conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0] == 0


def test_tracked_hosted_context_demo_requires_ack_and_round_trips(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "examples" / "hosted-context-demo" / "manifest.json"
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)

    with pytest.raises(ValueError, match="--allow-hosted-egress"):
        apply_curated_manifest(conn, manifest)
    assert conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0] == 0

    applied = apply_curated_manifest(conn, manifest, allow_hosted_egress=True)
    assert len(applied["applied"]) == 4
    assert applied["hosted_egress_acknowledged"] is True
    assert apply_curated_manifest(
        conn, manifest, allow_hosted_egress=True
    )["unchanged"] == applied["applied"]

    packet = _mcp_payload(
        handle_request(
            conn,
            _mcp_call(
                "brain.context",
                {
                    "query": "OCBrain installation requirements and client constraints",
                    "context": {
                        "project": "ocbrain",
                        "runtime": "test",
                        "task": "hosted-demo-acceptance",
                    },
                    "limit": 10,
                },
                request_id=1,
            ),
            delivery_target="hosted_model",
        )
    )
    returned = {item["id"] for item in packet["items"]}
    assert "curated:ocbrain:installation-requirements" in returned
    assert "curated:ocbrain:client-constraints" in returned
    assert packet["coverage"]["excluded_delivery_count"] == 0
    assert packet["coverage"]["ranking"]["eligible_count"] == 4
    source_id = packet["items"][0]["sources"][0]["id"]
    source = _mcp_payload(
        handle_request(
            conn,
            _mcp_call(
                "brain.source",
                {
                    "id": source_id,
                    "context": {"project": "ocbrain", "runtime": "test"},
                },
                request_id=2,
            ),
            delivery_target="hosted_model",
        )
    )
    assert source["hash_verified"] is True
    assert source["uri"].startswith("ocbrain://evidence/")

    feedback = _mcp_payload(
        handle_request(
            conn,
            _mcp_call(
                "brain.feedback",
                {
                    "retrieval_use_id": packet["retrieval_use_id"],
                    "outcome": "used",
                    "note": "hosted demo contract informed acceptance",
                },
                request_id=3,
            ),
            delivery_target="hosted_model",
        )
    )
    assert feedback["outcome"] == "used"
    closeout = _mcp_payload(
        handle_request(
            conn,
            _mcp_call(
                "brain.closeout",
                {
                    "task_ref": "hosted-demo-acceptance",
                    "status": "completed",
                    "summary": "Verified hosted context and hash-checked source expansion.",
                    "retrieval_use_ids": [packet["retrieval_use_id"]],
                    "decision_impact": "informed",
                    "verifier_refs": [
                        {
                            "uri": "pytest://test_tracked_hosted_context_demo",
                            "kind": "pytest",
                            "status": "passed",
                        }
                    ],
                },
                request_id=4,
            ),
            delivery_target="hosted_model",
        )
    )
    assert closeout["schema_version"] == "ocbrain.closeout.v1"
    assert closeout["verification_status"] == "verified"
    assert str(root) not in json.dumps(packet)
    assert str(root) not in json.dumps(source)


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


def test_context_reports_scope_and_delivery_inventory_without_leaking_hosted_samples(
    tmp_path: Path,
) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_belief(
        conn,
        belief_id="curated:bountiful:safe",
        body="Hosted inventory sentinel safe.",
    )
    _seed_belief(
        conn,
        belief_id="curated:bountiful:local-only",
        body="PRIVATE_LOCAL_ONLY_SENTINEL",
        egress_policy="local_only",
    )
    _seed_belief(
        conn,
        belief_id="curated:bountiful:confidential",
        body="PRIVATE_CONFIDENTIAL_SENTINEL",
        visibility="confidential",
    )
    _seed_belief(
        conn,
        belief_id="curated:foreign:hosted",
        body="PRIVATE_FOREIGN_SCOPE_SENTINEL",
        project="foreign",
    )
    conn.commit()

    packet, _handles = build_context_v1(
        conn,
        "query with no lexical match",
        context=ScopeContext(project="bountiful"),
        limit=10,
        cross_scope=False,
        delivery_target="hosted_model",
    )

    assert packet["items"] == []
    assert packet["coverage"]["scope_mix"] == {}
    assert packet["coverage"]["excluded_delivery_count"] == 2
    assert packet["coverage"]["exclusion_count_basis"] == "current_serving_inventory"
    assert packet["coverage"]["ranking"]["eligible_count"] == 1
    assert packet["coverage"]["excluded_sample"] == []
    encoded = json.dumps(packet)
    assert "PRIVATE_" not in encoded
    assert "curated:bountiful:local-only" not in encoded
    assert "curated:bountiful:confidential" not in encoded
    assert "curated:foreign:hosted" not in encoded


def test_sql_prefilters_match_global_and_client_scope_semantics(tmp_path: Path) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_belief(
        conn,
        belief_id="curated:global:alternate",
        body="Scope SQL sentinel globally visible.",
        scope=ScopeTag(
            "global",
            "global:alternate",
            visibility="internal",
            egress_policy="hosted_ok",
            provenance="test",
        ),
    )
    _seed_belief(
        conn,
        belief_id="curated:client:internal",
        body="PRIVATE_CLIENT_SCOPE_SENTINEL",
        scope=ScopeTag(
            "client",
            "client:codex",
            visibility="internal",
            egress_policy="hosted_ok",
            provenance="test",
        ),
    )
    conn.commit()

    exact_client, _handles = build_context_v1(
        conn,
        "Scope SQL sentinel private client",
        context=ScopeContext(project="bountiful", client="codex"),
        limit=10,
        cross_scope=False,
        delivery_target="hosted_model",
    )
    assert [item["id"] for item in exact_client["items"]] == [
        "curated:global:alternate"
    ]
    assert exact_client["coverage"]["scope_mix"] == {"global:alternate": 1}
    assert exact_client["coverage"]["excluded_delivery_count"] == 1
    assert exact_client["coverage"]["ranking"]["eligible_count"] == 1
    assert "PRIVATE_CLIENT_SCOPE_SENTINEL" not in json.dumps(exact_client)

    # ``cross_scope`` is accepted and ignored. Hosted delivery keeps its scope
    # IN-list either way, so a foreign caller still sees only global material.
    cross_scope, _handles = build_context_v1(
        conn,
        "Scope SQL sentinel private client",
        context=ScopeContext(project="other"),
        limit=10,
        cross_scope=True,
        delivery_target="hosted_model",
    )
    assert [item["id"] for item in cross_scope["items"]] == [
        "curated:global:alternate"
    ]
    assert cross_scope["coverage"]["scope_mix"] == {"global:alternate": 1}
    assert cross_scope["coverage"]["excluded_delivery_count"] == 0
    assert cross_scope["coverage"]["ranking"]["eligible_count"] == 1
    assert "PRIVATE_CLIENT_SCOPE_SENTINEL" not in json.dumps(cross_scope)


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


def _record_feedback(
    conn,
    *,
    belief_id: str,
    outcome: str,
    count: int,
    prefix: str,
) -> None:
    """Insert ``count`` judged retrievals for one belief."""
    for index in range(count):
        use_id = f"ret_{prefix}_{index}"
        conn.execute(
            "INSERT INTO retrieval_uses (id, served_to_runtime, outcome, served_at) "
            "VALUES (?,?,?,?)",
            (use_id, "test", outcome, "2026-08-04T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO retrieval_items (retrieval_use_id, object_id, object_kind, rank, score) "
            "VALUES (?,?,?,?,?)",
            (use_id, belief_id, "belief", 1, 0.5),
        )


def test_lexical_hit_below_dense_floor_is_rejected(tmp_path: Path, monkeypatch) -> None:
    """A shared generic token must not serve a belief the dense arm rejects.

    Reproduces the live failure: a query with zero topical overlap returned two
    unrelated beliefs at dense similarity ~0.33 purely because FTS matched one
    generic token, and unweighted lexical RRF outranked every dense candidate.
    """
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    filler_a = "curated:bountiful:orchestration-preference"
    filler_b = "curated:bountiful:lake-root"
    _seed_belief(
        conn,
        belief_id=filler_a,
        body="Strategy work is advisory and execution stays with the local root agent.",
    )
    _seed_belief(
        conn,
        belief_id=filler_b,
        body="The data lake is rooted on local disk with a storage budget.",
    )
    conn.commit()

    similarities = {filler_a: 0.336, filler_b: 0.328}
    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, candidate_ids=None, **_kwargs: (
            [
                {"belief_id": belief_id, "similarity": similarities[belief_id]}
                for belief_id in sorted(candidate_ids or [])
            ],
            None,
            {},
        ),
    )

    result = search_core_v1(
        conn,
        "recommender replication from analytics export data",
        context=ScopeContext(project="bountiful"),
        limit=10,
        delivery_target="hosted_model",
    )

    assert result["items"] == []
    assert result["ranking"]["require_dense_support"] is True


def test_lexical_hit_below_dense_floor_survives_exact_locator(
    tmp_path: Path, monkeypatch
) -> None:
    """Naming a belief outright must still fetch it, whatever the dense score."""
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    target = "curated:bountiful:exact-locator"
    _seed_belief(conn, belief_id=target, body="Deployment probes run after each release.")
    conn.commit()
    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, **_kwargs: ([{"belief_id": target, "similarity": 0.01}], None, {}),
    )

    result = search_core_v1(
        conn,
        f"what does {target} say about probes",
        context=ScopeContext(project="bountiful"),
        limit=10,
        delivery_target="hosted_model",
    )

    assert [item["belief_id"] for item in result["items"]] == [target]


def test_lexical_hit_kept_when_dense_arm_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    """A stale or missing sidecar must degrade to lexical, not to silence."""
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    target = "curated:bountiful:lexical-only"
    _seed_belief(
        conn,
        belief_id=target,
        body="Deployment probes run after each production release.",
    )
    conn.commit()
    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, **_kwargs: ([], "vector_sidecar_missing", {}),
    )

    result = search_core_v1(
        conn,
        "production release deployment probes",
        context=ScopeContext(project="bountiful"),
        limit=10,
        delivery_target="hosted_model",
    )

    assert [item["belief_id"] for item in result["items"]] == [target]
    assert result["ranking"]["dense_fallback"] == "vector_sidecar_missing"
    assert result["ranking"]["require_dense_support"] is False
    assert result["ranking"]["degraded_excluded_procedures"] == 0


def test_degraded_mode_drops_procedures_but_keeps_gotchas(
    tmp_path: Path, monkeypatch
) -> None:
    """A wrong belief is a wrong sentence; a wrong procedure is a wrong afternoon.

    With the dense arm down the relevance floors stand down, so a procedure
    could be served on one shared token. Gotchas are sentence-shaped claims and
    carry the same risk a belief does, so they keep serving.
    """
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_belief(
        conn,
        belief_id="procedure:bountiful:release",
        body="Deployment probes run after each production release.",
        belief_type="procedure",
    )
    _seed_belief(
        conn,
        belief_id="gotcha:bountiful:release",
        body="Deployment probes fail on 46% of production release calls.",
        belief_type="gotcha",
    )
    conn.commit()
    query = "production release deployment probes"
    context = ScopeContext(project="bountiful")

    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, **_kwargs: (
            [
                {"belief_id": "procedure:bountiful:release", "similarity": 0.91},
                {"belief_id": "gotcha:bountiful:release", "similarity": 0.90},
            ],
            None,
            {},
        ),
    )
    healthy = search_core_v1(conn, query, context=context, limit=10)
    assert {item["belief_id"] for item in healthy["items"]} == {
        "procedure:bountiful:release",
        "gotcha:bountiful:release",
    }
    assert healthy["ranking"]["degraded_excluded_procedures"] == 0

    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, **_kwargs: ([], "vector_sidecar_missing", {}),
    )
    degraded = search_core_v1(conn, query, context=context, limit=10)

    assert [item["belief_id"] for item in degraded["items"]] == ["gotcha:bountiful:release"]
    assert degraded["ranking"]["degraded_excluded_procedures"] == 1


def test_uncorroborated_multi_term_query_drops_every_lexical_row(
    tmp_path: Path, monkeypatch
) -> None:
    """When no row clears the multi-term bar, the filter must not fail open.

    Previously the redundancy filter only ran when at least one row achieved
    two-term overlap; with none, every one-generic-token row was served.
    """
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    first = "curated:bountiful:one-token-a"
    second = "curated:bountiful:one-token-b"
    _seed_belief(conn, belief_id=first, body="Nightly deployment finished without incident.")
    _seed_belief(conn, belief_id=second, body="A deployment window opens on Tuesday.")
    conn.commit()
    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, **_kwargs: ([], None, {}),
    )

    result = search_core_v1(
        conn,
        "deployment strategy for quarterly forecasting revenue attribution models",
        context=ScopeContext(project="bountiful"),
        limit=10,
        delivery_target="hosted_model",
    )

    assert result["items"] == []
    assert result["ranking"]["lexical_candidates"] == 0


def test_retrieval_thresholds_honor_env_overrides(tmp_path: Path, monkeypatch) -> None:
    """Operators must be able to tune the gates without editing source."""
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    target = "curated:bountiful:tunable"
    _seed_belief(conn, belief_id=target, body="Meyer lemons are available nearby.")
    conn.commit()
    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, **_kwargs: ([{"belief_id": target, "similarity": 0.40}], None, {}),
    )
    probe = "otherwise unmatched semantic probe"
    context = ScopeContext(project="bountiful")

    # 0.40 sits below the shipped dense-only floor of 0.55.
    default_result = search_core_v1(
        conn, probe, context=context, limit=10, delivery_target="hosted_model"
    )
    assert default_result["items"] == []

    monkeypatch.setenv("OCBRAIN_RETRIEVAL_MIN_DENSE_ONLY_COSINE", "0.35")
    lowered = search_core_v1(
        conn, probe, context=context, limit=10, delivery_target="hosted_model"
    )
    assert [item["belief_id"] for item in lowered["items"]] == [target]
    assert lowered["ranking"]["min_dense_only_cosine"] == 0.35


def test_retrieval_feedback_can_reorder_results(tmp_path: Path, monkeypatch) -> None:
    """Judged retrievals must be able to move a belief, not just decorate it."""
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    disliked = "curated:bountiful:aaa-disliked"
    liked = "curated:bountiful:zzz-liked"
    _seed_belief(conn, belief_id=disliked, body="Meyer lemons ripen in winter nearby.")
    _seed_belief(conn, belief_id=liked, body="Meyer lemons ripen in winter locally.")
    conn.commit()
    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, **_kwargs: (
            [
                {"belief_id": disliked, "similarity": 0.70},
                {"belief_id": liked, "similarity": 0.70},
            ],
            None,
            {},
        ),
    )
    probe = "meyer lemons ripen winter"
    context = ScopeContext(project="bountiful")

    before = search_core_v1(conn, probe, context=context, limit=10, delivery_target="hosted_model")
    baseline = [item["belief_id"] for item in before["items"]]
    assert sorted(baseline) == sorted([disliked, liked])

    # Reward whichever belief the ranker put second, and penalize the leader:
    # feedback must be strong enough to overturn the baseline order.
    leader, runner_up = baseline
    _record_feedback(conn, belief_id=leader, outcome="irrelevant", count=8, prefix="bad")
    _record_feedback(conn, belief_id=runner_up, outcome="helpful", count=8, prefix="good")
    conn.commit()

    after = search_core_v1(conn, probe, context=context, limit=10, delivery_target="hosted_model")
    assert [item["belief_id"] for item in after["items"]] == [runner_up, leader]


def test_feedback_boost_is_damped_by_observation_count(tmp_path: Path) -> None:
    """One verdict must not swing a belief as far as a consistent record."""
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    single = "curated:bountiful:one-vote"
    many = "curated:bountiful:many-votes"
    _seed_belief(conn, belief_id=single, body="Single verdict belief.")
    _seed_belief(conn, belief_id=many, body="Repeated verdict belief.")
    _record_feedback(conn, belief_id=single, outcome="helpful", count=1, prefix="one")
    _record_feedback(conn, belief_id=many, outcome="helpful", count=20, prefix="many")
    conn.commit()

    scores = _retrieval_feedback_scores(conn, {single, many})
    assert 0 < scores[single] < scores[many]
    assert scores[many] <= 0.25


def test_deduplicated_candidates_counts_only_duplicates(tmp_path: Path, monkeypatch) -> None:
    """The counter must not fold `limit` truncation into the dedup total."""
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    ids = [f"curated:bountiful:distinct-{index}" for index in range(4)]
    for index, belief_id in enumerate(ids):
        _seed_belief(conn, belief_id=belief_id, body=f"Distinct harvest note number {index}.")
    conn.commit()
    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, candidate_ids=None, **_kwargs: (
            [
                {"belief_id": belief_id, "similarity": 0.80}
                for belief_id in sorted(candidate_ids or [])
            ],
            None,
            {},
        ),
    )

    result = search_core_v1(
        conn,
        "unmatched semantic probe",
        context=ScopeContext(project="bountiful"),
        limit=2,
        delivery_target="hosted_model",
    )

    # Four distinct bodies, two served: the two unserved were truncated, not deduped.
    assert len(result["items"]) == 2
    assert result["ranking"]["deduplicated_candidates"] == 0


def test_uncorroborated_lexical_rows_survive_when_dense_arm_is_down(
    tmp_path: Path, monkeypatch
) -> None:
    """Degraded mode must not be stricter than hybrid mode.

    With no dense arm to answer instead, dropping uncorroborated lexical rows
    would turn a sidecar outage into total silence.
    """
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    target = "curated:bountiful:degraded-only"
    _seed_belief(conn, belief_id=target, body="Nightly deployment finished without incident.")
    conn.commit()
    probe = "deployment strategy for quarterly forecasting revenue attribution"
    context = ScopeContext(project="bountiful")

    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, **_kwargs: ([], None, {}),
    )
    healthy = search_core_v1(conn, probe, context=context, limit=10, delivery_target="hosted_model")
    assert healthy["items"] == []

    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, **_kwargs: ([], "vector_sidecar_missing", {}),
    )
    degraded = search_core_v1(
        conn, probe, context=context, limit=10, delivery_target="hosted_model"
    )
    assert [item["belief_id"] for item in degraded["items"]] == [target]


def _rerank_fixture(conn, monkeypatch) -> list[str]:
    ids = [f"curated:bountiful:rerank-{index}" for index in range(5)]
    for index, belief_id in enumerate(ids):
        _seed_belief(conn, belief_id=belief_id, body=f"Rerank candidate number {index}.")
    conn.commit()
    similarities = {belief_id: 0.90 - 0.02 * index for index, belief_id in enumerate(ids)}
    monkeypatch.setattr(
        "ocbrain.core_v1.semantic_neighbors",
        lambda *_args, candidate_ids=None, **_kwargs: (
            [
                {"belief_id": belief_id, "similarity": similarity}
                for belief_id, similarity in similarities.items()
            ],
            None,
            {},
        ),
    )
    return ids


def test_rerank_stage_promotes_a_low_ranked_candidate_into_the_packet(
    tmp_path: Path, monkeypatch
) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    ids = _rerank_fixture(conn, monkeypatch)
    monkeypatch.setenv("OCBRAIN_CONFIG", str(tmp_path / "absent.json"))
    monkeypatch.setenv("OCBRAIN_RERANK_ENABLED", "true")

    class _Scorer:
        def score(self, query: str, documents: list[str]) -> list[float]:
            return [10.0 if "number 4" in document else 0.0 for document in documents]

    monkeypatch.setattr("ocbrain.rerank.get_scorer", lambda _config: _Scorer())

    result = search_core_v1(
        conn,
        "unmatched semantic probe",
        context=ScopeContext(project="bountiful"),
        limit=3,
        delivery_target="hosted_model",
    )

    assert result["ranking"]["rerank"]["mode"] == "applied"
    assert result["ranking"]["rerank"]["candidates"] == 5
    served = [item["belief_id"] for item in result["items"]]
    assert served == [ids[4], ids[0], ids[1]]
    assert result["items"][0]["ranking"]["pre_rerank_rank"] == 5
    assert result["items"][0]["ranking"]["rerank_score"] == 10.0


def test_rerank_stage_off_by_default_changes_nothing(tmp_path: Path, monkeypatch) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    ids = _rerank_fixture(conn, monkeypatch)
    monkeypatch.setenv("OCBRAIN_CONFIG", str(tmp_path / "absent.json"))
    monkeypatch.delenv("OCBRAIN_RERANK_ENABLED", raising=False)

    result = search_core_v1(
        conn,
        "unmatched semantic probe",
        context=ScopeContext(project="bountiful"),
        limit=3,
        delivery_target="hosted_model",
    )

    assert result["ranking"]["rerank"] == {"mode": "off"}
    assert [item["belief_id"] for item in result["items"]] == ids[:3]
    assert all("rerank_score" not in item["ranking"] for item in result["items"])


ASA2_VOCABULARY = {"asa2": ["asa2", "applied-science-analytics-2"]}


def _entity_vocabulary(monkeypatch) -> None:
    monkeypatch.setenv("OCBRAIN_ENTITIES_VOCABULARY", json.dumps(ASA2_VOCABULARY))


def _mention_entities(conn, belief_id: str) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT entity FROM entity_mentions WHERE belief_id=?", (belief_id,)
        )
    }


def test_a_query_entity_with_enough_corpus_filters_the_candidate_pool(
    tmp_path: Path, monkeypatch
) -> None:
    """A query naming one thing must not serve beliefs about another.

    Five asa2 beliefs and three slt beliefs all say "cadence"; the entity is the
    only thing that separates them.
    """
    _entity_vocabulary(monkeypatch)
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    asa2_ids = [f"curated:bountiful:asa2-{index}" for index in range(5)]
    slt_ids = [f"curated:bountiful:slt-{index}" for index in range(3)]
    for index, belief_id in enumerate(asa2_ids):
        _seed_belief(
            conn,
            belief_id=belief_id,
            body=f"The asa2 headroom leg cadence is weekly ({index}).",
        )
    for index, belief_id in enumerate(slt_ids):
        _seed_belief(
            conn,
            belief_id=belief_id,
            body=f"The slt retrain cadence is quarterly ({index}).",
        )
    conn.commit()

    result = search_core_v1(
        conn,
        "asa2 headroom leg cadence",
        context=ScopeContext(project="bountiful"),
        limit=8,
        delivery_target="hosted_model",
    )
    assert result["ranking"]["entities"] == {
        "query": ["asa2"],
        "mode": "filter",
        "candidates": 5,
    }
    assert {item["belief_id"] for item in result["items"]} == set(asa2_ids)


def test_a_query_entity_with_a_thin_corpus_boosts_instead_of_filtering(
    tmp_path: Path, monkeypatch
) -> None:
    """Two mentions is not enough evidence to hide the beliefs that answer
    without repeating the entity name."""
    _entity_vocabulary(monkeypatch)
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    asa2_ids = ["curated:bountiful:asa2-a", "curated:bountiful:asa2-b"]
    for index, belief_id in enumerate(asa2_ids):
        _seed_belief(
            conn,
            belief_id=belief_id,
            body=f"The asa2 headroom leg cadence is weekly ({index}).",
        )
    for index in range(3):
        _seed_belief(
            conn,
            belief_id=f"curated:bountiful:slt-{index}",
            body=f"The slt retrain cadence is quarterly ({index}).",
        )
    conn.commit()

    result = search_core_v1(
        conn,
        "asa2 headroom leg cadence",
        context=ScopeContext(project="bountiful"),
        limit=8,
        delivery_target="hosted_model",
    )
    assert result["ranking"]["entities"]["mode"] == "boost"
    assert {item["belief_id"] for item in result["items"][:2]} == set(asa2_ids)
    assert result["items"][0]["ranking"]["entity_boost"] > 0
    assert result["items"][0]["ranking"]["matched_entities"] >= 1


def test_query_shape_and_the_keyword_hint_reach_the_caller(tmp_path: Path, monkeypatch) -> None:
    _entity_vocabulary(monkeypatch)
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_belief(
        conn, belief_id="curated:bountiful:cadence", body="The asa2 cadence is weekly."
    )
    conn.commit()
    context = ScopeContext(project="bountiful")

    question = search_core_v1(
        conn, "What is the asa2 cadence?", context=context, limit=3, delivery_target="hosted_model"
    )
    assert question["ranking"]["query_shape"] == "question"
    assert question["ranking"]["hint"] is None

    keywords = search_core_v1(
        conn,
        "cadence retrain notes",
        context=context,
        limit=3,
        delivery_target="hosted_model",
    )
    assert keywords["ranking"]["query_shape"] == "keywords"
    assert keywords["ranking"]["hint"] == KEYWORD_QUERY_HINT

    packet, _handles = build_context_v1(
        conn,
        "cadence retrain notes",
        context=context,
        limit=3,
        delivery_target="hosted_model",
    )
    assert packet["coverage"]["ranking"]["query_shape"] == "keywords"
    assert packet["coverage"]["ranking"]["hint"] == KEYWORD_QUERY_HINT


def test_reprojection_rewrites_entity_mentions_when_a_body_changes(
    tmp_path: Path, monkeypatch
) -> None:
    _entity_vocabulary(monkeypatch)
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    belief_id = "curated:bountiful:constraint"
    _seed_belief(conn, belief_id=belief_id, body="The asa2 headroom leg is the constraint.")
    conn.commit()
    assert _mention_entities(conn, belief_id) == {"asa2"}

    _seed_belief(conn, belief_id=belief_id, body="The slt retrain cadence is quarterly.")
    conn.commit()
    assert _mention_entities(conn, belief_id) == set()


def _seed_recency_pair(conn, *, lifecycle: str, prefix: str) -> tuple[str, str]:
    old_id = f"curated:test:{prefix}-old"
    new_id = f"curated:test:{prefix}-new"
    attributes = {"source_quality": 0.95, "lifecycle": lifecycle}
    _seed_belief(
        conn,
        belief_id=old_id,
        body=f"citrus grove note {prefix} record one",
        attributes=attributes,
    )
    _seed_belief(
        conn,
        belief_id=new_id,
        body=f"citrus grove note {prefix} record two",
        attributes=attributes,
    )
    compiled = (datetime.now(UTC) - timedelta(days=90)).isoformat(timespec="microseconds")
    conn.execute(
        "UPDATE current_beliefs SET last_compiled_at=?, pinned=1 WHERE belief_id=?",
        (compiled, old_id),
    )
    conn.commit()
    return old_id, new_id


def test_current_lifecycle_recency_lifts_the_newer_belief(tmp_path: Path) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    old_id, new_id = _seed_recency_pair(conn, lifecycle="current", prefix="current")
    packet = search_core_v1(
        conn,
        "citrus grove note",
        context=ScopeContext(project="bountiful"),
        limit=5,
    )
    items = [item for item in packet["items"] if item["belief_id"] in {old_id, new_id}]
    assert [item["belief_id"] for item in items] == [new_id, old_id]
    assert items[0]["ranking"]["recency_model"] == "current"
    assert items[0]["ranking"]["recency_half_life_days"] == 30.0


def test_durable_recency_does_not_reorder_the_pair(tmp_path: Path) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    old_id, new_id = _seed_recency_pair(conn, lifecycle="durable", prefix="durable")
    packet = search_core_v1(
        conn,
        "citrus grove note",
        context=ScopeContext(project="bountiful"),
        limit=5,
    )
    items = [item for item in packet["items"] if item["belief_id"] in {old_id, new_id}]
    assert [item["belief_id"] for item in items] == [old_id, new_id]
    assert items[0]["ranking"]["recency_model"] == "durable"
    assert items[0]["ranking"]["recency_half_life_days"] == 365.0


def test_expired_belief_is_excluded_at_read_time_and_counted(tmp_path: Path) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    target = "curated:test:expired-window"
    _seed_belief(
        conn,
        belief_id=target,
        body="Quince paste sets hardest when the fruit is barely ripe.",
        attributes={
            "source_quality": 0.95,
            "valid_from": "2019-01-01T00:00:00+00:00",
            "valid_until": "2020-01-01T00:00:00+00:00",
        },
    )
    conn.commit()
    context = ScopeContext(project="bountiful")
    query = "quince paste sets hardest"

    current_view = search_core_v1(conn, query, context=context, limit=5)
    assert current_view["items"] == []
    assert current_view["expired_excluded"] == 1

    past_view = search_core_v1(
        conn, query, context=context, limit=5, as_of="2019-06-01T00:00:00+00:00"
    )
    assert [item["belief_id"] for item in past_view["items"]] == [target]
    assert past_view["items"][0]["era"] == {
        "valid_from": "2019-01-01T00:00:00+00:00",
        "valid_until": "2020-01-01T00:00:00+00:00",
    }


def test_as_of_serves_the_belief_that_was_current_then(tmp_path: Path) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    old_id = "curated:test:quay-old"
    new_id = "curated:test:quay-new"
    _seed_belief(
        conn,
        belief_id=old_id,
        body="The quay gate closes at dusk in winter.",
        attributes={"source_quality": 0.95, "valid_from": "2020-01-01T00:00:00+00:00"},
    )
    _seed_belief(
        conn,
        belief_id=new_id,
        body="The quay gate closes at dawn in winter.",
        attributes={"source_quality": 0.95, "valid_from": "2026-01-01T00:00:00+00:00"},
    )
    append_core_event(
        conn,
        "correction_recorded",
        {
            "target_layer": "belief",
            "target_id": old_id,
            "op": "supersede",
            "successor_id": new_id,
            "author": "test",
        },
        writer="test",
        project=True,
    )
    conn.commit()
    context = ScopeContext(project="bountiful")
    query = "quay gate closes winter"

    current_view = search_core_v1(conn, query, context=context, limit=5)
    assert [item["belief_id"] for item in current_view["items"]] == [new_id]

    past_view = search_core_v1(
        conn, query, context=context, limit=5, as_of="2025-06-01T00:00:00+00:00"
    )
    assert [item["belief_id"] for item in past_view["items"]] == [old_id]
    assert past_view["items"][0]["status"] == "retracted"
    assert past_view["items"][0]["era"]["valid_until"] is not None
    assert past_view["retired_included"] >= 1


def _seed_dense_pair(conn, tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "core.sqlite"
    _seed_belief(conn, belief_id="curated:bountiful:citrus", body="Citrus lemons are ready.")
    _seed_belief(conn, belief_id="curated:bountiful:tomato", body="Tomatoes are available.")
    conn.commit()
    _local_dense_arm(monkeypatch)
    build_vector_index(path, model="test-local")
    return path


def test_sidecar_model_is_authoritative_over_the_configured_model(
    tmp_path: Path, monkeypatch
) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    path = _seed_dense_pair(conn, tmp_path, monkeypatch)
    build_vector_index(path, model="model-a")

    monkeypatch.setenv("OCBRAIN_EMBED_MODEL", "model-b")
    neighbors, fallback, _stats = semantic_neighbors(conn, "citrus harvest")
    assert fallback is None
    assert neighbors[0]["belief_id"] == "curated:bountiful:citrus"

    status = vector_status(path)
    assert status["sidecar_model"] == "model-a"
    assert status["configured_model"] == "model-b"
    assert status["model_matches_configured"] is False
    assert status["identity_fresh"] is True
    assert status["healthy"] is True


def test_sidecar_model_ollama_does_not_know_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_dense_pair(conn, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "ocbrain.hybrid._ollama_model_metadata", lambda *_a, **_k: {"digest": "unknown"}
    )
    neighbors, fallback, _stats = semantic_neighbors(conn, "citrus harvest")
    assert neighbors == []
    assert fallback == "vector_model_identity_unavailable"


def test_sidecar_dimension_metadata_must_match_the_stored_blob(
    tmp_path: Path, monkeypatch
) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    path = _seed_dense_pair(conn, tmp_path, monkeypatch)
    sidecar = sqlite3.connect(vector_db_path(path))
    try:
        sidecar.execute("UPDATE meta SET value='3' WHERE key='dimensions'")
        sidecar.commit()
    finally:
        sidecar.close()
    neighbors, fallback, _stats = semantic_neighbors(conn, "citrus harvest")
    assert neighbors == []
    assert fallback == "vector_dimension_config_mismatch"


def test_embed_missing_beliefs_uses_the_sidecar_model(tmp_path: Path, monkeypatch) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_dense_pair(conn, tmp_path, monkeypatch)
    seen: list[str] = []

    def _record(texts, *, model, **kwargs):
        seen.append(model)
        return [[1.0, 0.0] for _text in texts]

    monkeypatch.setattr("ocbrain.hybrid.embed_texts", _record)
    monkeypatch.setenv("OCBRAIN_RETRIEVAL_EMBED_ON_WRITE", "0")
    _seed_belief(conn, belief_id="curated:bountiful:pear", body="Pears are ready.")
    conn.commit()
    monkeypatch.delenv("OCBRAIN_RETRIEVAL_EMBED_ON_WRITE")
    monkeypatch.setenv("OCBRAIN_EMBED_MODEL", "model-b")

    refresh = embed_missing_beliefs(conn)
    assert refresh["embedded"] == 1
    assert seen == ["test-local"]


def test_fusion_weights_follow_the_query_shape(tmp_path: Path, monkeypatch) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_dense_pair(conn, tmp_path, monkeypatch)

    question = _search(conn, "What is the citrus harvest like?")
    assert question["ranking"]["query_shape"] == "question"
    assert question["ranking"]["fusion"] == {
        "lexical_weight": pytest.approx(0.35),
        "dense_weight": pytest.approx(0.65),
        "reason": "question",
    }

    identifier = _search(conn, "belief_0123456789abcdef citrus")
    assert identifier["ranking"]["fusion"]["reason"] == "identifiers"
    assert identifier["ranking"]["fusion"]["lexical_weight"] == pytest.approx(0.65)

    keywords = _search(conn, "citrus harvest")
    assert keywords["ranking"]["query_shape"] == "keywords"
    assert keywords["ranking"]["fusion"] == {
        "lexical_weight": pytest.approx(0.65),
        "dense_weight": pytest.approx(0.35),
        "reason": "keywords",
    }

    question_with_identifier = _search(conn, "What changed in belief_0123456789abcdef?")
    assert question_with_identifier["ranking"]["fusion"]["reason"] == "identifiers"


def test_adaptive_fusion_off_reproduces_the_unweighted_fusion(
    tmp_path: Path, monkeypatch
) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_dense_pair(conn, tmp_path, monkeypatch)

    monkeypatch.setenv("OCBRAIN_RETRIEVAL_ADAPTIVE_FUSION", "0")
    result = _search(conn, "citrus harvest", limit=5)
    assert result["ranking"]["fusion"] == {
        "lexical_weight": 0.5,
        "dense_weight": 0.5,
        "reason": "disabled",
    }
    assert result["items"]
    for item in result["items"]:
        unweighted = item["ranking"]["lexical_component"] + item["ranking"]["dense_component"]
        assert item["relevance"] == pytest.approx(unweighted)


def test_fusion_reports_dense_unavailable_when_the_arm_is_down(
    tmp_path: Path, monkeypatch
) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_belief(conn, belief_id="curated:bountiful:citrus", body="Citrus lemons are ready.")
    _seed_belief(conn, belief_id="curated:bountiful:tomato", body="Tomatoes are available.")
    conn.commit()

    result = _search(conn, "What is the citrus harvest like?")
    assert result["ranking"]["dense_fallback"] == "vector_sidecar_missing"
    assert result["ranking"]["fusion"] == {
        "lexical_weight": 0.5,
        "dense_weight": 0.5,
        "reason": "dense_unavailable",
    }
