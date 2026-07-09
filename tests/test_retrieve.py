from __future__ import annotations

import sqlite3
from pathlib import Path

from ocbrain.db import connect, init_db, upsert_knowledge
from ocbrain.embed import encode_embedding
from ocbrain.retrieve import (
    blend_scores,
    cosine_similarity,
    hybrid_knowledge_search,
    semantic_neighbors,
)


def _db(tmp_path: Path) -> sqlite3.Connection:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    return conn


def _know(
    conn: sqlite3.Connection,
    predicate: str,
    value_text: str,
    *,
    scope: str = "workspace",
    vector: list[float] | None = None,
) -> str:
    kid = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime",
        predicate=predicate,
        value_text=value_text,
        status="current",
        privacy_scope=scope,
    )
    if vector is not None:
        conn.execute(
            "UPDATE knowledge SET embedding=? WHERE id=?", (encode_embedding(vector), kid)
        )
    return kid


# --------------------------------------------------------------------------- #
# cosine_similarity
# --------------------------------------------------------------------------- #
def test_cosine_identical_orthogonal_and_edge_cases() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    # Same direction, different magnitude -> still 1.
    assert abs(cosine_similarity([2.0, 0.0], [5.0, 0.0]) - 1.0) < 1e-9
    # Empty / mismatched length / zero-norm -> 0.
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# --------------------------------------------------------------------------- #
# blend_scores
# --------------------------------------------------------------------------- #
def test_blend_none_semantic_is_lexical_only() -> None:
    assert blend_scores(0.7, None) == 0.7


def test_blend_half_and_weight_clamp() -> None:
    assert abs(blend_scores(0.4, 0.8, semantic_weight=0.5) - 0.6) < 1e-9
    # weight=1 -> pure semantic; weight clamps above 1.
    assert abs(blend_scores(0.4, 0.8, semantic_weight=1.0) - 0.8) < 1e-9
    assert abs(blend_scores(0.4, 0.8, semantic_weight=5.0) - 0.8) < 1e-9
    # weight=0 -> pure lexical.
    assert abs(blend_scores(0.4, 0.8, semantic_weight=0.0) - 0.4) < 1e-9


# --------------------------------------------------------------------------- #
# semantic_neighbors
# --------------------------------------------------------------------------- #
def test_semantic_neighbors_rank_by_cosine(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    near = _know(conn, "near", "near claim", vector=[1.0, 0.0, 0.0])
    mid = _know(conn, "mid", "mid claim", vector=[1.0, 1.0, 0.0])
    far = _know(conn, "far", "far claim", vector=[0.0, 1.0, 0.0])

    neighbors = semantic_neighbors(conn, [1.0, 0.0, 0.0], limit=3)
    ids = [n["knowledge_id"] for n in neighbors]
    assert ids[0] == near
    assert ids[-1] == far
    assert mid in ids
    assert neighbors[0]["similarity"] == 1.0


def test_semantic_neighbors_empty_query_returns_empty(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    _know(conn, "a", "a", vector=[1.0, 0.0])
    assert semantic_neighbors(conn, []) == []


def test_semantic_neighbors_excludes_private_by_default(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    pub = _know(conn, "pub", "pub", vector=[1.0, 0.0])
    priv = _know(conn, "priv", "priv", scope="private", vector=[1.0, 0.0])
    default_ids = {n["knowledge_id"] for n in semantic_neighbors(conn, [1.0, 0.0])}
    assert pub in default_ids
    assert priv not in default_ids
    incl = semantic_neighbors(conn, [1.0, 0.0], include_private=True)
    assert priv in {n["knowledge_id"] for n in incl}


def test_semantic_neighbors_min_similarity_filter(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    _know(conn, "orthogonal", "x", vector=[0.0, 1.0])
    near = _know(conn, "near", "y", vector=[1.0, 0.0])
    out = semantic_neighbors(conn, [1.0, 0.0], min_similarity=0.5)
    assert [n["knowledge_id"] for n in out] == [near]


# --------------------------------------------------------------------------- #
# hybrid_knowledge_search
# --------------------------------------------------------------------------- #
def test_hybrid_lexical_only_fallback_without_query_vector(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    hit = _know(conn, "hit", "postgres connection pooling settings", vector=[1.0, 0.0])
    _know(conn, "miss", "unrelated gardening topic", vector=[0.0, 1.0])
    out = hybrid_knowledge_search(conn, "postgres pooling", query_vector=None)
    ids = [r["knowledge_id"] for r in out]
    assert ids == [hit]
    # No query vector -> similarity component is zero, score == lexical.
    assert out[0]["similarity"] == 0.0
    assert out[0]["score"] == out[0]["lexical"]


def test_hybrid_blend_promotes_semantic_match(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    # Both share the query's lexical terms equally, but only ``vec_match`` is a
    # near neighbor of the query vector -> it must rank first under the blend.
    vec_match = _know(conn, "vecmatch", "alpha beta gamma", vector=[1.0, 0.0, 0.0])
    vec_off = _know(conn, "vecoff", "alpha beta gamma", vector=[0.0, 1.0, 0.0])
    out = hybrid_knowledge_search(
        conn, "alpha beta gamma", query_vector=[1.0, 0.0, 0.0], semantic_weight=0.5
    )
    ids = [r["knowledge_id"] for r in out]
    assert ids[0] == vec_match
    assert vec_off in ids
    match_row = next(r for r in out if r["knowledge_id"] == vec_match)
    off_row = next(r for r in out if r["knowledge_id"] == vec_off)
    assert match_row["score"] > off_row["score"]
    # Lexical component is identical; the vector broke the tie.
    assert abs(match_row["lexical"] - off_row["lexical"]) < 1e-9


def test_hybrid_row_without_vector_scores_lexical_only(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    # Query vector present, but this row has no stored embedding -> lexical-only.
    kid = _know(conn, "novec", "kafka consumer lag alerting")
    out = hybrid_knowledge_search(conn, "kafka lag", query_vector=[1.0, 0.0])
    row = next(r for r in out if r["knowledge_id"] == kid)
    # No stored vector -> semantic component is absent; score collapses to lexical.
    assert row["similarity"] == 0.0
    assert row["lexical"] > 0.0
    assert abs(row["score"] - row["lexical"]) < 1e-9


def test_hybrid_excludes_private(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    priv = _know(conn, "priv", "secret matching terms", scope="private", vector=[1.0, 0.0])
    hit = _know(conn, "hit", "matching terms here", vector=[1.0, 0.0])
    out = hybrid_knowledge_search(conn, "matching terms", query_vector=[1.0, 0.0])
    ids = {r["knowledge_id"] for r in out}
    assert hit in ids
    # A private row is never surfaced even when it is a strong lexical+vector match.
    assert priv not in ids
    assert all(r["scope"] != "private" for r in out)
