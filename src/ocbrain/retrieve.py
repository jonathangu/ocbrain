from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any

from ocbrain.embed import decode_embedding, knowledge_text
from ocbrain.events import projected_rows_as_of
from ocbrain.scope import ScopeContext, ScopeTag, scope_match

# Default lexical/semantic blend when a query vector is available (0.5/0.5).
DEFAULT_SEMANTIC_WEIGHT = 0.5

NEGATION_TERMS = {"no", "not", "never", "without", "cannot", "can't", "doesn't", "isn't"}
STOP_TERMS = {
    "and",
    "are",
    "for",
    "from",
    "has",
    "into",
    "must",
    "not",
    "the",
    "that",
    "this",
    "uses",
    "with",
}


@dataclass(frozen=True)
class RetrievalItem:
    belief_id: str
    body: str
    scope: dict[str, Any]
    score: float
    relevance: float
    scope_weight: float
    confidence: float
    confidence_band: str
    evidence_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContradictionItem:
    belief_id: str
    other_belief_id: str
    score: float
    shared_terms: list[str]
    reasons: list[str]
    scope: dict[str, Any]
    other_scope: dict[str, Any]
    body: str
    other_body: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def retrieve(
    conn: sqlite3.Connection,
    query: str,
    *,
    context: ScopeContext | None = None,
    limit: int = 12,
    cross_scope: bool = False,
    at_ts: str | None = None,
) -> dict[str, Any]:
    context = context or ScopeContext()
    scored: list[RetrievalItem] = []
    excluded: list[dict[str, Any]] = []
    excluded_count = 0
    for row in belief_rows(conn, at_ts=at_ts):
        if row["status"] != "current":
            continue
        scope = ScopeTag(
            scope_type=row["scope_type"],
            scope_id=row["scope_id"],
            visibility=row["visibility"],
            egress_policy=row["egress_policy"],
        )
        scope_weight = scope_match(scope, context, cross_scope=cross_scope)
        if scope_weight == 0:
            excluded_count += 1
            if len(excluded) < limit:
                excluded.append(
                    {
                        "belief_id": row["belief_id"],
                        "scope": scope.to_dict(),
                        "reason": "scope_mismatch",
                    }
                )
            continue
        relevance = lexical_relevance(query, row["body"])
        if relevance == 0:
            continue
        confidence = float(row["confidence"] if row["confidence"] is not None else 0.5)
        pinned_weight = 1.1 if row["pinned"] else 1.0
        score = relevance * scope_weight * confidence * pinned_weight
        scored.append(
            RetrievalItem(
                belief_id=row["belief_id"],
                body=row["body"],
                scope=scope.to_dict(),
                score=round(score, 6),
                relevance=round(relevance, 6),
                scope_weight=scope_weight,
                confidence=confidence,
                confidence_band=row["confidence_band"] or confidence_band(confidence),
                evidence_ids=parse_json_list(row["evidence_ids"]),
            )
        )
    items = sorted(scored, key=lambda item: (-item.score, item.belief_id))[:limit]
    item_dicts = [item.to_dict() for item in items]
    return {
        "query": query,
        "context": context.to_dict(),
        "cross_scope": cross_scope,
        "at_ts": at_ts,
        "items": item_dicts,
        "contradictions": [
            item.to_dict()
            for item in rank_contradictions(
                conn,
                query,
                context=context,
                cross_scope=cross_scope,
                at_ts=at_ts,
                limit=limit,
            )
        ],
        "applied_global": [
            item for item in item_dicts if item["scope"]["scope_type"] == "global"
        ],
        "excluded": excluded,
        "excluded_count": excluded_count,
        "token_budget": estimate_tokens([item["body"] for item in item_dicts]),
    }


def rank_contradictions(
    conn: sqlite3.Connection,
    query: str,
    *,
    context: ScopeContext | None = None,
    cross_scope: bool = False,
    at_ts: str | None = None,
    limit: int = 12,
) -> list[ContradictionItem]:
    context = context or ScopeContext()
    query_terms = meaningful_terms(query)
    visible = contradiction_candidate_rows(
        visible_belief_rows(conn, context=context, cross_scope=cross_scope, at_ts=at_ts),
        query_terms=query_terms,
    )
    ranked: list[ContradictionItem] = []
    for index, row in enumerate(visible):
        for other in visible[index + 1 :]:
            item = score_contradiction_pair(row, other, query_terms)
            if item is not None:
                ranked.append(item)
    return sorted(ranked, key=lambda item: (-item.score, item.belief_id, item.other_belief_id))[
        :limit
    ]


def contradiction_candidate_rows(
    rows: list[dict[str, Any]], *, query_terms: set[str], max_rows: int = 200
) -> list[dict[str, Any]]:
    if not rows:
        return []
    if query_terms:
        filtered = [
            row for row in rows if meaningful_terms(str(row["body"])) & query_terms
        ]
        if filtered:
            rows = filtered
    return rows[:max_rows]


def visible_belief_rows(
    conn: sqlite3.Connection,
    *,
    context: ScopeContext,
    cross_scope: bool,
    at_ts: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in belief_rows(conn, at_ts=at_ts):
        if row["status"] != "current":
            continue
        scope = ScopeTag(
            scope_type=row["scope_type"],
            scope_id=row["scope_id"],
            visibility=row["visibility"],
            egress_policy=row["egress_policy"],
        )
        weight = scope_match(scope, context, cross_scope=cross_scope)
        if weight == 0:
            continue
        rows.append(
            {
                "belief_id": row["belief_id"],
                "body": row["body"],
                "scope": scope.to_dict(),
                "scope_weight": weight,
                "confidence": float(row["confidence"] if row["confidence"] is not None else 0.5),
            }
        )
    return rows


def score_contradiction_pair(
    row: dict[str, Any],
    other: dict[str, Any],
    query_terms: set[str],
) -> ContradictionItem | None:
    row_terms = meaningful_terms(row["body"])
    other_terms = meaningful_terms(other["body"])
    shared = sorted(row_terms & other_terms)
    if len(shared) < 2:
        return None
    reasons: list[str] = []
    if has_negation(row["body"]) != has_negation(other["body"]):
        reasons.append("negation_mismatch")
    if not reasons:
        return None
    query_overlap = len(query_terms & (row_terms | other_terms)) / max(len(query_terms), 1)
    shared_density = len(shared) / max(min(len(row_terms), len(other_terms)), 1)
    scope_weight = min(float(row["scope_weight"]), float(other["scope_weight"]))
    confidence = min(float(row["confidence"]), float(other["confidence"]))
    score = (1.0 + query_overlap + shared_density) * scope_weight * confidence
    return ContradictionItem(
        belief_id=str(row["belief_id"]),
        other_belief_id=str(other["belief_id"]),
        score=round(score, 6),
        shared_terms=shared[:12],
        reasons=reasons,
        scope=row["scope"],
        other_scope=other["scope"],
        body=str(row["body"]),
        other_body=str(other["body"]),
    )


def belief_rows(conn: sqlite3.Connection, *, at_ts: str | None = None):
    if at_ts is not None:
        return sorted(
            projected_rows_as_of(conn, at_ts=at_ts),
            key=lambda row: (int(row["pinned"]), row["last_compiled_at"], row["belief_id"]),
            reverse=True,
        )
    return list(
        conn.execute(
            """
            SELECT *
            FROM current_beliefs
            ORDER BY pinned DESC, last_compiled_at DESC, belief_id ASC
            """
        )
    )


def lexical_relevance(query: str, body: str) -> float:
    query_terms = set(terms(query))
    if not query_terms:
        return 0.0
    body_terms = terms(body)
    if not body_terms:
        return 0.0
    body_set = set(body_terms)
    overlap = query_terms & body_set
    if not overlap:
        return 0.0
    density = sum(body_terms.count(term) for term in overlap) / max(len(body_terms), 1)
    coverage = len(overlap) / len(query_terms)
    return coverage + density


def terms(text: str) -> list[str]:
    return re.findall(r"[\w-]{2,}", text.lower())


def meaningful_terms(text: str) -> set[str]:
    return {term for term in terms(text) if term not in STOP_TERMS | NEGATION_TERMS}


def has_negation(text: str) -> bool:
    return bool(set(terms(text)) & NEGATION_TERMS)


def estimate_tokens(texts: list[str]) -> int:
    # Keep one estimator for preview/search payloads. Four chars per token is a
    # rough upper-bound convention, good enough for local budgeting without a model dep.
    return sum(max(len(text) // 4, 1) for text in texts)


def parse_json_list(text: str) -> list[str]:
    import json

    value = json.loads(text)
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def confidence_band(confidence: float) -> str:
    if confidence >= 0.75:
        return "strong"
    if confidence >= 0.45:
        return "moderate"
    return "weak"


# --------------------------------------------------------------------------- #
# Semantic layer — vector scoring over embedded knowledge rows (v0.3)
#
# Embeddings live on the ``knowledge`` table (``knowledge.embedding``, written by
# :mod:`ocbrain.embed`), parallel to the event-sourced belief store the lexical
# ``retrieve`` above reads. These helpers score the query against stored vectors
# and blend that with the existing lexical score, falling back to lexical-only
# whenever a query vector or a row vector is absent.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NeighborItem:
    knowledge_id: str
    similarity: float
    lexical: float
    score: float
    body: str
    scope: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors; ``0.0`` for empty, mismatched, or zero-norm."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def blend_scores(
    lexical: float,
    semantic: float | None,
    *,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
) -> float:
    """Blend a lexical and semantic score.

    ``semantic`` of ``None`` means no vector was available (query or row) — the
    blend collapses to the lexical score so retrieval degrades gracefully to
    lexical-only. Otherwise the two are combined as
    ``(1 - w) * lexical + w * semantic``.
    """
    if semantic is None:
        return lexical
    weight = min(max(float(semantic_weight), 0.0), 1.0)
    return (1.0 - weight) * lexical + weight * semantic


def embedded_knowledge_rows(
    conn: sqlite3.Connection,
    *,
    include_private: bool = False,
    statuses: tuple[str, ...] = ("candidate", "current"),
) -> list[sqlite3.Row]:
    """Un-quarantined knowledge rows that carry a stored embedding vector."""
    placeholders = ",".join("?" for _ in statuses)
    clauses = [
        "embedding IS NOT NULL",
        "quarantine_reason IS NULL",
        f"status IN ({placeholders})",
    ]
    params: list[Any] = list(statuses)
    if not include_private:
        clauses.append("privacy_scope != 'private'")
    return list(
        conn.execute(
            f"SELECT * FROM knowledge WHERE {' AND '.join(clauses)}",  # noqa: S608 - fixed clause set
            params,
        )
    )


def semantic_neighbors(
    conn: sqlite3.Connection,
    query_vector: list[float],
    *,
    limit: int = 10,
    min_similarity: float = 0.0,
    include_private: bool = False,
) -> list[dict[str, Any]]:
    """Nearest embedded knowledge rows to ``query_vector`` by cosine similarity.

    Exported for autolabel attribution as the vector analog of the FTS
    ``search`` path: given a signal/query embedding, return the closest knowledge
    rows so a signal can be attributed semantically rather than lexically. Returns
    ``[]`` when the query vector is empty (caller falls back to lexical). Private
    rows are excluded by default (attribution is internal, but private vectors are
    never stored by :mod:`ocbrain.embed` anyway — this is defense in depth).
    """
    if not query_vector:
        return []
    scored: list[NeighborItem] = []
    for row in embedded_knowledge_rows(conn, include_private=include_private):
        vector = decode_embedding(row["embedding"])
        similarity = cosine_similarity(query_vector, vector)
        if similarity < min_similarity:
            continue
        scored.append(
            NeighborItem(
                knowledge_id=row["id"],
                similarity=round(similarity, 6),
                lexical=0.0,
                score=round(similarity, 6),
                body=knowledge_text(row),
                scope=row["privacy_scope"],
            )
        )
    ranked = sorted(scored, key=lambda item: (-item.similarity, item.knowledge_id))
    return [item.to_dict() for item in ranked[:limit]]


def hybrid_knowledge_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    query_vector: list[float] | None = None,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    limit: int = 12,
    include_private: bool = False,
    statuses: tuple[str, ...] = ("candidate", "current"),
) -> list[dict[str, Any]]:
    """Rank knowledge rows by a lexical + semantic blend over stored vectors.

    For every candidate row the lexical relevance of ``query`` against the row
    text is blended with the cosine similarity between ``query_vector`` and the
    row's stored embedding. When ``query_vector`` is ``None`` (no query embedding)
    or a row has no stored vector, that row is scored lexical-only — so the search
    always works, embeddings just sharpen it. Rows with a zero blended score are
    dropped.
    """
    placeholders = ",".join("?" for _ in statuses)
    clauses = ["quarantine_reason IS NULL", f"status IN ({placeholders})"]
    params: list[Any] = list(statuses)
    if not include_private:
        clauses.append("privacy_scope != 'private'")
    rows = conn.execute(
        f"SELECT * FROM knowledge WHERE {' AND '.join(clauses)}",  # noqa: S608 - fixed clause set
        params,
    ).fetchall()

    scored: list[NeighborItem] = []
    for row in rows:
        text = knowledge_text(row)
        lexical = lexical_relevance(query, text)
        semantic: float | None = None
        if query_vector:
            vector = decode_embedding(row["embedding"])
            if vector:
                semantic = cosine_similarity(query_vector, vector)
        score = blend_scores(lexical, semantic, semantic_weight=semantic_weight)
        if score <= 0.0:
            continue
        scored.append(
            NeighborItem(
                knowledge_id=row["id"],
                similarity=round(semantic if semantic is not None else 0.0, 6),
                lexical=round(lexical, 6),
                score=round(score, 6),
                body=text,
                scope=row["privacy_scope"],
            )
        )
    ranked = sorted(scored, key=lambda item: (-item.score, item.knowledge_id))
    return [item.to_dict() for item in ranked[:limit]]
