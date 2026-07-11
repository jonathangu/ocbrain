from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ocbrain.embed import decode_embedding, knowledge_text
from ocbrain.events import projected_rows_as_of, sha256_text
from ocbrain.scope import ScopeContext, ScopeTag, scope_match
from ocbrain.text import find_probable_injection, redact_secrets

# Default lexical/semantic blend when a query vector is available (0.5/0.5).
DEFAULT_SEMANTIC_WEIGHT = 0.5
CATALOG_STUB_WEIGHT = 0.15

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
    "what",
    "how",
    "why",
    "when",
    "where",
    "who",
    "which",
    "does",
    "did",
    "should",
    "could",
    "would",
    "about",
    "current",
    "is",
    "it",
    "its",
    "my",
    "your",
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
        catalog_weight = CATALOG_STUB_WEIGHT if looks_like_catalog_stub(row["body"]) else 1.0
        score = relevance * scope_weight * confidence * pinned_weight * catalog_weight
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
    if context.repo:
        item_dicts = _merge_repo_fts_fallback(
            conn,
            query,
            context=context,
            existing=item_dicts,
            limit=limit,
        )
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
        "applied_global": [item for item in item_dicts if item["scope"]["scope_type"] == "global"],
        "excluded": excluded,
        "excluded_count": excluded_count,
        "token_budget": estimate_tokens([item["body"] for item in item_dicts]),
    }


def _merge_repo_fts_fallback(
    conn: sqlite3.Connection,
    query: str,
    *,
    context: ScopeContext,
    existing: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Add repo-bounded FTS results when scoped beliefs are only catalog stubs."""
    from ocbrain.db import PUBLIC_SCOPES, knowledge_evidence, search

    rows = search(
        conn,
        query,
        limit=max(limit * 2, 10),
        scopes=PUBLIC_SCOPES,
        filters={"repo": context.repo},
    )
    merged = [dict(item, source="event_core") for item in existing]
    seen = {str(item["belief_id"]) for item in merged}
    for item in _repo_file_items(query, context.repo, limit=limit):
        if item["belief_id"] in seen:
            continue
        merged.append(item)
        seen.add(str(item["belief_id"]))
    for rank, row in enumerate(rows):
        object_id = str(row["doc_id"])
        if object_id in seen:
            continue
        body = " ".join(
            part for part in (str(row["title"] or ""), str(row["snippet"] or "")) if part
        )
        if not body.strip():
            continue
        relevance = lexical_relevance(query, body)
        if relevance < 0.4:
            continue
        score = max(0.21, 0.6 - rank * 0.02)
        evidence_ids = [object_id] if object_id.startswith("evd_") else []
        if object_id.startswith("know_"):
            evidence_ids = [str(item["id"]) for item in knowledge_evidence(conn, object_id)]
        merged.append(
            {
                "belief_id": object_id,
                "body": body,
                "scope": {
                    "scope_type": "repo",
                    "scope_id": f"repo:{context.repo}",
                    "visibility": "internal",
                    "egress_policy": "local_only",
                    "provenance": "inferred",
                },
                "score": round(score, 6),
                "relevance": round(relevance, 6),
                "scope_weight": 1.0,
                "confidence": 0.5,
                "confidence_band": "moderate",
                "evidence_ids": evidence_ids,
                "source": "fts_repo_fallback",
            }
        )
        seen.add(object_id)
    return sorted(merged, key=lambda item: (-float(item["score"]), str(item["belief_id"])))[:limit]


@lru_cache(maxsize=8)
def _repo_documents(repo: str) -> tuple[tuple[str, str, str], ...]:
    """Load a bounded, source-hashed local repo corpus for retrieval fallback."""
    root = Path(repo).expanduser().resolve()
    candidates: set[Path] = set()
    for name in ("README.md", "CHANGELOG.md", "NOTICE", "LICENSE"):
        path = root / name
        if path.is_file():
            candidates.add(path)
    for pattern in ("docs/**/*.md", "src/ocbrain/**/*.py"):
        candidates.update(path for path in root.glob(pattern) if path.is_file())
    documents: list[tuple[str, str, str]] = []
    for path in sorted(candidates)[:1000]:
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        if len(payload) > 512_000:
            payload = payload[:512_000]
        text = payload.decode("utf-8", errors="replace")
        relative = str(path.relative_to(root))
        documents.append((relative, text, sha256_text(text)))
    return tuple(documents)


def _repo_file_items(query: str, repo: str, *, limit: int) -> list[dict[str, Any]]:
    query_terms = meaningful_terms(query)
    if not query_terms:
        return []
    candidates: list[tuple[float, str, str, str]] = []
    minimum_overlap = 2 if len(query_terms) >= 3 else 1
    documents = _repo_documents(repo)
    indexed_documents = [
        (relative, text, digest, meaningful_terms(text)) for relative, text, digest in documents
    ]
    document_frequency = {
        term: sum(term in doc_terms for _relative, _text, _digest, doc_terms in indexed_documents)
        for term in query_terms
    }
    ranked_terms = sorted(document_frequency.items(), key=lambda item: (item[1], item[0]))
    present_terms = [item for item in ranked_terms if item[1] > 0]
    # When the repo contains at least three query concepts, use the rarest
    # present concepts. Otherwise retain zero-frequency terms as a conservative
    # negative-query guard (for example, a query about a foreign medical record
    # should not match generic privacy documentation on only "private/project").
    distinctive_source = present_terms if len(present_terms) >= 3 else ranked_terms
    distinctive = {term for term, _frequency in distinctive_source[:2]}
    for relative, text, digest, doc_terms in indexed_documents:
        if not (doc_terms & distinctive):
            continue
        paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        chunks: list[str] = []
        for index, paragraph in enumerate(paragraphs):
            if re.match(r"^#{1,4}\s", paragraph) and index + 1 < len(paragraphs):
                chunks.append(f"{paragraph}\n\n{paragraphs[index + 1]}")
            elif not re.match(r"^#{1,4}\s", paragraph):
                chunks.append(paragraph)
        best: tuple[float, str] | None = None
        for chunk in chunks:
            if len(chunk) < 60 or len(chunk) > 20_000:
                continue
            overlap = query_terms & meaningful_terms(chunk)
            if len(overlap) < minimum_overlap:
                continue
            score = lexical_relevance(query, chunk)
            if best is None or score > best[0]:
                best = (score, chunk)
        if best is None:
            continue
        snippet = redact_secrets(best[1])[:1200]
        if find_probable_injection(snippet):
            continue
        candidates.append((best[0], relative, snippet, digest))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    items: list[dict[str, Any]] = []
    for rank, (relevance, relative, snippet, digest) in enumerate(candidates[:limit]):
        items.append(
            {
                "belief_id": f"artifact:{relative}:{digest[:12]}",
                "body": f"{relative}\n{snippet}",
                "scope": {
                    "scope_type": "repo",
                    "scope_id": f"repo:{repo}",
                    "visibility": "internal",
                    "egress_policy": "local_only",
                    "provenance": "explicit",
                },
                "score": round(max(0.7, 0.95 - rank * 0.03), 6),
                "relevance": round(relevance, 6),
                "scope_weight": 1.0,
                "confidence": 0.9,
                "confidence_band": "strong",
                "evidence_ids": [],
                "artifact_refs": [{"path": relative, "sha256": digest}],
                "source": "repo_file",
            }
        )
    return items


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
        filtered = [row for row in rows if meaningful_terms(str(row["body"])) & query_terms]
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
    query_terms = meaningful_terms(query)
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


def looks_like_catalog_stub(body: str) -> bool:
    """Identify path-only harvested catalog beliefs without hiding real docs."""
    text = " ".join(str(body).split())
    if len(text) > 320 or len(text.split()) > 18:
        return False
    path_only_shape = bool(
        re.match(
            r"^(?:[^.!?\n]{0,120}\s+)?(?:/Users/|/private/|~/|[A-Za-z]:\\)\S+$",
            text,
            re.I,
        )
    )
    has_file_suffix = bool(re.search(r"\.[A-Za-z0-9]{1,8}(?:\s|$)", text))
    return path_only_shape and has_file_suffix


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
