"""Optional cross-encoder second stage over the fused ranking.

Fusion decides what is a candidate; a cross-encoder reading query and candidate
together decides which of them actually fits. The stage is disabled by default
and its dependency (``sentence-transformers``) is an extra, so it is imported
lazily: when the extra is absent, the model cannot load, or the scorer raises,
retrieval returns exactly what fusion produced and the packet records why.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ocbrain.config import RerankConfig


class RerankUnavailable(RuntimeError):
    """The reranker cannot run; the message names the part that failed."""


class Scorer(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...


_SCORERS: dict[tuple[str, str], Scorer] = {}


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 - no torch means no accelerator, not a failure
        return "cpu"
    return "cpu"


def _document_text(item: dict[str, Any], max_chars: int) -> str:
    text = item.get("body") or item.get("excerpt") or ""
    return str(text)[:max_chars]


class CrossEncoderScorer:
    def __init__(self, model: str, device: str = "auto") -> None:
        self.model_name = model
        self.device = _resolve_device(device)
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankUnavailable("extra_not_installed") from exc
        try:
            self.model = CrossEncoder(model, device=self.device)
        except Exception as exc:  # noqa: BLE001 - any load failure is unavailability
            raise RerankUnavailable(f"model_load_failed: {type(exc).__name__}") from exc

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = list(zip([query] * len(documents), documents, strict=True))
        return [float(value) for value in self.model.predict(pairs)]


def get_scorer(config: RerankConfig) -> Scorer:
    """Return the cached scorer for this model and device, loading it if needed."""
    device = _resolve_device(config.device)
    key = (config.model, device)
    scorer = _SCORERS.get(key)
    if scorer is None:
        scorer = CrossEncoderScorer(config.model, device)
        _SCORERS[key] = scorer
    return scorer


def reset_scorers() -> None:
    _SCORERS.clear()


def rerank(
    query: str,
    items: list[dict[str, Any]],
    *,
    config: RerankConfig,
    scorer: Scorer | None = None,
    now: Callable[[], float] = time.monotonic,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-score the fused head and return ``(items, provenance)``.

    Nothing here raises into retrieval: every failure returns the input order
    plus the reason, so a scorer problem degrades to the fused ranking.
    """
    if not config.enabled:
        return items, {"mode": "off"}
    head_count = min(len(items), max(0, int(config.candidates)))
    if len(items) < config.min_candidates or head_count == 0:
        return items, {"mode": "skipped", "reason": "too_few_candidates"}
    head = items[:head_count]
    tail = items[head_count:]
    documents = [_document_text(item, config.max_document_chars) for item in head]
    started = now()
    try:
        active = scorer if scorer is not None else get_scorer(config)
        scores = active.score(query, documents)
    except RerankUnavailable as exc:
        return items, {"mode": "unavailable", "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 - a scorer failure must not break retrieval
        return items, {"mode": "failed", "reason": type(exc).__name__}
    if len(scores) != len(head):
        return items, {"mode": "failed", "reason": "score_count_mismatch"}
    latency_ms = int(max(0.0, now() - started) * 1000)
    order = sorted(range(len(head)), key=lambda index: (-scores[index], index))
    reranked: list[dict[str, Any]] = []
    for index in order:
        item = head[index]
        detail = item.setdefault("ranking", {})
        detail["rerank_score"] = round(float(scores[index]), 6)
        detail["pre_rerank_rank"] = index + 1
        reranked.append(item)
    return reranked + tail, {
        "mode": "applied",
        "model": config.model,
        "candidates": len(head),
        "latency_ms": latency_ms,
    }
