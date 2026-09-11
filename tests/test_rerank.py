from __future__ import annotations

import sys

import pytest

from ocbrain import rerank as rerank_module
from ocbrain.config import RerankConfig
from ocbrain.rerank import CrossEncoderScorer, RerankUnavailable, get_scorer, rerank, reset_scorers


class MarkerScorer:
    def __init__(self, marker: str = "marker") -> None:
        self.marker = marker
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        return [1.0 if self.marker in document else 0.0 for document in documents]


def _items(count: int, *, marker_index: int | None = None) -> list[dict]:
    items = []
    for index in range(count):
        body = f"note {index}"
        if index == marker_index:
            body = f"note {index} marker"
        items.append({"belief_id": f"b{index}", "body": body, "ranking": {"fused": index}})
    return items


def test_disabled_by_default_leaves_the_order_alone() -> None:
    items = _items(6, marker_index=4)
    out, info = rerank("probe", items, config=RerankConfig(), scorer=MarkerScorer())
    assert out == items
    assert info == {"mode": "off"}
    assert [item["belief_id"] for item in out] == [f"b{i}" for i in range(6)]
    assert "rerank_score" not in out[4]["ranking"]


def test_applied_promotes_a_marked_item_and_keeps_the_tail() -> None:
    items = _items(8, marker_index=3)
    scorer = MarkerScorer()
    clock = iter([10.0, 10.25])
    out, info = rerank(
        "probe",
        items,
        config=RerankConfig(enabled=True, candidates=4),
        scorer=scorer,
        now=lambda: next(clock),
    )
    assert [item["belief_id"] for item in out] == ["b3", "b0", "b1", "b2", "b4", "b5", "b6", "b7"]
    assert out[0]["ranking"]["pre_rerank_rank"] == 4
    assert out[0]["ranking"]["rerank_score"] == 1.0
    assert out[1]["ranking"]["pre_rerank_rank"] == 1
    assert out[1]["ranking"]["rerank_score"] == 0.0
    assert all("rerank_score" in item["ranking"] for item in out[:4])
    assert all("rerank_score" not in item["ranking"] for item in out[4:])
    assert info == {
        "mode": "applied",
        "model": "BAAI/bge-reranker-v2-m3",
        "candidates": 4,
        "latency_ms": 250,
    }
    assert scorer.calls == [("probe", ["note 0", "note 1", "note 2", "note 3 marker"])]


def test_min_candidates_skips_a_short_ranking() -> None:
    items = _items(2, marker_index=1)
    out, info = rerank(
        "probe",
        items,
        config=RerankConfig(enabled=True, min_candidates=3),
        scorer=MarkerScorer(),
    )
    assert out == items
    assert info == {"mode": "skipped", "reason": "too_few_candidates"}


def test_candidates_cap_leaves_beyond_the_head_unscored() -> None:
    items = _items(30, marker_index=29)
    scorer = MarkerScorer()
    out, info = rerank(
        "probe",
        items,
        config=RerankConfig(enabled=True, candidates=3),
        scorer=scorer,
    )
    assert info["mode"] == "applied"
    assert info["candidates"] == 3
    assert [item["belief_id"] for item in out] == [f"b{i}" for i in range(30)]
    assert out[29]["belief_id"] == "b29"
    assert "rerank_score" not in out[29]["ranking"]
    assert len(scorer.calls[0][1]) == 3


def test_unavailable_scorer_keeps_the_fused_order(monkeypatch) -> None:
    items = _items(6, marker_index=4)

    def _unavailable(_config):
        raise RerankUnavailable("extra_not_installed")

    monkeypatch.setattr(rerank_module, "get_scorer", _unavailable)
    out, info = rerank("probe", items, config=RerankConfig(enabled=True))
    assert out == items
    assert info == {"mode": "unavailable", "reason": "extra_not_installed"}


def test_scorer_error_is_reported_not_raised() -> None:
    items = _items(6, marker_index=4)

    class Boom:
        def score(self, query: str, documents: list[str]) -> list[float]:
            raise ValueError("bad batch")

    out, info = rerank("probe", items, config=RerankConfig(enabled=True), scorer=Boom())
    assert out == items
    assert info == {"mode": "failed", "reason": "ValueError"}


def test_cross_encoder_reports_the_missing_extra(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RerankUnavailable) as excinfo:
        CrossEncoderScorer("BAAI/bge-reranker-v2-m3", "cpu")
    assert str(excinfo.value) == "extra_not_installed"


def test_scorer_cache_is_keyed_and_clearable(monkeypatch) -> None:
    built: list[tuple[str, str]] = []

    class FakeScorer:
        def __init__(self, model: str, device: str = "auto") -> None:
            built.append((model, device))

        def score(self, query: str, documents: list[str]) -> list[float]:
            return [0.0 for _ in documents]

    monkeypatch.setattr(rerank_module, "CrossEncoderScorer", FakeScorer)
    reset_scorers()
    try:
        config = RerankConfig(enabled=True, device="cpu")
        first = get_scorer(config)
        second = get_scorer(config)
        assert first is second
        assert built == [("BAAI/bge-reranker-v2-m3", "cpu")]

        get_scorer(RerankConfig(enabled=True, device="cpu", model="other/model"))
        assert len(built) == 2

        reset_scorers()
        get_scorer(config)
        assert len(built) == 3
    finally:
        reset_scorers()
