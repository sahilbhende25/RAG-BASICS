"""Tests for the semantic answer cache (no network)."""

from __future__ import annotations

from lexrag.generation.semantic_cache import SemanticCache
from lexrag.models import Answer


def _answer(text: str) -> Answer:
    return Answer(question="q", answer=text)


def test_empty_cache_misses():
    assert SemanticCache().get([1.0, 0.0, 0.0]) is None


def test_identical_vector_hits():
    cache = SemanticCache(threshold=0.95)
    cache.set([1.0, 0.0, 0.0], _answer("cached"))
    hit = cache.get([1.0, 0.0, 0.0])
    assert hit is not None and hit.answer == "cached"


def test_near_duplicate_hits_within_threshold():
    cache = SemanticCache(threshold=0.9)
    cache.set([1.0, 0.0, 0.0], _answer("cached"))
    # Vector very close in direction -> cosine ~0.9998 >= threshold.
    assert cache.get([0.98, 0.02, 0.0]) is not None


def test_orthogonal_vector_misses():
    cache = SemanticCache(threshold=0.9)
    cache.set([1.0, 0.0, 0.0], _answer("cached"))
    assert cache.get([0.0, 1.0, 0.0]) is None  # cosine 0 < threshold


def test_returns_nearest_of_several():
    cache = SemanticCache(threshold=0.8)
    cache.set([1.0, 0.0, 0.0], _answer("x-axis"))
    cache.set([0.0, 1.0, 0.0], _answer("y-axis"))
    hit = cache.get([0.9, 0.1, 0.0])
    assert hit is not None and hit.answer == "x-axis"
