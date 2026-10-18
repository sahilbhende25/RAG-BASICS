"""Tests for retrieval metrics (pure functions, no network)."""

from __future__ import annotations

from lexrag.eval.metrics import (
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieved_doc_ids,
)
from lexrag.models import Chunk, ScoredChunk


def _ctx(doc_id: str, ordinal: int = 0) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"{doc_id}::{ordinal}",
        doc_id=doc_id,
        text="t",
        ordinal=ordinal,
        token_count=1,
        title=doc_id,
        source="s",
    )
    return ScoredChunk(chunk=chunk, score=1.0, retriever="hybrid")


def test_retrieved_doc_ids_dedupes_preserving_order():
    contexts = [_ctx("a", 0), _ctx("a", 1), _ctx("b", 0)]
    assert retrieved_doc_ids(contexts, k=5) == ["a", "b"]


def test_retrieved_doc_ids_respects_k_over_chunks():
    contexts = [_ctx("a", 0), _ctx("b", 0), _ctx("c", 0)]
    assert retrieved_doc_ids(contexts, k=2) == ["a", "b"]


def test_hit_at_k():
    assert hit_at_k(["a", "b"], ["b"]) == 1.0
    assert hit_at_k(["a", "b"], ["z"]) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b"], ["a", "b"]) == 1.0
    assert recall_at_k(["a"], ["a", "b"]) == 0.5
    assert recall_at_k(["a"], []) == 1.0  # nothing to find


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], ["a"], k=3) == 1 / 3
    assert precision_at_k(["a", "b"], ["a", "b"], k=2) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "b"], ["a"]) == 0.5  # first relevant at rank 2
    assert reciprocal_rank(["a", "x"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0  # none relevant
