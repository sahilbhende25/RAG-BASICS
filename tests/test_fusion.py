"""Tests for Reciprocal Rank Fusion — a pure function, no network/deps."""

from __future__ import annotations

from lexrag.models import Chunk, ScoredChunk
from lexrag.retrieval.hybrid import reciprocal_rank_fusion


def _sc(cid: str, score: float, retriever: str) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=cid,
        doc_id=cid.split("::")[0],
        text=f"text-{cid}",
        ordinal=0,
        token_count=1,
        title=cid,
        source="s",
    )
    return ScoredChunk(chunk=chunk, score=score, retriever=retriever)


def test_item_ranked_high_in_both_wins():
    # 'b' is #2 in dense and #1 in sparse -> should beat items that top only one.
    dense = [_sc("a::0", 0.9, "dense"), _sc("b::0", 0.8, "dense")]
    sparse = [_sc("b::0", 10.0, "sparse"), _sc("c::0", 9.0, "sparse")]
    fused = reciprocal_rank_fusion([dense, sparse], k=60, top_k=3)
    assert fused[0].chunk.chunk_id == "b::0"
    assert fused[0].retriever == "hybrid"


def test_fusion_is_scale_invariant():
    # Sparse scores are 1000x dense scores; RRF must ignore magnitude (rank only).
    dense = [_sc("a::0", 0.9, "dense"), _sc("b::0", 0.1, "dense")]
    sparse = [_sc("a::0", 9000.0, "sparse"), _sc("b::0", 1.0, "sparse")]
    fused = reciprocal_rank_fusion([dense, sparse], k=60, top_k=2)
    assert fused[0].chunk.chunk_id == "a::0"  # top of both lists


def test_dedupes_across_retrievers():
    dense = [_sc("a::0", 0.9, "dense")]
    sparse = [_sc("a::0", 5.0, "sparse")]
    fused = reciprocal_rank_fusion([dense, sparse], k=60, top_k=5)
    assert len(fused) == 1  # same chunk, merged


def test_respects_top_k():
    dense = [_sc(f"d{i}::0", 1.0 - i * 0.1, "dense") for i in range(10)]
    fused = reciprocal_rank_fusion([dense], k=60, top_k=3)
    assert len(fused) == 3


def test_k_dampens_top_rank_dominance():
    # With larger k, the gap between rank 0 and rank 1 contributions shrinks.
    small_k = 1.0 / (1 + 0) - 1.0 / (1 + 1)   # k=1
    large_k = 1.0 / (60 + 0) - 1.0 / (60 + 1)  # k=60
    assert large_k < small_k
