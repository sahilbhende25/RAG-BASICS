"""Tests for BM25 sparse retrieval (rank-bm25, no network)."""

from __future__ import annotations

from lexrag.models import Chunk
from lexrag.retrieval.sparse import BM25Retriever


def _chunk(cid: str, text: str, juris: str = "US-CA", dtype: str = "case") -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=cid.split("::")[0],
        text=text,
        ordinal=0,
        token_count=len(text.split()),
        title=cid,
        source="s",
        jurisdiction=juris,
        doc_type=dtype,
    )


def _retriever() -> BM25Retriever:
    return BM25Retriever(
        [
            _chunk("a::0", "The Nguyen v Pacific Logistics remote work case."),
            _chunk("b::0", "Overtime pay and misclassification of employees.", dtype="case"),
            _chunk("c::0", "Vacation accrual and carryover rules.", juris="US-NY", dtype="statute"),
        ]
    )


def test_exact_keyword_ranks_first():
    hits = _retriever().retrieve("Nguyen Pacific Logistics", top_k=3)
    assert hits[0].chunk.chunk_id == "a::0"
    assert hits[0].retriever == "sparse"


def test_post_filter_by_metadata():
    hits = _retriever().retrieve("rules", top_k=5, filters={"doc_type": "statute"})
    assert all(h.chunk.doc_type == "statute" for h in hits)
    assert {h.chunk.chunk_id for h in hits} == {"c::0"}


def test_top_k_limit():
    hits = _retriever().retrieve("employees case rules", top_k=2)
    assert len(hits) <= 2
