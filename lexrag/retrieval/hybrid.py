"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Problem: dense scores (cosine ~0-1) and BM25 scores (unbounded) live on totally
different scales, so you can't just add them. RRF sidesteps this by fusing on
**rank, not score**:

    rrf_score(d) = sum over retrievers of  1 / (k + rank_r(d))

where rank is the item's 0-based position in that retriever's list and k is a
constant (60 is the canonical default from the original paper). Properties:
  - Scale-free: only ordering matters, so no fragile score normalization.
  - Rewards items that rank highly in *multiple* retrievers.
  - k dampens the influence of very top ranks (rank 0 vs 1 isn't 2x).

This is the workhorse fusion method behind most production hybrid search.
"""

from __future__ import annotations

from lexrag.models import Chunk, ScoredChunk
from lexrag.retrieval.dense import DenseRetriever
from lexrag.retrieval.sparse import BM25Retriever


def reciprocal_rank_fusion(
    rankings: list[list[ScoredChunk]],
    k: int = 60,
    top_k: int = 20,
) -> list[ScoredChunk]:
    fused_scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}

    for ranking in rankings:
        for rank, scored in enumerate(ranking):
            cid = scored.chunk.chunk_id
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_by_id[cid] = scored.chunk

    ordered = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        ScoredChunk(chunk=chunk_by_id[cid], score=score, retriever="hybrid")
        for cid, score in ordered[:top_k]
    ]


class HybridRetriever:
    """Dense + BM25, fused with RRF.

    We pull a *wide* candidate set from each retriever (candidate_k) before
    fusing, because a doc might rank, say, #15 in dense but #2 in sparse — we'd
    miss it if each only returned its top few.
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[ScoredChunk]:
        dense_hits = self.dense.retrieve(query, top_k=self.candidate_k, filters=filters)
        sparse_hits = self.sparse.retrieve(query, top_k=self.candidate_k, filters=filters)
        return reciprocal_rank_fusion(
            [dense_hits, sparse_hits], k=self.rrf_k, top_k=top_k
        )
