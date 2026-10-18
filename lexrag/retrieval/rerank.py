"""Cross-encoder reranking (the second stage of two-stage retrieval).

Bi-encoder (what we use for dense retrieval): encodes the query and each chunk
*independently* into vectors, then compares. Fast and precomputable, but the
query and chunk never "see" each other, so scoring is coarse.

Cross-encoder (reranker): feeds (query, chunk) **together** through a
transformer so every query token can attend to every chunk token. Far more
accurate relevance scores — but you must run the model per candidate at query
time, so it's too slow to score the whole corpus.

The standard pattern, implemented here:
  1. Retrieve a wide candidate set cheaply (hybrid, top ~20-50).
  2. Rerank just those candidates with the cross-encoder.
  3. Keep the top few for the LLM.

This "retrieve wide, rerank narrow" is one of the biggest quality wins in RAG
and a near-guaranteed interview topic.
"""

from __future__ import annotations

from typing import Protocol

from lexrag.models import ScoredChunk


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


class CrossEncoderReranker:
    """Wraps a sentence-transformers CrossEncoder.

    Default model `ms-marco-MiniLM-L-6-v2` is small (~80MB) and CPU-friendly.
    The import is lazy so merely importing this module doesn't pull torch +
    transformers unless you actually rerank.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [
            ScoredChunk(
                chunk=candidates[i].chunk,
                score=float(scores[i]),
                retriever="rerank",
            )
            for i in order[:top_k]
        ]
