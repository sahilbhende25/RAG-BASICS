"""The capstone: an end-to-end RAG pipeline tying every stage together.

    question
      -> semantic cache check (short-circuit on hit)
      -> hybrid retrieve (dense + BM25, RRF)
      -> cross-encoder rerank (optional)
      -> grounded generation with citations
      -> cache the answer

This is the object the API (Hour 6) will expose.
"""

from __future__ import annotations

import time
from typing import Optional

from lexrag.config import Settings
from lexrag.embeddings.base import Embedder
from lexrag.generation.generator import Generator
from lexrag.generation.semantic_cache import SemanticCache
from lexrag.models import Answer, Chunk
from lexrag.retrieval.dense import DenseRetriever
from lexrag.retrieval.hybrid import HybridRetriever
from lexrag.retrieval.rerank import Reranker
from lexrag.retrieval.sparse import BM25Retriever
from lexrag.stores.base import VectorStore


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        embedder: Embedder,
        chunks: list[Chunk],
        reranker: Optional[Reranker] = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.dense = DenseRetriever(store, embedder)
        self.sparse = BM25Retriever(chunks)
        self.hybrid = HybridRetriever(self.dense, self.sparse)
        self.reranker = reranker
        self.generator = Generator(settings)
        self.cache = SemanticCache(threshold=settings.semantic_cache_threshold)

    def answer(
        self,
        question: str,
        filters: Optional[dict] = None,
        top_k: Optional[int] = None,
        use_cache: bool = True,
    ) -> Answer:
        start = time.perf_counter()
        top_k = top_k or self.settings.rerank_top_k

        # 1) Semantic cache. We reuse the query embedding for retrieval below.
        query_vector = self.embedder.embed_query(question)
        if use_cache:
            hit = self.cache.get(query_vector)
            if hit is not None:
                cached = hit.model_copy(
                    update={
                        "cached": True,
                        "latency_ms": (time.perf_counter() - start) * 1000,
                    }
                )
                return cached

        # 2) Retrieve wide (hybrid), then 3) rerank narrow (optional).
        candidates = self.hybrid.retrieve(
            question, top_k=self.settings.retrieval_top_k, filters=filters
        )
        if self.reranker is not None:
            contexts = self.reranker.rerank(question, candidates, top_k=top_k)
        else:
            contexts = candidates[:top_k]

        # 4) Grounded generation.
        answer = self.generator.generate(question, contexts)
        answer.latency_ms = (time.perf_counter() - start) * 1000

        # 5) Cache for future semantically-similar queries.
        if use_cache:
            self.cache.set(query_vector, answer)
        return answer
