"""OpenAI embedder with batching, caching, and Matryoshka dimension control.

Interview-relevant details baked in:
  - **Batching**: one API call for many texts amortizes latency and cost.
  - **Caching**: only cache-miss texts hit the API (see EmbeddingCache).
  - **dimensions param**: text-embedding-3-* are Matryoshka models, so we can
    request a shorter vector (e.g. 512) to trade a little quality for much less
    memory/compute in the vector index.
"""

from __future__ import annotations

from openai import OpenAI

from lexrag.config import Settings
from lexrag.embeddings.base import Embedder
from lexrag.embeddings.cache import EmbeddingCache

# OpenAI accepts up to 2048 inputs/request; 128 keeps requests small & robust.
_BATCH = 128


class OpenAIEmbedder(Embedder):
    def __init__(self, settings: Settings, cache: EmbeddingCache | None = None) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.dim = settings.embedding_dim
        self.cache = cache

    def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dim,
        )
        # The API guarantees output order matches input order.
        return [item.embedding for item in resp.data]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        misses: list[str] = []
        miss_positions: list[int] = []

        # 1) Serve cache hits; collect misses.
        for i, text in enumerate(texts):
            cached = self.cache.get(self.model, self.dim, text) if self.cache else None
            if cached is None:
                misses.append(text)
                miss_positions.append(i)
            else:
                results[i] = cached

        # 2) Embed misses in batches, fill results + cache.
        for start in range(0, len(misses), _BATCH):
            batch = misses[start : start + _BATCH]
            vectors = self._embed_raw(batch)
            for offset, vector in enumerate(vectors):
                pos = miss_positions[start + offset]
                results[pos] = vector
                if self.cache:
                    self.cache.set(self.model, self.dim, texts[pos], vector)

        return [v for v in results if v is not None]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
