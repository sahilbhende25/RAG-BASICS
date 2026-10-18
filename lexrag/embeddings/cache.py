"""Content-addressed embedding cache.

Embeddings are deterministic for a given (model, dimension, text). So we key a
persistent cache by a hash of exactly those three things. Benefits:
  - Re-indexing after a code change doesn't re-pay for unchanged chunks.
  - Repeated/duplicate chunks are embedded once.
  - Cost + latency drop dramatically on warm runs.

This is the cheapest, highest-value optimization in a RAG pipeline and a
guaranteed interview talking point ("where do you cache, and what's the key?").
"""

from __future__ import annotations

import hashlib

from diskcache import Cache


class EmbeddingCache:
    def __init__(self, directory: str) -> None:
        # diskcache is process/thread-safe and backed by SQLite on disk.
        self._cache = Cache(directory)

    @staticmethod
    def _key(model: str, dim: int, text: str) -> str:
        # Include model + dim so we never mix vectors from different models.
        payload = f"{model}|{dim}|{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, model: str, dim: int, text: str) -> list[float] | None:
        return self._cache.get(self._key(model, dim, text))

    def set(self, model: str, dim: int, text: str, vector: list[float]) -> None:
        self._cache.set(self._key(model, dim, text), vector)
