"""Embedder interface.

Programming to an interface (not a concrete vendor) is what lets us swap
OpenAI for a local model (bge/e5) or Cohere/Voyage later without touching the
retrieval or pipeline code. Classic dependency inversion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    #: output dimensionality of the vectors this embedder produces
    dim: int

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents/passages."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query.

        Kept separate from embed_texts because some models use *asymmetric*
        embeddings (different prefixes/instructions for queries vs passages).
        """
