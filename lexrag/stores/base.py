"""Vector store interface.

Abstracting the store means the retrieval layer doesn't care whether we're on
Qdrant, pgvector, FAISS, or a managed service. Swapping is a config change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from lexrag.models import Chunk, ScoredChunk


class VectorStore(ABC):
    @abstractmethod
    def ensure_collection(self, recreate: bool = False) -> None: ...

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: Optional[dict] = None,
    ) -> list[ScoredChunk]: ...
