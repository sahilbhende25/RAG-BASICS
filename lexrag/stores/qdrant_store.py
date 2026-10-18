"""Qdrant-backed vector store.

Why Qdrant for the interview story:
  - Production-grade ANN (HNSW graph index) with on-disk persistence.
  - First-class **payload filtering** — we can constrain the ANN search to,
    say, jurisdiction == "US-CA" in a single pass (pre-filtering), instead of
    retrieving then filtering (which wrecks top-k).
  - Supports sparse vectors too, which we'll exploit for hybrid search (Hr 3).

Distance = COSINE because embeddings encode direction/semantics, not magnitude.
(If vectors are L2-normalized, cosine == dot product, which is cheaper.)

ID note: Qdrant point ids must be uint or UUID. Our chunk_ids are strings like
"doc::3", so we map each to a deterministic UUIDv5 and keep the original
chunk_id in the payload. Deterministic means re-indexing the same chunk
overwrites its point instead of duplicating it (idempotent upserts).
"""

from __future__ import annotations

import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from lexrag.config import Settings
from lexrag.models import Chunk, ScoredChunk
from lexrag.stores.base import VectorStore

# Fixed namespace so chunk_id -> point id is stable across runs/machines.
_NAMESPACE = uuid.UUID("b2c3d4e5-f6a7-5b8c-9d0e-1f2a3b4c5d6e")

_UPSERT_BATCH = 128


class QdrantStore(VectorStore):
    def __init__(self, settings: Settings) -> None:
        # ":memory:" gives an ephemeral store (great for tests); otherwise we
        # persist to a local directory with no server to run.
        if settings.qdrant_path == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(path=settings.qdrant_path)
        self.collection = settings.collection_name
        self.dim = settings.embedding_dim

    def close(self) -> None:
        """Release the local storage lock. Important for the embedded (path=)
        backend; without it you get a noisy '__del__ during shutdown' warning
        and can leave a stale lock for the next process."""
        self.client.close()

    def __enter__(self) -> "QdrantStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(_NAMESPACE, chunk_id))

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        points = [
            PointStruct(
                id=self._point_id(chunk.chunk_id),
                vector=vector,
                # mode="json" serializes date -> ISO string so it round-trips.
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        for start in range(0, len(points), _UPSERT_BATCH):
            self.client.upsert(
                collection_name=self.collection,
                points=points[start : start + _UPSERT_BATCH],
            )

    @staticmethod
    def _build_filter(filters: Optional[dict]) -> Optional[Filter]:
        if not filters:
            return None
        # All conditions AND-ed together (Qdrant `must`).
        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filters.items()
        ]
        return Filter(must=conditions)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: Optional[dict] = None,
    ) -> list[ScoredChunk]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            query_filter=self._build_filter(filters),
            with_payload=True,
        )
        results: list[ScoredChunk] = []
        for point in response.points:
            chunk = Chunk(**point.payload)
            results.append(ScoredChunk(chunk=chunk, score=point.score, retriever="dense"))
        return results
