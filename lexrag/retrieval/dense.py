"""Dense (semantic) retrieval.

Embed the query with the *same* model used for the chunks, then do an
approximate-nearest-neighbor lookup in the vector store. Dense retrieval is
strong at **semantic** matching ("time off" ~ "vacation") but can miss exact
keywords/IDs — which is exactly why we add sparse/BM25 in Hour 3.
"""

from __future__ import annotations

from typing import Optional

from lexrag.embeddings.base import Embedder
from lexrag.models import ScoredChunk
from lexrag.stores.base import VectorStore


class DenseRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[dict] = None,
    ) -> list[ScoredChunk]:
        query_vector = self.embedder.embed_query(query)
        return self.store.search(query_vector, top_k=top_k, filters=filters)
