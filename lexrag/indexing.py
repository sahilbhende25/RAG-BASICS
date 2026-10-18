"""Indexing pipeline: corpus file -> chunks -> embeddings -> vector store.

One function so the API, scripts, and tests all build the index the same way.
"""

from __future__ import annotations

from lexrag.config import Settings
from lexrag.embeddings.cache import EmbeddingCache
from lexrag.embeddings.openai_embedder import OpenAIEmbedder
from lexrag.ingestion.chunking import chunk_documents
from lexrag.ingestion.loaders import load_jsonl
from lexrag.stores.qdrant_store import QdrantStore


def build_index(
    corpus_path: str,
    settings: Settings,
    recreate: bool = True,
) -> tuple[QdrantStore, OpenAIEmbedder, int]:
    """Load, chunk, embed, and upsert a corpus. Returns (store, embedder, n_chunks)."""
    docs = load_jsonl(corpus_path)
    chunks = chunk_documents(
        docs,
        max_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )

    embedder = OpenAIEmbedder(settings, EmbeddingCache(settings.cache_dir))
    vectors = embedder.embed_texts([c.text for c in chunks])

    store = QdrantStore(settings)
    store.ensure_collection(recreate=recreate)
    store.upsert(chunks, vectors)

    return store, embedder, len(chunks)
