"""Vector store tests using an in-memory Qdrant (no network, no OpenAI).

We feed hand-crafted vectors so the expected nearest-neighbor ordering is
deterministic, then assert ranking and metadata filtering behave correctly.
"""

from __future__ import annotations

from lexrag.config import Settings
from lexrag.models import Chunk
from lexrag.stores.qdrant_store import QdrantStore


def _settings() -> Settings:
    # Explicit kwargs override any .env; tiny dim keeps vectors readable.
    return Settings(
        openai_api_key="test",
        embedding_dim=3,
        qdrant_path=":memory:",
        collection_name="test_chunks",
    )


def _chunk(cid: str, juris: str, dtype: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=cid.split("::")[0],
        text=f"text-{cid}",
        ordinal=0,
        token_count=3,
        title=f"Title {cid}",
        source="s",
        jurisdiction=juris,
        doc_type=dtype,
    )


def _store() -> QdrantStore:
    store = QdrantStore(_settings())
    store.ensure_collection(recreate=True)
    chunks = [
        _chunk("a::0", "US-CA", "statute"),
        _chunk("b::0", "US-NY", "case"),
        _chunk("c::0", "US-CA", "case"),
    ]
    vectors = [
        [1.0, 0.0, 0.0],  # a -> aligned with x
        [0.0, 1.0, 0.0],  # b -> aligned with y
        [0.9, 0.1, 0.0],  # c -> close to x
    ]
    store.upsert(chunks, vectors)
    return store


def test_ranking_by_cosine_similarity():
    store = _store()
    hits = store.search([1.0, 0.0, 0.0], top_k=3)
    # x-aligned vectors (a, then c) should outrank the y-aligned b.
    assert hits[0].chunk.chunk_id == "a::0"
    assert hits[1].chunk.chunk_id == "c::0"
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_metadata_prefilter_restricts_results():
    store = _store()
    hits = store.search([1.0, 0.0, 0.0], top_k=5, filters={"jurisdiction": "US-CA"})
    ids = {h.chunk.chunk_id for h in hits}
    assert ids == {"a::0", "c::0"}  # b (US-NY) excluded by the pre-filter


def test_combined_filters_are_anded():
    store = _store()
    hits = store.search(
        [1.0, 0.0, 0.0], top_k=5, filters={"jurisdiction": "US-CA", "doc_type": "case"}
    )
    assert {h.chunk.chunk_id for h in hits} == {"c::0"}


def test_upsert_is_idempotent():
    store = _store()
    # Re-upsert same ids; count must stay 3, not grow to 6.
    store.upsert(
        [_chunk("a::0", "US-CA", "statute")],
        [[1.0, 0.0, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0], top_k=10)
    assert len({h.chunk.chunk_id for h in hits}) == 3


def test_deterministic_point_id():
    assert QdrantStore._point_id("a::0") == QdrantStore._point_id("a::0")
    assert QdrantStore._point_id("a::0") != QdrantStore._point_id("a::1")
