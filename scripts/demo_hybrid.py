"""Hour 3 demo: dense vs sparse vs hybrid, plus reranking and query transforms.

Run:  PYTHONPATH=. python scripts/demo_hybrid.py
"""

from __future__ import annotations

from lexrag.config import get_settings
from lexrag.indexing import build_index
from lexrag.models import ScoredChunk
from lexrag.retrieval.dense import DenseRetriever
from lexrag.retrieval.hybrid import HybridRetriever
from lexrag.retrieval.sparse import BM25Retriever

settings = get_settings()
store, embedder, chunks = build_index("data/corpus.jsonl", settings, recreate=True)

dense = DenseRetriever(store, embedder)
sparse = BM25Retriever(chunks)
hybrid = HybridRetriever(dense, sparse, candidate_k=10)


def line(hits: list[ScoredChunk], top: int = 3) -> None:
    for h in hits[:top]:
        print(f"      {h.score:7.3f} | {h.chunk.title}")


def compare(query: str) -> None:
    print(f"\nQ: {query}")
    print("   dense:")
    line(dense.retrieve(query, top_k=10))
    print("   sparse (BM25):")
    line(sparse.retrieve(query, top_k=10))
    print("   hybrid (RRF):")
    line(hybrid.retrieve(query, top_k=10))


# Exact-name query: sparse should shine; hybrid keeps it on top.
compare("Nguyen Pacific Logistics")
# Semantic query: dense should shine.
compare("saving unused time off until next year")

# --- Reranking (downloads a small cross-encoder on first run) ---
print("\n--- Cross-encoder reranking ---")
try:
    from lexrag.retrieval.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    q = "can California employees lose earned vacation at year end?"
    candidates = hybrid.retrieve(q, top_k=10)
    print(f"Q: {q}")
    print("   hybrid top-3 (before rerank):")
    line(candidates)
    print("   reranked top-3 (after cross-encoder):")
    line(reranker.rerank(q, candidates, top_k=3))
except Exception as exc:  # noqa: BLE001 - demo resilience if model can't download
    print(f"   (skipped rerank: {exc})")

# --- Query transforms ---
print("\n--- Query transforms ---")
try:
    from lexrag.retrieval.query_transform import QueryTransformer

    qt = QueryTransformer(settings)
    print("Multi-query expansion of 'time off rollover':")
    for v in qt.multi_query("time off rollover", n=3):
        print(f"   - {v}")
    print("\nHyDE hypothetical passage for 'overtime for misclassified staff':")
    print("  ", qt.hyde("overtime for misclassified staff")[:200], "...")
except Exception as exc:  # noqa: BLE001
    print(f"   (skipped transforms: {exc})")

store.close()
