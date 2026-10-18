"""End-to-end dense retrieval demo. Run:  PYTHONPATH=. python scripts/demo_retrieval.py

Builds the index (embeddings are cached, so the 2nd run is much faster) and
runs a few queries, including one with a metadata pre-filter.
"""

from __future__ import annotations

import time

from lexrag.config import get_settings
from lexrag.indexing import build_index
from lexrag.retrieval.dense import DenseRetriever

settings = get_settings()

t0 = time.perf_counter()
store, embedder, chunks = build_index("data/corpus.jsonl", settings, recreate=True)
print(f"Indexed {len(chunks)} chunks in {time.perf_counter() - t0:.2f}s\n")

retriever = DenseRetriever(store, embedder)


def show(query: str, filters: dict | None = None, top_k: int = 3) -> None:
    label = f"  (filter={filters})" if filters else ""
    print(f"Q: {query}{label}")
    t = time.perf_counter()
    hits = retriever.retrieve(query, top_k=top_k, filters=filters)
    dt = (time.perf_counter() - t) * 1000
    for h in hits:
        print(f"   {h.score:.3f} | {h.chunk.jurisdiction:<10} | {h.chunk.doc_type:<8} | {h.chunk.title}")
    print(f"   ({dt:.0f} ms)\n")


# Semantic match: "save time off" has no literal overlap with "carry over vacation".
show("Can I save unused time off for next year?")

# Keyword-ish query.
show("overtime pay for misclassified employees")

# Same query, constrained to California sources via metadata pre-filter.
show("rules about carrying over vacation", filters={"jurisdiction": "US-CA"})

# Constrain to case law only.
show("remote work approval", filters={"doc_type": "case"})

store.close()  # release the local Qdrant lock cleanly
