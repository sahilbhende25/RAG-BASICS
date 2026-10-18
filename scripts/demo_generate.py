"""Hour 4 demo: grounded generation, citations, 'I don't know', semantic cache, streaming.

Run:  PYTHONPATH=. python scripts/demo_generate.py
"""

from __future__ import annotations

import sys

from lexrag.config import get_settings
from lexrag.indexing import build_index
from lexrag.pipeline import RAGPipeline

settings = get_settings()
store, embedder, chunks = build_index("data/corpus.jsonl", settings, recreate=True)

# Rerank on if the cross-encoder is available; otherwise pipeline still works.
reranker = None
try:
    from lexrag.retrieval.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker()
except Exception as exc:  # noqa: BLE001
    print(f"(rerank disabled: {exc})")

pipe = RAGPipeline(settings, store, embedder, chunks, reranker=reranker)


def ask(q: str, filters: dict | None = None) -> None:
    ans = pipe.answer(q, filters=filters)
    tag = " (CACHED)" if ans.cached else ""
    print(f"\nQ: {q}{tag}   [{ans.latency_ms:.0f} ms]")
    print(f"A: {ans.answer}")
    if ans.citations:
        print("   Citations:")
        for c in ans.citations:
            print(f"     - {c.title} ({c.source})")


# 1) Grounded answer with citations.
ask("Can California employees carry unused vacation into next year?")

# 2) Out-of-corpus question -> safe refusal, no hallucination.
ask("What is the capital gains tax rate in Japan?")

# 3) Semantic cache HIT: exact repeat of #1 -> served from cache (cosine 1.0),
#    no retrieval, no LLM call. Watch the latency drop and the (CACHED) tag.
ask("Can California employees carry unused vacation into next year?")

# 3b) Threshold tradeoff: a loose paraphrase may embed < 0.95 away and MISS the
#     cache, regenerating a fresh answer. Lower settings.semantic_cache_threshold
#     to trade a higher hit rate for more risk of serving a near-but-wrong answer.
ask("Do CA workers keep unused vacation for the following year?")

# 4) Streaming (better perceived latency in a UI).
print("\nStreaming answer to: 'What happens to unused sick days at separation?'")
from lexrag.retrieval.dense import DenseRetriever  # noqa: E402

contexts = pipe.hybrid.retrieve("unused sick days at separation", top_k=settings.rerank_top_k)
sys.stdout.write("A: ")
for token in pipe.generator.stream("What happens to unused sick days at separation?", contexts):
    sys.stdout.write(token)
    sys.stdout.flush()
print()

store.close()
