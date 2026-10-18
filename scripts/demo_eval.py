"""Hour 5 demo: run the evaluation harness over the golden set.

Run (retrieval metrics only, free):      PYTHONPATH=. python scripts/demo_eval.py
Run with LLM-as-judge (costs API calls): PYTHONPATH=. python scripts/demo_eval.py --judge
"""

from __future__ import annotations

import sys

from lexrag.config import get_settings
from lexrag.eval.dataset import load_golden
from lexrag.eval.runner import evaluate
from lexrag.indexing import build_index
from lexrag.pipeline import RAGPipeline

judge = "--judge" in sys.argv
settings = get_settings()
store, embedder, chunks = build_index("data/corpus.jsonl", settings, recreate=True)

reranker = None
try:
    from lexrag.retrieval.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker()
except Exception as exc:  # noqa: BLE001
    print(f"(rerank disabled: {exc})")

pipe = RAGPipeline(settings, store, embedder, chunks, reranker=reranker)
golden = load_golden("data/golden.jsonl")

report = evaluate(pipe, golden, k=settings.rerank_top_k, judge=judge)

print("\n=== Per-item ===")
for row in report.per_item:
    flag = "OK " if row["refusal_correct"] else "BAD"
    extra = ""
    if "faithfulness" in row:
        extra = f" faith={row['faithfulness']} rel={row['answer_relevance']}"
    print(
        f"[{flag}] hit={row['hit']:.0f} recall={row['recall']:.2f} "
        f"mrr={row['mrr']:.2f} {row['latency_ms']:.0f}ms "
        f"${row['cost_usd']:.5f}{extra}  | {row['question'][:52]}"
    )

print("\n=== Aggregate ===")
print(f"  items            : {report.n}")
print(f"  hit@k            : {report.hit_at_k:.2f}")
print(f"  recall@k         : {report.recall_at_k:.2f}")
print(f"  precision@k      : {report.precision_at_k:.2f}")
print(f"  MRR              : {report.mrr:.2f}")
print(f"  refusal accuracy : {report.refusal_accuracy:.2f}")
if report.faithfulness is not None:
    print(f"  faithfulness     : {report.faithfulness:.2f}")
    print(f"  answer relevance : {report.answer_relevance:.2f}")
print(f"  avg latency      : {report.avg_latency_ms:.0f} ms")
print(f"  total cost       : ${report.total_cost_usd:.5f}")

store.close()
