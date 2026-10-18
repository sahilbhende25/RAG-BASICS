"""Evaluation harness: run the pipeline over the golden set and report metrics.

Retrieval metrics are computed for every item (free). Generation metrics
(LLM-as-judge) are opt-in via judge=True because they cost API calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from lexrag.eval.dataset import GoldenItem
from lexrag.eval.metrics import (
    LLMJudge,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieved_doc_ids,
)
from lexrag.models import Answer
from lexrag.pipeline import RAGPipeline

_REFUSAL = "i don't know"


@dataclass
class EvalReport:
    n: int = 0
    hit_at_k: float = 0.0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    refusal_accuracy: float = 0.0  # correct refuse vs answer decisions
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    avg_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    per_item: list[dict] = field(default_factory=list)


def evaluate(
    pipeline: RAGPipeline,
    golden: list[GoldenItem],
    k: int = 5,
    judge: bool = False,
) -> EvalReport:
    llm_judge = LLMJudge(pipeline.settings) if judge else None
    report = EvalReport(n=len(golden))

    agg = {"hit": 0.0, "recall": 0.0, "precision": 0.0, "mrr": 0.0, "refusal": 0.0}
    faith_scores: list[float] = []
    rel_scores: list[float] = []

    for item in golden:
        # Disable the semantic cache so every eval item exercises the full path.
        ans: Answer = pipeline.answer(item.question, top_k=k, use_cache=False)
        got_ids = retrieved_doc_ids(ans.contexts, k)

        hit = hit_at_k(got_ids, item.relevant_doc_ids)
        rec = recall_at_k(got_ids, item.relevant_doc_ids)
        prec = precision_at_k(got_ids, item.relevant_doc_ids, k)
        rr = reciprocal_rank(got_ids, item.relevant_doc_ids)

        refused = _REFUSAL in ans.answer.lower()
        should_refuse = item.expectation == "refuse"
        refusal_correct = refused == should_refuse

        agg["hit"] += hit
        agg["recall"] += rec
        agg["precision"] += prec
        agg["mrr"] += rr
        agg["refusal"] += 1.0 if refusal_correct else 0.0
        report.avg_latency_ms += ans.latency_ms
        report.total_cost_usd += ans.cost_usd

        row = {
            "question": item.question,
            "expectation": item.expectation,
            "retrieved": got_ids,
            "relevant": item.relevant_doc_ids,
            "hit": hit,
            "recall": rec,
            "mrr": rr,
            "refused": refused,
            "refusal_correct": refusal_correct,
            "latency_ms": round(ans.latency_ms, 1),
            "cost_usd": round(ans.cost_usd, 6),
        }

        # Only judge answerable items that actually produced an answer.
        if llm_judge is not None and should_refuse is False:
            f = llm_judge.faithfulness(ans)
            r = llm_judge.answer_relevance(ans)
            faith_scores.append(f)
            rel_scores.append(r)
            row["faithfulness"] = round(f, 2)
            row["answer_relevance"] = round(r, 2)

        report.per_item.append(row)

    n = max(len(golden), 1)
    report.hit_at_k = agg["hit"] / n
    report.recall_at_k = agg["recall"] / n
    report.precision_at_k = agg["precision"] / n
    report.mrr = agg["mrr"] / n
    report.refusal_accuracy = agg["refusal"] / n
    report.avg_latency_ms /= n
    if faith_scores:
        report.faithfulness = sum(faith_scores) / len(faith_scores)
    if rel_scores:
        report.answer_relevance = sum(rel_scores) / len(rel_scores)
    return report
