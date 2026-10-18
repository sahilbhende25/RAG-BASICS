"""Evaluation metrics: retrieval (deterministic) + generation (LLM-as-judge).

Retrieval metrics work on **document ids** and need no API calls, so they're
cheap enough to run on every change. Generation metrics judge the free-text
answer and require an LLM grader.

RAG quality splits cleanly into two questions, and these metrics map to them:
  - Did we fetch the right context?      -> hit@k, recall@k, precision@k, MRR
  - Did we answer faithfully & usefully? -> faithfulness, answer relevance
"""

from __future__ import annotations

import re

from openai import OpenAI

from lexrag.config import Settings
from lexrag.models import Answer, ScoredChunk

# --------------------------------------------------------------------------- #
# Retrieval metrics (pure functions, no network)
# --------------------------------------------------------------------------- #


def retrieved_doc_ids(contexts: list[ScoredChunk], k: int) -> list[str]:
    """Unique parent doc_ids from the top-k contexts, preserving rank order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for scored in contexts[:k]:
        did = scored.chunk.doc_id
        if did not in seen:
            seen.add(did)
            ordered.append(did)
    return ordered


def hit_at_k(retrieved: list[str], relevant: list[str]) -> float:
    """1.0 if any relevant doc appears in the retrieved list, else 0.0."""
    return 1.0 if set(retrieved) & set(relevant) else 0.0


def recall_at_k(retrieved: list[str], relevant: list[str]) -> float:
    if not relevant:
        return 1.0  # nothing to find -> vacuously perfect
    return len(set(retrieved) & set(relevant)) / len(set(relevant))


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / k


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """1/rank of the first relevant doc (MRR for a single query)."""
    rel = set(relevant)
    for idx, did in enumerate(retrieved, start=1):
        if did in rel:
            return 1.0 / idx
    return 0.0


# --------------------------------------------------------------------------- #
# Generation metrics (LLM-as-judge)
# --------------------------------------------------------------------------- #

_SCORE_RE = re.compile(r"[01](?:\.\d+)?")


class LLMJudge:
    """Grades answers with an LLM. This is how RAGAS-style faithfulness works:
    a stronger model checks whether the answer is supported by the context."""

    def __init__(self, settings: Settings) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.generation_model

    def _score(self, instructions: str, payload: str) -> float:
        resp = self.client.responses.create(
            model=self.model, instructions=instructions, input=payload
        )
        match = _SCORE_RE.search(resp.output_text.strip())
        return max(0.0, min(1.0, float(match.group()))) if match else 0.0

    def faithfulness(self, answer: Answer) -> float:
        """Is every claim in the answer supported by the retrieved context?
        1.0 = fully grounded, 0.0 = contains unsupported claims."""
        context = "\n\n".join(f"- {c.chunk.text}" for c in answer.contexts)
        return self._score(
            "You are a strict grader. Output ONLY a number in [0,1]: the "
            "fraction of the answer that is directly supported by the context. "
            "1 means fully grounded, 0 means unsupported/hallucinated.",
            f"Context:\n{context}\n\nAnswer:\n{answer.answer}",
        )

    def answer_relevance(self, answer: Answer) -> float:
        """Does the answer actually address the question? 1.0 = fully."""
        return self._score(
            "Output ONLY a number in [0,1] rating how well the answer addresses "
            "the question (1 = fully answers, 0 = irrelevant).",
            f"Question: {answer.question}\n\nAnswer:\n{answer.answer}",
        )
