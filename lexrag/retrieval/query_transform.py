"""Query transformation — improve retrieval by rewriting the query first.

Two classic techniques, both implemented here:

1) Multi-query expansion: ask an LLM for a few paraphrases of the question,
   retrieve for each, and fuse. Robust to vocabulary mismatch — if the user
   says "time off" but the corpus says "vacation/PTO", one paraphrase bridges it.

2) HyDE (Hypothetical Document Embeddings): ask the LLM to *write a plausible
   answer* to the question, then embed THAT and retrieve. A hypothetical answer
   lives in the same vector neighborhood as real answer passages, which often
   beats embedding the bare question. Risk: if the LLM hallucinates, you can
   retrieve off-target — so it's a tradeoff, not a free win.

Other techniques to mention: query decomposition (split a multi-part question
into sub-questions), and step-back prompting (ask a more general question first).
"""

from __future__ import annotations

from openai import OpenAI

from lexrag.config import Settings


class QueryTransformer:
    def __init__(self, settings: Settings) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.generation_model

    def multi_query(self, query: str, n: int = 3) -> list[str]:
        """Return the original query plus n paraphrases (deduped)."""
        resp = self.client.responses.create(
            model=self.model,
            instructions=(
                "You rewrite a user's search query into alternative phrasings "
                "that would retrieve the same information. Return ONLY the "
                f"rewrites, one per line, no numbering. Produce exactly {n}."
            ),
            input=query,
        )
        variants = [line.strip("-• ").strip() for line in resp.output_text.splitlines()]
        variants = [v for v in variants if v]
        # Keep the original first; dedupe case-insensitively, preserve order.
        seen: set[str] = set()
        out: list[str] = []
        for q in [query, *variants]:
            key = q.lower()
            if key not in seen:
                seen.add(key)
                out.append(q)
        return out

    def hyde(self, query: str) -> str:
        """Generate a short hypothetical answer passage to embed instead of the query."""
        resp = self.client.responses.create(
            model=self.model,
            instructions=(
                "Write a brief, factual-sounding paragraph that would directly "
                "answer the user's question, as if excerpted from a reference "
                "document. Do not add caveats or say you are unsure."
            ),
            input=query,
        )
        return resp.output_text.strip()
