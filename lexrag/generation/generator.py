"""Grounded answer generation.

Takes the (reranked) retrieved contexts and produces an Answer that:
  - uses only those contexts,
  - carries structured citations parsed from the model's [n] markers,
  - degrades safely to "I don't know" when retrieval returned nothing.
"""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from lexrag.config import Settings
from lexrag.generation.prompts import (
    SYSTEM_INSTRUCTIONS,
    build_context_block,
    build_user_input,
    extract_citations,
)
from lexrag.models import Answer, ScoredChunk
from lexrag.obs.cost import estimate_cost

_NO_CONTEXT = "I don't know based on the available documents."


class Generator:
    def __init__(self, settings: Settings) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.generation_model

    def generate(self, question: str, contexts: list[ScoredChunk]) -> Answer:
        # Guardrail: if retrieval found nothing, never call the LLM — there's
        # nothing to ground on, so any answer would be a hallucination.
        if not contexts:
            return Answer(question=question, answer=_NO_CONTEXT, citations=[], contexts=[])

        context_block = build_context_block(contexts)
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=build_user_input(question, context_block),
        )
        text = response.output_text.strip()
        citations = extract_citations(text, contexts)

        # Observability: capture token usage + estimated cost. The Responses API
        # reports input_tokens/output_tokens; guard with getattr for safety.
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)

        return Answer(
            question=question,
            answer=text,
            citations=citations,
            contexts=contexts,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=estimate_cost(self.model, prompt_tokens, completion_tokens),
        )

    def stream(self, question: str, contexts: list[ScoredChunk]) -> Iterator[str]:
        """Yield answer tokens as they arrive (better perceived latency in a UI)."""
        if not contexts:
            yield _NO_CONTEXT
            return
        context_block = build_context_block(contexts)
        with self.client.responses.stream(
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=build_user_input(question, context_block),
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta
