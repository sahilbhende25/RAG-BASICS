"""Prompt construction + citation parsing for grounded generation.

Grounding is the whole game in a legal assistant: the model must answer ONLY
from retrieved context and must cite which passage supports each claim. We do
that by numbering the contexts [1], [2], ... in the prompt and instructing the
model to cite with those markers. After generation we parse the markers back
into structured Citation objects, so every answer is auditable.
"""

from __future__ import annotations

import re

from lexrag.models import Citation, ScoredChunk

# The system instruction is where most anti-hallucination leverage lives.
SYSTEM_INSTRUCTIONS = (
    "You are a legal research assistant for paralegals. Answer the question "
    "using ONLY the numbered context passages provided. Cite every claim with "
    "the bracketed number(s) of the supporting passage, e.g. [1] or [2][3]. "
    "If the context does not contain enough information to answer, reply "
    "exactly: 'I don't know based on the available documents.' Do not use "
    "outside knowledge. Do not fabricate citations, statutes, or case names."
)

_CITATION_RE = re.compile(r"\[(\d+)\]")


def build_context_block(contexts: list[ScoredChunk]) -> str:
    """Render retrieved chunks as a numbered, source-attributed context block."""
    blocks = []
    for i, scored in enumerate(contexts, start=1):
        c = scored.chunk
        header = f"[{i}] {c.title} — {c.source} ({c.jurisdiction or 'n/a'}, {c.doc_type})"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


def build_user_input(question: str, context_block: str) -> str:
    return f"Context:\n{context_block}\n\nQuestion: {question}"


def extract_citations(text: str, contexts: list[ScoredChunk]) -> list[Citation]:
    """Map [n] markers in the answer back to the chunks they reference.

    Out-of-range markers are ignored (a cheap guard against the model inventing
    a [99] that doesn't exist)."""
    numbers = sorted({int(m) for m in _CITATION_RE.findall(text)})
    citations: list[Citation] = []
    for n in numbers:
        if 1 <= n <= len(contexts):
            c = contexts[n - 1].chunk
            citations.append(
                Citation(chunk_id=c.chunk_id, doc_id=c.doc_id, title=c.title, source=c.source)
            )
    return citations
