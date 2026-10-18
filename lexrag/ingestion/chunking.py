"""Chunking: turning long documents into retrievable units.

This is one of the highest-leverage decisions in a RAG system. Interviewers love
it because the tradeoff is concrete:

  - Chunks too large  -> embeddings become "averaged"/blurry, retrieval is
    imprecise, and you waste the LLM context window on irrelevant text.
  - Chunks too small  -> you lose surrounding context and a single answer gets
    split across many chunks, hurting recall.

Strategy implemented here: token-budgeted, sentence-aware greedy packing with
overlap. We never split mid-sentence, we respect a token budget (not a char
count, because the model thinks in tokens), and we carry an overlap window so a
fact that straddles a boundary still appears intact in one chunk.

More advanced strategies to mention in an interview:
  - Recursive character splitting (LangChain-style): split on the most
    semantic separator that fits (\n\n -> \n -> sentence -> word).
  - Semantic chunking: start a new chunk when embedding similarity between
    adjacent sentences drops below a threshold.
  - Parent-document / small-to-big: embed small chunks for precise retrieval
    but return the larger parent passage to the LLM for context.
"""

from __future__ import annotations

import re

import tiktoken

from lexrag.models import Chunk, Document

# cl100k_base is the tokenizer used by text-embedding-3-* and gpt-4.x models,
# so counting with it matches how the models actually bill and truncate.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Split on sentence terminators followed by whitespace. Good enough for prose;
# for legal text with lots of "v." and "§" you'd refine this.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_RE.split(text.strip())]
    return [s for s in parts if s]


def _overlap_tail(sentences: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Return the trailing sentences whose combined tokens ~= overlap_tokens.

    This becomes the beginning of the next chunk so context carries over.
    """
    tail: list[str] = []
    total = 0
    for sent in reversed(sentences):
        t = count_tokens(sent)
        if total + t > overlap_tokens and tail:
            break
        tail.insert(0, sent)
        total += t
    return tail, total


def chunk_document(
    doc: Document,
    max_tokens: int = 320,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Split one document into overlapping, sentence-aligned chunks."""
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    sentences = _split_sentences(doc.text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0
    ordinal = 0

    def flush() -> None:
        nonlocal current, current_tokens, ordinal
        if not current:
            return
        text = " ".join(current).strip()
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{ordinal}",
                doc_id=doc.doc_id,
                text=text,
                ordinal=ordinal,
                token_count=count_tokens(text),
                title=doc.title,
                source=doc.source,
                jurisdiction=doc.jurisdiction,
                date=doc.date,
                doc_type=doc.doc_type,
            )
        )
        ordinal += 1

    for sent in sentences:
        sent_tokens = count_tokens(sent)

        # A single sentence longer than the budget: hard-split it by tokens so
        # we never produce an over-budget chunk.
        if sent_tokens > max_tokens:
            flush()
            current, current_tokens = [], 0
            token_ids = _ENCODING.encode(sent)
            for i in range(0, len(token_ids), max_tokens):
                piece = _ENCODING.decode(token_ids[i : i + max_tokens])
                current = [piece]
                current_tokens = count_tokens(piece)
                flush()
            current, current_tokens = [], 0
            continue

        # Adding this sentence would overflow -> emit the chunk, then start the
        # next one seeded with the overlap tail.
        if current_tokens + sent_tokens > max_tokens and current:
            flush()
            current, current_tokens = _overlap_tail(current, overlap_tokens)

        current.append(sent)
        current_tokens += sent_tokens

    flush()
    return chunks


def chunk_documents(
    docs: list[Document],
    max_tokens: int = 320,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, max_tokens, overlap_tokens))
    return out
