"""Tests for prompt building + citation parsing (pure functions, no network)."""

from __future__ import annotations

from lexrag.generation.prompts import (
    build_context_block,
    build_user_input,
    extract_citations,
)
from lexrag.models import Chunk, ScoredChunk


def _ctx(cid: str, title: str, text: str) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=cid,
        doc_id=cid.split("::")[0],
        text=text,
        ordinal=0,
        token_count=len(text.split()),
        title=title,
        source=f"src-{cid}",
        jurisdiction="US-CA",
        doc_type="statute",
    )
    return ScoredChunk(chunk=chunk, score=1.0, retriever="rerank")


def test_context_block_is_numbered_from_one():
    block = build_context_block([_ctx("a::0", "Alpha", "x"), _ctx("b::0", "Beta", "y")])
    assert block.startswith("[1] Alpha")
    assert "[2] Beta" in block


def test_user_input_contains_question_and_context():
    text = build_user_input("What is X?", "[1] Alpha — src (US-CA, statute)\nbody")
    assert "Question: What is X?" in text
    assert "Context:" in text


def test_extract_citations_maps_markers_to_chunks():
    contexts = [_ctx("a::0", "Alpha", "x"), _ctx("b::0", "Beta", "y")]
    cites = extract_citations("Vacation vests as earned [1], and caps apply [2].", contexts)
    assert [c.chunk_id for c in cites] == ["a::0", "b::0"]
    assert cites[0].source == "src-a::0"


def test_extract_citations_dedupes_and_ignores_out_of_range():
    contexts = [_ctx("a::0", "Alpha", "x")]
    cites = extract_citations("See [1], again [1], and bogus [99].", contexts)
    assert [c.chunk_id for c in cites] == ["a::0"]  # deduped, [99] ignored


def test_extract_citations_empty_when_none():
    contexts = [_ctx("a::0", "Alpha", "x")]
    assert extract_citations("I don't know based on the available documents.", contexts) == []
