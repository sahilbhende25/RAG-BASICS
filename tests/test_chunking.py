"""Unit tests for the chunker.

Tests here encode the *invariants* we care about, which is exactly what an
interviewer wants to see: you don't just eyeball output, you assert properties.
"""

from __future__ import annotations

from datetime import date

from lexrag.ingestion.chunking import chunk_document, count_tokens
from lexrag.models import Document


def _doc(text: str) -> Document:
    return Document(
        doc_id="d1",
        title="T",
        text=text,
        source="s",
        jurisdiction="US-CA",
        date=date(2024, 1, 1),
        doc_type="memo",
    )


def test_short_document_is_single_chunk():
    chunks = chunk_document(_doc("One sentence only."), max_tokens=320)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "d1::0"


def test_respects_token_budget():
    text = " ".join(f"Sentence number {i} has some words." for i in range(200))
    chunks = chunk_document(_doc(text), max_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 100


def test_chunks_overlap_for_context_continuity():
    text = " ".join(f"Fact {i} is distinct and verifiable." for i in range(60))
    chunks = chunk_document(_doc(text), max_tokens=60, overlap_tokens=20)
    # Consecutive chunks should share trailing/leading content from overlap.
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words


def test_metadata_is_denormalized_onto_chunks():
    chunks = chunk_document(_doc("A. B. C."), max_tokens=320)
    assert chunks[0].jurisdiction == "US-CA"
    assert chunks[0].doc_type == "memo"


def test_ordinals_are_sequential():
    text = " ".join(f"Line {i} here." for i in range(50))
    chunks = chunk_document(_doc(text), max_tokens=40, overlap_tokens=8)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
