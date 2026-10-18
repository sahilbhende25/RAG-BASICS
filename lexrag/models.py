"""Typed data models that flow through the whole pipeline.

Using Pydantic gives us validation, serialization (to/from JSON for the API and
the vector store payloads), and self-documenting contracts between modules.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document before chunking (a case, statute, or internal memo)."""

    doc_id: str
    title: str
    text: str
    source: str  # citation / URL / file the text came from
    jurisdiction: Optional[str] = None  # e.g. "US-CA", "US-Federal"
    date: Optional[dt.date] = None
    doc_type: str = "case"  # case | statute | memo
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable unit. Metadata is denormalized onto the chunk so we can
    filter at query time (e.g. jurisdiction == 'US-CA') without a join."""

    chunk_id: str
    doc_id: str
    text: str
    ordinal: int  # position within the parent document
    token_count: int
    title: str
    source: str
    jurisdiction: Optional[str] = None
    date: Optional[dt.date] = None
    doc_type: str = "case"


class ScoredChunk(BaseModel):
    """A chunk with a relevance score and which retriever produced it."""

    chunk: Chunk
    score: float
    retriever: str = "dense"  # dense | sparse | hybrid | rerank


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source: str


class Answer(BaseModel):
    """Final grounded answer returned to the user."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[ScoredChunk] = Field(default_factory=list)
    latency_ms: float = 0.0
    cached: bool = False
