"""Sparse (lexical) retrieval with BM25.

Dense retrieval matches *meaning* but can whiff on *exact* tokens — case names,
docket numbers, statute sections, rare acronyms. BM25 is the opposite: it scores
by term frequency / inverse document frequency, so it nails exact keywords but
has no notion of synonyms. Running both and fusing them (Hour 3's hybrid) gives
you recall from semantics AND precision on exact terms.

Note the asymmetry vs the vector store: BM25 here is an in-memory index, so
metadata filtering is a *post-filter* (we drop non-matching chunks after
scoring). Qdrant could pre-filter; a pure BM25 engine (Elasticsearch) would
filter in-engine. Worth calling out in an interview.
"""

from __future__ import annotations

import re
from typing import Optional

from rank_bm25 import BM25Okapi

from lexrag.models import Chunk, ScoredChunk

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    # Lowercase word tokens. A production system would add stemming and
    # stopword handling; kept simple and transparent here.
    return _TOKEN_RE.findall(text.lower())


def _matches(chunk: Chunk, filters: Optional[dict]) -> bool:
    if not filters:
        return True
    return all(getattr(chunk, key, None) == value for key, value in filters.items())


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        # BM25 needs the whole tokenized corpus up front to compute IDF.
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[dict] = None,
    ) -> list[ScoredChunk]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[ScoredChunk] = []
        for i in order:
            chunk = self.chunks[i]
            if not _matches(chunk, filters):
                continue
            results.append(
                ScoredChunk(chunk=chunk, score=float(scores[i]), retriever="sparse")
            )
            if len(results) >= top_k:
                break
        return results
