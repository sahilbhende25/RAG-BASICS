"""Quick visual check of ingestion + chunking. Run:  python scripts/demo_chunking.py"""

from __future__ import annotations

from lexrag.config import get_settings
from lexrag.ingestion.chunking import chunk_documents
from lexrag.ingestion.loaders import load_jsonl

settings = get_settings()
docs = load_jsonl("data/corpus.jsonl")
chunks = chunk_documents(
    docs,
    max_tokens=settings.chunk_size_tokens,
    overlap_tokens=settings.chunk_overlap_tokens,
)

print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks\n")
for c in chunks:
    print(f"[{c.chunk_id}] {c.token_count:>3} tok | {c.jurisdiction:<10} | {c.doc_type:<8} | {c.title}")
    print(f"    {c.text[:110]}...")
