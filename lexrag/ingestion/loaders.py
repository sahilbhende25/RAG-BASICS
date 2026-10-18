"""Load raw documents into typed Document objects.

Right now we read a JSONL corpus. In production this is where you'd add PDF
parsing (e.g. pypdf / unstructured), OCR, HTML cleaning, and deduplication.
Keeping it behind one function means the rest of the pipeline never changes
when the source format does.
"""

from __future__ import annotations

import json
from pathlib import Path

from lexrag.models import Document


def load_jsonl(path: str | Path) -> list[Document]:
    docs: list[Document] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        docs.append(Document(**json.loads(line)))
    return docs
