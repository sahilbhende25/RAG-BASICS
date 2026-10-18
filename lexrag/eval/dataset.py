"""Golden evaluation set.

A small, hand-labeled set of questions with the document ids that *should* be
retrieved. This is the single most important asset for improving a RAG system:
without it, "I changed the chunk size and it feels better" is a vibe, not a
measurement. Even 20-50 well-chosen examples catch most regressions.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldenItem(BaseModel):
    question: str
    relevant_doc_ids: list[str] = Field(default_factory=list)
    expectation: str = "answerable"  # "answerable" | "refuse"
    notes: str = ""


def load_golden(path: str | Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(GoldenItem(**json.loads(line)))
    return items
