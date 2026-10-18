"""API tests using a stub pipeline — no network, no model downloads.

We skip the heavy startup build (LEXRAG_SKIP_STARTUP) and override the
get_pipeline dependency with a canned stub. This tests the HTTP layer (routing,
validation, serialization) in isolation from retrieval/generation.
"""

from __future__ import annotations

import os

os.environ["LEXRAG_SKIP_STARTUP"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from lexrag.api.app import app, get_pipeline  # noqa: E402
from lexrag.models import Answer, Citation  # noqa: E402


class _StubPipeline:
    def answer(self, question, filters=None, top_k=None):
        return Answer(
            question=question,
            answer="Stub answer [1].",
            citations=[Citation(chunk_id="a::0", doc_id="a", title="T", source="s")],
            latency_ms=1.0,
            cost_usd=0.0001,
        )


app.dependency_overrides[get_pipeline] = lambda: _StubPipeline()
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask_returns_answer_with_citations():
    r = client.post("/ask", json={"question": "Can I carry over vacation?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Stub answer [1]."
    assert body["citations"][0]["doc_id"] == "a"


def test_ask_validates_empty_question():
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422  # Pydantic min_length rejects it


def test_ask_accepts_filters():
    r = client.post(
        "/ask", json={"question": "vacation", "filters": {"jurisdiction": "US-CA"}}
    )
    assert r.status_code == 200
