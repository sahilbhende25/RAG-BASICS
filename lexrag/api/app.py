"""FastAPI service exposing the RAG pipeline.

Design choices that read as production-minded:
  - **Index built once at startup** (lifespan), not per request — building per
    request would re-embed the whole corpus every call.
  - **Pipeline injected via a dependency** (`get_pipeline`) so tests can swap in
    a stub with `app.dependency_overrides` — no network needed in unit tests.
  - **Typed request/response** via Pydantic for automatic validation + OpenAPI
    docs at /docs.
  - **/health** for load-balancer/k8s liveness probes.
  - A startup kill-switch (LEXRAG_SKIP_STARTUP=1) so tests don't do the heavy
    (network) index build.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from lexrag.config import get_settings
from lexrag.indexing import build_index
from lexrag.models import Answer
from lexrag.pipeline import RAGPipeline


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    filters: Optional[dict] = None  # e.g. {"jurisdiction": "US-CA"}
    top_k: Optional[int] = None


def build_pipeline() -> RAGPipeline:
    settings = get_settings()
    store, embedder, chunks = build_index("data/corpus.jsonl", settings, recreate=True)
    reranker = None
    try:
        from lexrag.retrieval.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker()
    except Exception:  # noqa: BLE001 - rerank is optional; service still works
        reranker = None
    return RAGPipeline(settings, store, embedder, chunks, reranker=reranker)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Heavy, network-touching build happens once — unless a test opts out.
    if os.getenv("LEXRAG_SKIP_STARTUP") != "1":
        app.state.pipeline = build_pipeline()
    yield
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is not None:
        try:
            pipeline.dense.store.close()
        except Exception:  # noqa: BLE001
            pass


app = FastAPI(title="LexRAG API", version="1.0.0", lifespan=lifespan)


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> Answer:
    return pipeline.answer(req.question, filters=req.filters, top_k=req.top_k)
