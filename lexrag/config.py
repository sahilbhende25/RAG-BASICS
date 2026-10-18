"""Central configuration.

Why a single typed Settings object instead of reading os.environ everywhere:
- One source of truth, validated at startup (fail fast on misconfig).
- Easy to override per-environment (dev/test/prod) via env vars or .env.
- No secrets hard-coded in code (the bug we fixed in rag-basic.py).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Models ---
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536  # text-embedding-3-small native dimension
    generation_model: str = "gpt-4.1-mini"

    # --- Chunking ---
    # Token budget per chunk. Small enough to be precise, large enough to be
    # self-contained. 256-512 is the usual sweet spot for prose.
    chunk_size_tokens: int = 320
    chunk_overlap_tokens: int = 64  # overlap preserves context across boundaries

    # --- Retrieval ---
    retrieval_top_k: int = 20  # wide net for recall before reranking
    rerank_top_k: int = 5      # precise set we actually feed the LLM

    # --- Storage ---
    qdrant_path: str = ".qdrant"        # on-disk local Qdrant (no server needed)
    collection_name: str = "lex_chunks"

    # --- Caching ---
    cache_dir: str = ".cache"
    semantic_cache_threshold: float = 0.95  # cosine sim to count as a cache hit


@lru_cache
def get_settings() -> Settings:
    """Cached so we parse .env once per process."""
    return Settings()
