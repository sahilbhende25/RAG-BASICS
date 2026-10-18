# LexRAG — RAG Interview Study Sheet

A growing cheat-sheet built alongside the code. Each section = concept + why it
matters + the tradeoff + likely interview questions.

---

## 0. The 30-second system pitch

> LexRAG is a grounded legal-research assistant. A paralegal asks a question;
> we retrieve the most relevant passages from a corpus of cases, statutes, and
> memos, then have an LLM answer **using only those passages, with citations**.
> The interesting engineering is in retrieval quality (hybrid search +
> reranking), grounding (no citation, no claim), caching for cost/latency, and
> an evaluation harness so we can prove the system got better.

**Why RAG at all (vs. fine-tuning or a bare LLM)?**
- Knowledge is **fresh and updatable** — add a document, it's instantly usable;
  no retraining.
- **Grounding + citations** — answers trace to sources, essential in law.
- **Cost** — no training; you pay for retrieval + a bounded context window.
- **Access control** — you can filter what a given user is allowed to retrieve.
Fine-tuning changes *style/behavior*; RAG changes *knowledge*. They compose.

---

## 1. Architecture (end-to-end)

```
            ingest                       query time
 docs ──► load ──► chunk ──► embed ──► [ Vector + Sparse index ]
                                             │
 question ─► transform ─► retrieve (dense ⊕ BM25) ─► fuse (RRF)
                                             │
                                      rerank (cross-encoder)
                                             │
                               build prompt (top-k + citations)
                                             │
                                   LLM (grounded) ─► Answer + citations
                                             │
                        eval harness + caching + observability wrap it
```

Module map in the repo:
- `lexrag/config.py` — one typed settings object (fail-fast, no hard-coded keys).
- `lexrag/models.py` — Pydantic contracts (`Document`, `Chunk`, `ScoredChunk`, `Answer`).
- `lexrag/ingestion/` — loaders + chunking.
- `lexrag/stores/` — vector store (Hour 2).
- `lexrag/retrieval/` — dense, sparse, hybrid, rerank, query-transform (Hr 2-3).
- `lexrag/generation/` — prompts + grounded generator (Hour 4).
- `lexrag/eval/` — metrics + golden set (Hour 5).
- `lexrag/api/` — FastAPI service (Hour 6).

---

## 2. Chunking (Hour 1)

**What:** split long docs into retrievable units sized in *tokens*.

**Why it's high-leverage:** retrieval can only return chunks; the chunk is the
unit of both recall and the context you spend on the LLM.

**The tradeoff:**
| Chunk size | Recall | Precision | Context cost |
|---|---|---|---|
| Large (800+) | high | low (blurry embedding) | high |
| Small (100-) | low (context lost) | high | low |
| Sweet spot (256–512) | balanced | balanced | balanced |

**Our implementation** (`ingestion/chunking.py`): token-budgeted,
sentence-aware greedy packing with an **overlap window** so a fact crossing a
boundary still appears intact in one chunk. We count tokens with
`cl100k_base` — the actual tokenizer for `text-embedding-3-*` and GPT-4.x — so
budgets match how the model bills/truncates.

**Advanced strategies to name-drop:**
- *Recursive character splitting* — split on the most semantic separator that
  fits: `\n\n` → `\n` → sentence → word.
- *Semantic chunking* — new chunk when adjacent-sentence embedding similarity
  drops below a threshold (chunks follow meaning, not length).
- *Parent-document / small-to-big* — embed small chunks for precise retrieval,
  but feed the LLM the larger **parent** passage for context.
- *Structure-aware* — for legal/markdown/code, split on headings/sections and
  keep the heading in each chunk for context.

**Interview Qs (try answering out loud):**
1. Your answers are missing facts that are *in* the corpus. Walk me through how
   chunking could be the cause and how you'd diagnose it.
2. Why overlap? What's the cost of too much overlap?
3. Why count tokens instead of characters?
4. How would you chunk a 300-page contract differently from a tweet thread?
5. A chunk ends mid-table and the number lands in the next chunk. Fixes?

---

## 3. Embeddings (preview — deep dive Hour 2)

**What:** map text → a vector so that semantically similar text is nearby
(cosine/dot distance). We use `text-embedding-3-small` (1536-dim).

**Key talking points for later:**
- `text-embedding-3-*` supports **Matryoshka** dimension shortening — you can
  truncate 1536→512 to cut memory/latency with modest quality loss.
- Normalize vectors → cosine similarity == dot product (faster).
- Symmetric vs asymmetric search; query vs passage may need different handling.
- Local/open alternatives: `bge`, `e5`, `gte`, Cohere, Voyage. Tradeoff:
  cost/latency/privacy vs quality & ops burden.

(Expanded below in Hour 2.)

---

## 4. Embeddings, vector search & caching (Hour 2)

### 4.1 Embeddings
- Map text → vector so semantic similarity = geometric closeness.
- We use `text-embedding-3-small` (1536-dim). It's **Matryoshka**: you can
  request fewer `dimensions` (e.g. 512) to shrink the index with modest quality
  loss — a real cost/latency lever (`OpenAIEmbedder` passes `dimensions`).
- **Symmetric vs asymmetric**: some models want different prefixes for queries
  vs passages — that's why `Embedder` has separate `embed_query`/`embed_texts`.
- Local/open alternatives: `bge`, `e5`, `gte`, Cohere, Voyage. Tradeoff:
  cost/latency/privacy vs quality & ops burden.

### 4.2 Distance metric
- We use **cosine** — embeddings encode *direction* (meaning), not magnitude.
- If vectors are L2-normalized, cosine == dot product (cheaper). 

### 4.3 ANN & HNSW (why search is fast)
- Exact nearest-neighbor over millions of vectors is O(N) per query — too slow.
- **HNSW** (Hierarchical Navigable Small World) is a layered proximity graph;
  search greedily hops toward the query in ~O(log N). It's *approximate*: you
  trade a little recall for huge speed.
- Knobs: `M` (graph degree), `ef_construction` (build quality), `ef_search`
  (recall↔latency at query time). Alternatives: IVF, IVF-PQ (PQ compresses
  vectors to cut memory at some recall cost).

### 4.4 Metadata filtering (pre- vs post-filter)
- We denormalized `jurisdiction`/`doc_type` onto chunks (Hour 1), so Qdrant can
  **pre-filter**: restrict the ANN traversal to matching points in one pass.
- Post-filtering (retrieve top-k then drop non-matches) is worse: if few of the
  global top-k match, you can end up with far fewer than k results.
- See `QdrantStore._build_filter` + `search(filters=...)`.

### 4.5 Idempotent upserts
- `chunk_id` → deterministic **UUIDv5** point id. Re-indexing the same chunk
  overwrites rather than duplicates. Critical for incremental re-indexing.

### 4.6 Embedding cache (the cheapest big win)
- Embeddings are deterministic for `(model, dim, text)`, so we key a persistent
  disk cache (SQLite via `diskcache`) on `sha256(model|dim|text)`.
- Warm re-index embeds only changed chunks; duplicate chunks embed once.
- Demo proof: a cached query vector returned in ~1 ms vs ~1.3 s cold.

### 4.7 Dense retrieval's blind spot
- Great at semantics ("time off" ~ "vacation"), weak at **exact** tokens
  (case numbers, names, rare terms). Fix = hybrid search (Hour 3).

**Interview Qs:**
1. Walk me through what HNSW does and the recall/latency knobs.
2. Why cosine over Euclidean for text embeddings? When are they equivalent?
3. Pre-filtering vs post-filtering by metadata — why does post-filtering break
   top-k, and when is pre-filtering expensive?
4. Where exactly do you cache in a RAG pipeline, and what's the cache key?
5. Your corpus grows from 10K to 100M chunks. What breaks first, and what do
   you change (sharding, quantization, `ef_search`, disk vs memory)?
6. Embeddings are idempotent — how do you make *re-indexing* idempotent too?
