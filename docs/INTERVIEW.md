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

---

## 5. Advanced retrieval (Hour 3) — the senior-signal hour

### 5.1 Sparse / lexical (BM25)
- Scores by term frequency × inverse document frequency; great at **exact**
  tokens (case names, docket numbers, statute sections), blind to synonyms.
- Complements dense retrieval's exact-match blind spot.
- Demo: BM25 scored "Nguyen Pacific Logistics" at 1.25 for the right case and
  **0.00** for everything else — pure keyword precision.
- Note: our BM25 is in-memory, so filtering is a *post-filter* (vs Qdrant's
  pre-filter, vs Elasticsearch filtering in-engine).

### 5.2 Hybrid via Reciprocal Rank Fusion (RRF)
- Dense (cosine ~0-1) and BM25 (unbounded) scores are **non-comparable**, so
  you can't add them. RRF fuses on **rank, not score**:
  `score(d) = Σ_retrievers 1/(k + rank_r(d))`, k=60 canonical.
- Scale-free (no fragile normalization), rewards items ranked well by multiple
  retrievers, and k dampens top-rank dominance.
- Pull a **wide** candidate set per retriever before fusing (an item can be #15
  dense but #2 sparse).

### 5.3 Cross-encoder reranking (retrieve wide, rerank narrow)
- **Bi-encoder** (our dense retriever): encodes query and chunk *separately* →
  fast, precomputable, but coarse (they never attend to each other).
- **Cross-encoder** (reranker): feeds `(query, chunk)` *together* so tokens
  cross-attend → much more accurate, but must run per candidate at query time
  (can't score the whole corpus).
- Pattern: cheap wide retrieval (hybrid, ~top 20-50) → cross-encoder rerank →
  keep top 3-5 for the LLM.
- Demo proof: bi-encoder scores were ~0.032 for all three (indistinguishable);
  the cross-encoder split relevant (+7.0) from irrelevant (−4.2).

### 5.4 Query transformation
- **Multi-query**: LLM paraphrases the question; retrieve for each; fuse. Fixes
  vocabulary mismatch ("time off" → "vacation/PTO carryover"). Demo did exactly
  this rewrite.
- **HyDE**: LLM writes a *hypothetical answer*, embed THAT (a fake answer sits
  near real answer passages). Win when queries are short/vague; risk if the LLM
  hallucinates off-topic.
- Others: query **decomposition** (split multi-part questions), **step-back**
  (ask a more general question first).

### 5.5 The full query-time flow now
```
query ─► (optional) transform ─► dense ⊕ BM25 ─► RRF fuse ─► cross-encoder rerank ─► top-k
```

**Latency/cost tradeoffs to say out loud:** every stage adds latency —
multi-query multiplies retrieval calls + an LLM call; HyDE adds an LLM call on
the critical path; reranking adds a model inference over N candidates. You tune
candidate_k and which stages are on vs off per latency budget. Cheapest big
wins first: hybrid + rerank usually beats fancy query transforms.

**Interview Qs:**
1. Dense misses an exact docket number that's in the corpus. Why, and which
   retriever fixes it?
2. Why can't you just add dense and BM25 scores? How does RRF avoid that?
3. Bi-encoder vs cross-encoder: the core difference and why you use *both*.
4. Walk me through "retrieve wide, rerank narrow" and how you pick candidate_k.
5. What is HyDE and when does it *hurt*?
6. Your p95 latency is too high. Which retrieval stages do you cut first and why?

---

## 6. Generation & grounding (Hour 4)

### 6.1 Grounding = answer only from retrieved context
- The whole value of RAG in law is **traceability**: every claim must map to a
  source. We number contexts `[1][2]...` in the prompt and instruct the model
  to cite with those markers (`generation/prompts.py`).
- **Hard guardrail**: if retrieval returns nothing, we **don't call the LLM** —
  there's nothing to ground on, so any output would be a hallucination. We
  return "I don't know based on the available documents." Demo proved this on
  "capital gains tax in Japan" (out of corpus).

### 6.2 Citations are parsed, not trusted
- After generation we regex the `[n]` markers and map them back to the actual
  chunks (`extract_citations`). Out-of-range markers (model invents `[99]`) are
  dropped. Result: a structured, auditable `Answer.citations`.

### 6.3 Anti-hallucination layers (name all of them)
1. Retrieval quality (hybrid + rerank) — right context in the window.
2. Strict system instructions — "only use context, don't fabricate."
3. No-context short-circuit — refuse instead of guessing.
4. Citation parsing — surfaces *which* passage supports each claim; an answer
   with zero valid citations is a red flag you can block or flag for review.
5. (Extension) a verifier pass / NLI check that each sentence is entailed by a
   cited chunk — this is what RAGAS "faithfulness" measures (Hour 5).

### 6.4 Streaming
- `Generator.stream()` yields tokens as they arrive. Doesn't reduce total
  latency, but massively improves **perceived** latency / TTFT in a UI.

### 6.5 Semantic cache
- Exact-string caches miss paraphrases. A **semantic** cache keys on the query
  **embedding** and returns a cached answer when cosine ≥ threshold.
- Demo: exact repeat hit in **~1 ms** vs ~1.4 s cold; a loose paraphrase at
  threshold 0.95 **missed** and regenerated — the precision/recall knob is the
  threshold. Too low → stale/wrong answers for merely-related questions.
- Production concerns: **TTL/invalidation** when the corpus changes, and
  **per-user scoping** so cached answers don't leak across access permissions.

### 6.6 The full pipeline (`pipeline.py`)
`cache check → hybrid retrieve → rerank → grounded generate → cache store`,
with latency measured end-to-end and the query embedding reused for both the
cache lookup and dense retrieval.

**Interview Qs:**
1. How do you stop a RAG system from hallucinating? Name the layers.
2. Retrieval returns nothing — what should generation do, and why?
3. How do you make answers auditable / attributable?
4. Exact-match vs semantic cache — tradeoffs and failure modes.
5. How do you invalidate a semantic cache when a document is updated?
6. Does streaming reduce latency? What does it actually improve?

---

## 7. Evaluation & observability (Hour 5)

> "I changed the chunk size and it feels better" is a vibe. A golden set turns
> it into a number. This hour is what separates mid from senior.

### 7.1 Split quality into two questions
- **Did we retrieve the right context?** → retrieval metrics (deterministic).
- **Did we answer faithfully & usefully?** → generation metrics (LLM judge).
If retrieval is bad, no amount of prompt tuning saves you — so measure both.

### 7.2 Golden set (`data/golden.jsonl`)
- Hand-labeled `question → relevant_doc_ids (+ expectation: answerable/refuse)`.
- Even 20-50 examples catch most regressions. Includes an **out-of-corpus**
  question to test that the system *refuses* instead of hallucinating.

### 7.3 Retrieval metrics (`eval/metrics.py`, no API cost)
- **hit@k**: did any relevant doc appear in top-k? (coarse)
- **recall@k**: fraction of relevant docs retrieved. *The* metric for "retrieve
  wide" — if recall@k is low, reranking can't help (the doc isn't in the set).
- **precision@k**: fraction of top-k that's relevant. **Low by construction
  when k ≫ #relevant** — our run showed 0.20 at k=5 with ~1 relevant doc each.
  Don't panic at low precision@k; read it alongside recall.
- **MRR**: 1/rank of the first relevant doc — rewards putting the right doc first
  (what reranking optimizes).

### 7.4 Generation metrics — LLM-as-judge (this is how RAGAS works)
- **Faithfulness**: is every claim supported by the retrieved context?
  (catches hallucination). Our run: 1.00.
- **Answer relevance**: does the answer address the question? Our run: 0.96.
- **Refusal accuracy**: did it refuse exactly when it should? Our run: 1.00.
- Caveat to state: the judge is itself an LLM (noisy, can be biased toward
  verbose answers). Use a strong grader, low temperature, and spot-check.

### 7.5 Observability (`obs/cost.py`, `Answer` fields)
- Per-query **token usage + $ cost** captured from the Responses API usage and a
  central price table. Plus **latency** end-to-end. Our run: avg 1757 ms,
  $0.0028 total for 6 questions + judging.
- Why it matters: you can't optimize or budget what you don't measure; this is
  how you justify caching or a cheaper model, and how you alarm on a prompt that
  ballooned.

### 7.6 The harness (`eval/runner.py`)
Runs the pipeline over the golden set (cache off so every item exercises the
full path), aggregates all metrics into one `EvalReport`. This is your
regression gate: run it before/after any change.

**Interview Qs:**
1. How do you know your RAG system is *good*? What do you measure, and how?
2. recall@k vs precision@k — which matters more for the retrieve-wide stage, and
   why can precision@k look terrible even when retrieval is fine?
3. What is "faithfulness" and how would you measure it without humans?
4. Risks of LLM-as-judge, and how do you mitigate them?
5. Retrieval recall is high but users complain answers are wrong — where do you
   look?
6. How do you track and control cost + latency in production?

---

## 8. Productionization (Hour 6)

### 8.1 The service (`lexrag/api/app.py`)
- **FastAPI** with typed request/response (auto validation + OpenAPI at `/docs`).
- **Index built once at startup** via `lifespan` — not per request (that would
  re-embed the whole corpus every call).
- **Pipeline injected** via a `get_pipeline` dependency so tests swap a stub
  with `dependency_overrides` — HTTP layer tested with zero network.
- **`/health`** for k8s/load-balancer probes; graceful store `close()` on
  shutdown. Verified live: grounded answer + citations + token/cost telemetry;
  out-of-corpus → refuse; empty question → 422.

### 8.2 Scaling from 10K → 100M chunks (say this as a story)
1. **Vector store**: embedded local → managed/distributed (Qdrant cluster,
   pgvector, or a managed service) with **sharding + replication**. Tune HNSW
   `ef_search` for recall/latency; add **quantization** (PQ / int8) to fit
   memory.
2. **Sparse**: in-memory BM25 → **Elasticsearch/OpenSearch** so lexical search
   scales and filters in-engine.
3. **Embeddings**: batch + the content-addressed cache; move ingestion to an
   **async pipeline** (queue + workers) so indexing doesn't block serving.
4. **Rerank**: the cross-encoder is the latency hot-spot → cap `candidate_k`,
   batch on GPU, or use a hosted rerank API; make it toggleable per latency SLA.
5. **Serving**: stateless API replicas behind a load balancer; the index/state
   lives in the stores, not the process.

### 8.3 Failure modes & mitigations
- **LLM/API down or slow** → timeouts, retries w/ backoff, circuit breaker,
  graceful "try later"; stream for perceived latency.
- **Nothing retrieved** → refuse, don't hallucinate (already enforced).
- **Stale index** after a doc changes → idempotent re-index by `chunk_id`;
  invalidate semantic cache (TTL or on-write).
- **Prompt injection** in retrieved docs ("ignore instructions…") → treat
  context as data, instruction/content separation, output validation.
- **Cost blowups** → cap `top_k`/context size, cache, alert on cost per query.

### 8.4 Security / multi-tenant
- **Access control at retrieval**: filter by the caller's permitted doc set
  (metadata pre-filter) so users can't retrieve what they can't see.
- **Per-user cache scoping** so cached answers don't leak across tenants.
- **PII / data retention** (our GDPR doc is a nod to this): deletion propagates
  to the index + cache.

### 8.5 Deployment
- **Dockerfile**: deps layer cached separately from app layer; `OPENAI_API_KEY`
  injected at runtime (never baked in). `uvicorn` entrypoint.
- Next steps: CI (run `pytest` + the eval harness as a **quality gate**),
  health/readiness probes, autoscaling on QPS/latency.

**Interview Qs:**
1. Walk me through scaling this from 10K to 100M documents.
2. The LLM API starts timing out intermittently — how does your service behave?
3. A retrieved document contains "ignore previous instructions" — what happens?
4. How do you ensure a user only gets answers from documents they're allowed to
   see?
5. A document is deleted for GDPR — what has to happen across your system?
6. How would you deploy this and gate quality in CI?
