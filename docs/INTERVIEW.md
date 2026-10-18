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

(Full section lands in Hour 2 with the vector store.)
