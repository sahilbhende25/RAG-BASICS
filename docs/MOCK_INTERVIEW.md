# LexRAG — Mock Interview (SDE-2, RAG)

Use this to rehearse out loud. Each section: the question, a strong answer
skeleton, and the follow-up they'll push on. Pair with `INTERVIEW.md` for depth.

---

## A. 2-minute project pitch (have this memorized)

> I built **LexRAG**, a grounded legal-research assistant. A paralegal asks a
> question; it retrieves the most relevant passages from a corpus of cases,
> statutes, and memos and answers **using only those passages, with citations**.
> The engineering depth is in retrieval quality — **hybrid search (dense + BM25)
> fused with Reciprocal Rank Fusion, then a cross-encoder reranker** — plus
> **grounding** (no citation, no claim; refuse when nothing is retrieved),
> **caching** (content-addressed embeddings + a semantic answer cache), and an
> **evaluation harness** with a golden set so every change is measured. It's
> served behind a FastAPI app with per-query latency and cost telemetry.

Follow-up: *"Why RAG over fine-tuning?"* → fresh/updatable knowledge, grounding
+ citations, no training cost, access control at retrieval. Fine-tuning changes
behavior/style; RAG changes knowledge. They compose.

---

## B. System-design whiteboard prompt

**"Design a RAG system for 10M legal documents, 500 QPS, strict citation
requirements."** Structure your answer:

1. **Ingestion** (offline, async): parse (PDF/OCR) → clean/dedupe →
   structure-aware chunk → embed (batched, cached) → upsert to vector + sparse
   indexes. Idempotent by `chunk_id`.
2. **Retrieval** (online): metadata **pre-filter** (jurisdiction/date/ACL) →
   dense (HNSW) ⊕ BM25 → **RRF** → cross-encoder **rerank** → top-k.
3. **Generation**: grounded prompt with numbered contexts → LLM → **parse
   citations** → refuse if unsupported.
4. **Cross-cutting**: semantic cache, observability (latency/cost/traces),
   eval harness as CI gate, access control, PII/retention.
5. **Scale**: sharded/replicated vector store + quantization; Elasticsearch for
   sparse; stateless API replicas; rerank on GPU or hosted.

Name **tradeoffs** at each step (recall vs precision vs latency vs cost).

---

## C. Rapid-fire (answer in 1-2 sentences)

1. Chunking: what breaks if chunks are too big? too small?
2. Why token budgets, not character counts?
3. Cosine vs dot product — when identical?
4. What does HNSW do; which knob trades recall for latency?
5. Pre-filter vs post-filter metadata — why does post-filter break top-k?
6. Why can't you add dense + BM25 scores? What does RRF do instead?
7. Bi-encoder vs cross-encoder — and why use both?
8. What is HyDE; when does it hurt?
9. How do you stop hallucination — name the layers.
10. recall@k vs precision@k — which for retrieve-wide, and why can precision@k
    look awful when retrieval is fine?
11. What is faithfulness; how measured without humans?
12. Where do you cache, and what are the keys (two different caches)?
13. Idempotent re-indexing — how?
14. Prompt injection via a retrieved doc — mitigations?
15. GDPR delete — what must propagate?

(Answers: all covered in `INTERVIEW.md` §2-§8.)

---

## D. "Tell me about a bug" (you have a real one)

> While wiring the Pydantic models, a field named `date` with a `None` default
> shadowed the imported `date` type under `from __future__ import annotations`,
> so Pydantic resolved the annotation to `None` and rejected all dates. I read
> the validation error, traced it to the name collision, and fixed it by
> aliasing the import (`import datetime as dt`). Lesson: forward-ref annotations
> resolve names at class-build time, so field names can shadow types.

---

## E. Things to proactively mention (senior signals)

- "I measure before/after with a golden set — here are the metrics."
- "Every expensive op is cached; every write is idempotent."
- "Retrieval and generation are separate interfaces, so I can swap vendors."
- "I refuse instead of guessing when retrieval is empty."
- "Here's the latency/cost per query and where I'd cut it under an SLA."
- "Here's how access control lives in the retrieval filter, not bolted on."
