# FinAnalyst AI

> Agentic RAG assistant for financial analysts — answers complex questions about SEC filings in seconds instead of hours of manual reading.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)
![Tests](https://img.shields.io/badge/tests-141%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-77%25-green)
![Ruff](https://img.shields.io/badge/code%20style-ruff-black)

---

## Problem

Financial analysts spend 3–5 hours per day reading 10-K and 10-Q filings. Generic chatbots fail because they:

- Don't understand domain terminology (EBITDA, FCF, non-GAAP, ЧОД)
- Return answers that aren't in the source documents (hallucinations)
- Can't compare metrics across companies or time periods
- Have no concept of document structure (Income Statement vs. MD&A vs. Risk Factors)

## Solution

A domain-specific **agentic RAG** system that goes beyond simple retrieval:

| Capability | How |
|---|---|
| Domain understanding | Financial glossary with 30+ terms and synonyms |
| Query decomposition | Complex questions split into independent sub-queries |
| Self-correction | Rewrites and retries when retrieved context is irrelevant |
| Hallucination guard | Every answer validated against source chunks before returning |
| Source citations | Responses include company, filing type, year, section, and verbatim quote |
| Observability | Full pipeline traced in Langfuse (per-node latency, inputs, outputs) |

---

## Architecture

### Full Pipeline

```
User Query
    │
    ▼
POST /agent/query
    │
    ▼
┌───────────────────────────── LangGraph State Machine ──────────────────────────────┐
│                                                                                     │
│  ┌─────────────┐     ┌──────────────────────────────────────┐     ┌─────────────┐ │
│  │  decompose  │────►│              retrieve                │────►│  synthesize │ │
│  │             │     │                                      │     │             │ │
│  │ splits into │     │  for each sub-query:                 │     │ generates   │ │
│  │ 1-5 sub-    │     │  ┌─────────────────────────────────┐│     │ answer with │ │
│  │ queries     │     │  │     Self-Correction Loop         ││     │ citations   │ │
│  └─────────────┘     │  │                                  ││     └──────┬──────┘ │
│                      │  │  HybridRetriever (dense+sparse)  ││            │        │
│                      │  │       ↓                          ││            ▼        │
│                      │  │  RelevanceGrader (LLM)           ││     ┌─────────────┐ │
│                      │  │       ↓ irrelevant?              ││     │    check    │ │
│                      │  │  QueryRewriter (glossary + LLM)  ││     │             │ │
│                      │  │       ↓ retry (max 3)            ││     │ hallucina-  │ │
│                      │  │  BGE Reranker (cross-encoder)    ││     │ tion check  │ │
│                      │  └─────────────────────────────────┘│     └─────────────┘ │
│                      │  deduplicate chunks across sub-queries│                    │
│                      └──────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────────── ┘
    │
    ▼
AgentResponse { answer, citations, sub_queries, is_hallucinated, groundedness, trace_id }
```

### Infrastructure

```
                 ┌────────────────────────────────────────┐
                 │              FastAPI App                │
                 │  /chat  /agent/query  /agent/trace/{id} │
                 │  /documents/upload   /health            │
                 └───────┬──────────────────┬─────────────┘
                         │                  │
              ┌──────────▼──────┐  ┌────────▼────────┐
              │      Redis      │  │     Qdrant       │
              │                 │  │                  │
              │ • conversations │  │ • hybrid search  │
              │ • rate limits   │  │   (dense+sparse) │
              │ • circuit break │  │ • metadata filter│
              └─────────────────┘  └──────────────────┘
                         │
              ┌──────────▼──────┐
              │    Langfuse     │
              │  (self-hosted)  │
              │                 │
              │ • trace/span    │
              │ • node latency  │
              │ • I/O logging   │
              └─────────────────┘
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI + uvicorn | Async-first, OpenAPI docs, Pydantic integration |
| Orchestration | LangGraph | Explicit state machine — auditable, visualisable, cycle-ready |
| LLM | OpenAI GPT-4o-mini / Ollama | Swappable via Strategy pattern |
| Embeddings | BAAI/bge-m3 | Dense + sparse in one model, multilingual |
| Reranker | BAAI/bge-reranker-v2-m3 | Cross-encoder accuracy at retrieval scale |
| Vector DB | Qdrant | Native hybrid search (RRF), metadata filters |
| Tracing | Langfuse (self-hosted) | Per-node spans, cost tracking, no vendor lock-in |
| Cache | Redis | Conversation history, rate limits, circuit breaker state |
| Validation | Pydantic v2 | Type-safe schemas across all layers |
| Reliability | stamina + custom circuit breaker | Retries with backoff, shared state across workers |
| Packaging | uv + uv.lock | Reproducible builds, 10× faster than pip |
| CI | GitHub Actions | ruff + pytest with 60% coverage gate |

---

## Key Design Decisions

**Why LangGraph over plain async functions?**
The pipeline has four distinct stages with different failure modes. LangGraph makes the state machine explicit — each node is independently testable, the flow is visualisable in LangSmith/Langfuse, and adding a retry cycle (re-synthesize on hallucination) is one `add_conditional_edges` call.

**Why self-correction instead of just better prompts?**
Retrieval quality varies per query. A single-pass system has no way to recover when the first retrieval returns irrelevant context. The self-correction loop detects low relevance via an LLM grader, rewrites with domain synonyms, and retries — the system improves its own inputs rather than hoping the first shot is good enough.

**Why bge-m3 embeddings instead of OpenAI?**
bge-m3 generates dense *and* sparse vectors simultaneously — enabling hybrid search without a separate BM25 index. It's multilingual (handles Russian financial reports), free at inference time, and benchmarks comparably to OpenAI embeddings on financial text.

**Why citation enrichment in the synthesizer?**
The LLM is asked only for `chunk_id` and a verbatim `quote` — all other citation fields (company, year, section) come from the chunk's metadata in our index. This prevents the LLM from hallucinating source metadata while still producing rich, structured citations.

**Why text-based Recall@k in the eval framework instead of chunk IDs?**
Chunk IDs are ephemeral — re-chunking or re-indexing invalidates them. Matching retrieved text against ground-truth context passages is stable across pipeline changes and directly tests what matters: did we surface the relevant information?

**Why a Redis-backed circuit breaker instead of in-memory?**
In-process state breaks across uvicorn workers — each worker has independent counters. Redis gives shared state so a surge of LLM errors trips the breaker across the entire cluster, not just one process.

---

## Evaluation

```bash
# End-to-end answer accuracy: LLM-only vs LLM+RAG
uv run python -m eval.answer_accuracy --k 5
# Retrieval ablation: dense vs sparse vs hybrid vs +rerank
uv run python -m eval.retrieval_ablation --k 5
```

All numbers below are measured on the full **53-question** dataset, n=2,092 indexed chunks, K=5, generator = `llama3.1` (local via Ollama).

### 1. Does RAG actually help? (end-to-end answer accuracy)

Accuracy = the generated answer contains the ground-truth fact (exact figure, digit-normalized; substring for textual facts). The decisive question — is the whole system worth it versus just asking the model?

| Condition | Answer accuracy | |
|---|---|---|
| **LLM only** (no retrieval, parametric memory) | **0.019** (1/53) | baseline |
| LLM + dense retrieval | 0.585 (31/53) | **+30.6×** |
| **LLM + hybrid RAG** (production) | **0.604** (32/53) | **+31.0×** |

The model alone answers ~2% of filing-specific questions correctly (it cannot know "$416,161M in net sales"); with RAG it reaches ~60%. **This 31× lift is the system's reason to exist**, and it is now measured, not asserted.

### 2. Where does the remaining 40% go? (the generation gap)

Retrieval finds the right chunk **88.7%** of the time, but the answer is correct only **60.4%** of the time — a **~28-point gap** that is pure generation loss: `llama3.1` is handed the correct context and still fails to extract the figure. The bottleneck is the local 8B generator, not retrieval — swapping in `gpt-4o` would close most of this gap (the architecture is sound; the cheap model is the ceiling).

### 3. Retrieval ablation — was hybrid search worth it?

| Retrieval | Recall@5 | MRR |
|---|---|---|
| Dense-only (bge-m3) | 0.887 | **0.789** |
| Sparse-only (bge-m3 lexical) | 0.811 | 0.567 |
| Hybrid RRF (dense + sparse) | 0.887 | 0.723 |
| Hybrid RRF + cross-encoder rerank | 0.792 | 0.649 |

Two honest, counter-intuitive findings the eval surfaced:

> **Hybrid did *not* beat dense here.** Same Recall@5 (0.887), and dense actually has the *better* MRR (0.789 vs 0.723) — RRF fusion with the weaker sparse list nudges the top dense hit down a rank. On this number-anchored eval, multilingual bge-m3 dense is already strong; hybrid's payoff would show on queries dominated by rare exact tokens (tickers, CUSIPs, GAAP line-item codes) where sparse dominates. Worth keeping for robustness, but not a free win — and I can now say that with data instead of marketing.

> **Reranking *hurts*.** The cross-encoder drops Recall@5 from 0.887 to 0.792: it re-ranks on holistic semantic similarity and pushes the literal-number chunk below more "topically fluent" passages. For fact-lookup over financial tables, lexical signal beats cross-attention — so rerank is **off** by default in the pipeline.

The self-correction loop targets the residual ~11% of retrieval misses: the grader flags a query that returns no relevant chunk in top-5, and the rewriter re-issues it with expanded terminology (e.g. "AWS revenue" → "Amazon Web Services net sales for the period").

> **Known limitation — the full agentic path is not in the table.** Every number above is measured, but the end-to-end agentic pipeline (decompose → self-correct → synthesize → hallucination-check) is deliberately excluded rather than estimated, for three reasons: (1) **metric definition** — the agentic path decomposes a query into sub-queries and *accumulates* their top-k results into a variable-size chunk set, so Recall@5 is no longer comparable to the single-query rows; (2) **missing plumbing** — `RAGAgentGraph.run()` returns the answer but not the per-query `SelfCorrectionResult`, so `% rewrites` / `avg iterations` cannot be reported (currently hardcoded in `run_eval.py`); (3) **cost/reliability** — 6–15 LLM calls per query × 53 questions on a local 8B model is slow, and llama3.1's flaky JSON degrades the grader/decomposer into fallbacks, which would measure a partially-disabled agent. The right fix is to evaluate it via **answer accuracy** (well-defined regardless of decomposition) and surface the loop stats from the graph.

> **Reproducibility note:** the eval requires `transformers==4.56.x` for FlagEmbedding 1.4.0 / bge-m3 — the version resolved in `uv.lock` (5.8.x) breaks the bge-m3 tokenizer.

Dataset: `eval/dataset.jsonl` — 53 QA pairs with ground-truth answers and validated source anchors across AAPL, AMZN, META, MSFT, NVDA (15 hand-curated + 38 authored, every anchor verified to appear verbatim in the indexed corpus). Balanced across Income Statement, Balance Sheet, Cash Flow, MD&A, and Risk Factors sections.

---

## Project Status

| Stage | Status | Description |
|---|---|---|
| Stage 1 | ✅ Complete | FastAPI + LLM, streaming SSE, Redis history, reliability, 18 tests |
| Stage 2 | ✅ Complete | Agentic RAG — LangGraph, Qdrant hybrid search, self-correction, hallucination check, Langfuse tracing, 141 tests |
