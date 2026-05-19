# FinAnalyst AI

> Agentic RAG assistant for financial analysts — answers complex questions about SEC filings in seconds instead of hours of manual reading.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)
![Tests](https://img.shields.io/badge/tests-140%20passing-brightgreen)
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
| Rate limiting | slowapi | Per-IP limits on chat + streaming endpoints |
| Evaluation | RAGAS | Faithfulness, Answer Relevancy, Context Precision/Recall |
| Logging | loguru | Structured logs with request_id context |
| Packaging | uv + uv.lock | Reproducible builds, 10× faster than pip |
| Containers | Docker + docker-compose | One-command local setup with all services |
| CI | GitHub Actions | ruff + pytest with 60% coverage gate |

---

## Quick Start

### Docker Compose (recommended)

```bash
git clone https://github.com/StepanovArt/finanalyst-ai.git
cd finanalyst-ai

cp .env.example .env
# Set OPENAI_API_KEY in .env

docker-compose up --build
```

Starts: **app** (8000) · **Redis** (6379) · **Qdrant** (6333) · **Langfuse** (3000)

### Local with uv

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Swagger UI: `http://localhost:8000/docs`

---

## API Reference

### Agent Query — `POST /agent/query`

Runs the full agentic pipeline: decompose → retrieve → synthesize → check.

```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare Apple and Microsoft operating margins in fiscal 2024",
    "filters": {"year": 2024},
    "session_id": "analyst-session-1"
  }'
```

```json
{
  "query": "Compare Apple and Microsoft operating margins in fiscal 2024",
  "answer": "Apple's operating margin was 31.5% in fiscal 2024, up from 29.8% in 2023. Microsoft's was 44.6%, up from 41.7%. Microsoft leads by ~13 percentage points, driven by high-margin cloud revenue mix.",
  "citations": [
    {
      "chunk_id": "aapl-10k-2024-income-stmt-3",
      "quote": "Total operating expenses were $267,818 million, resulting in operating income of $123,217 million",
      "company": "Apple Inc.",
      "filing_type": "10-K",
      "year": 2024,
      "quarter": "FY",
      "section": "Income Statement"
    }
  ],
  "sub_queries": [
    "Apple operating margin fiscal 2024",
    "Microsoft operating margin fiscal 2024"
  ],
  "is_hallucinated": false,
  "groundedness": "grounded",
  "trace_id": "a3f8c2d1-4b5e-..."
}
```

### Trace Details — `GET /agent/trace/{trace_id}`

Fetches per-node execution spans from Langfuse.

```json
{
  "trace_id": "a3f8c2d1-4b5e-...",
  "input": "Compare Apple and Microsoft operating margins...",
  "output": "Apple's operating margin was 31.5%...",
  "latency_ms": 3240.5,
  "observations": [
    {"name": "decompose",  "type": "SPAN",       "latency_ms": 320.1},
    {"name": "retrieve",   "type": "SPAN",       "latency_ms": 1850.3},
    {"name": "synthesize", "type": "GENERATION", "latency_ms": 890.2},
    {"name": "check",      "type": "SPAN",       "latency_ms": 180.0}
  ],
  "langfuse_url": "http://localhost:3000/trace/a3f8c2d1-..."
}
```

### Chat — `POST /chat`

Conversational endpoint with history (LLM only, no RAG).

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is EBITDA?"}]}'
```

### Streaming — `POST /chat/stream`

Same request body, returns `text/event-stream`:

```
data: {"content": "EBITDA"}
data: {"content": " stands for"}
...
data: [DONE]
```

### Document Upload — `POST /documents/upload`

Index a new SEC filing (background processing).

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@aapl-10k-2024.htm" \
  -F "ticker=AAPL" \
  -F "filing_type=10-K"
# → {"job_id": "...", "status": "pending"}

curl http://localhost:8000/documents/status/{job_id}
# → {"status": "done", "chunks_indexed": 847}
```

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

**Why Redis-backed circuit breaker instead of in-memory?**
In-process state breaks across uvicorn workers — each worker has independent counters. Redis gives shared state so a surge of LLM errors trips the breaker across the entire cluster, not just one process.

---

## Evaluation Framework

```bash
# Run retrieval metrics (Recall@k, MRR) against dataset
uv run python -m eval.run_eval --questions 15 --k 5
```

Output: Markdown comparison table across four pipeline variants:

| Variant | Recall@5 | MRR | Avg Iterations | Rewrites % |
|---|---|---|---|---|
| Naive RAG | — | — | 1.00 | 0% |
| Hybrid + Rerank | — | — | 1.00 | 0% |
| + Contextual | — | — | 1.00 | 0% |
| Full Agentic | — | — | 1.40 | 35% |

> Numbers populate after indexing real filings and running against Qdrant.

Dataset: `eval/dataset.jsonl` — 15 QA pairs with ground-truth answers and source passages across AAPL, MSFT, GOOGL, AMZN, META.

Generation metrics (Faithfulness, Answer Relevancy, Context Precision/Recall) via RAGAS:
```bash
uv sync --extra eval
uv run python -m eval.run_eval
```

---

## Development

```bash
# Tests
uv run python -m pytest -q                              # 140 tests
uv run python -m pytest --cov=app --cov-report=term    # coverage: 77%

# Lint + format
uv run ruff check app/ tests/ eval/
uv run ruff format app/ tests/ eval/
```

CI runs on every push: ruff → pytest with `--cov-fail-under=60`.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai` |
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama endpoint |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection |
| `LLM_TIMEOUT_SECONDS` | `30.0` | Per-request LLM timeout |
| `LLM_MAX_ATTEMPTS` | `3` | Retry attempts with backoff |
| `RATE_LIMIT_CHAT` | `20/minute` | Per-IP limit on `/chat` |
| `LANGFUSE_PUBLIC_KEY` | — | Optional: Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | — | Optional: Langfuse tracing |
| `LANGFUSE_HOST` | `http://localhost:3000` | Langfuse server URL |

---

## Project Status

| Stage | Status | Description |
|---|---|---|
| Stage 1 | ✅ Complete | FastAPI + LLM, streaming SSE, Redis history, reliability, 18 tests |
| Stage 2 | ✅ Complete | Agentic RAG — LangGraph, Qdrant hybrid search, self-correction, hallucination check, Langfuse tracing, 140 tests |
