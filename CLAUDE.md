# FinAnalyst AI — Project Context for Claude Code

> This file is automatically read by Claude Code at the start of every session.
> It contains the full project context, conventions, and current state.
> Keep it updated as the project evolves.

---

## 🎯 Project Overview

**FinAnalyst AI** is an agentic AI assistant for financial analysts working with quarterly and annual company reports (10-Q, 10-K, annual reports). The system automates the work of reading financial filings, allowing users to get precise answers to complex financial questions in seconds instead of hours of manual reading.

### Problem Being Solved
Financial analysts and investors spend 3-5 hours per day reading company filings. Standard tools (PDF Ctrl+F, generic chatbots) don't understand financial terminology, confuse GAAP/non-GAAP metrics, don't handle abbreviations (EBITDA, OPEX, FCF, ЧОД), and can't compare metrics across companies.

### Target Users
- Financial analysts (sell-side and buy-side)
- Retail investors studying company filings
- CFA/ACCA candidates and finance students
- Business journalists

### Unique Value Proposition
Unlike a standard RAG chatbot, this system uses an **agentic architecture with self-correction**: it understands domain terminology, automatically rephrases queries when retrieved results are irrelevant, validates answers for hallucinations, and cites specific report pages.

### Project Purpose (Important Context for Claude)
This project is being built as a **portfolio piece for AI Engineer job interviews**, NOT for production deployment or real users. Therefore:
- Code quality, architecture, and testing matter MORE than UI/UX
- Demonstrable trade-offs and design decisions matter MORE than feature completeness
- Local `docker-compose up` workflow is sufficient — no cloud deployment needed
- Every architectural choice should be defensible in an interview setting

---

---

## 🛠 Tech Stack

### Backend & API
- **Python 3.11** (NOT 3.12 — some ML libs are flaky)
- **FastAPI** — async web framework
- **Pydantic v2** — validation and schemas (NOT v1 syntax)
- **pydantic-settings** — config management via .env
- **httpx** (async) — HTTP client (NEVER use `requests`)
- **uvicorn[standard]** — ASGI server

### LLM & ML
- **OpenAI API** (gpt-4o, gpt-4o-mini) — primary LLM provider during development
- **Ollama** — local inference alternative
- **Qwen2.5-3B-Instruct** — base model for fine-tuning (Stage 3)
- **Unsloth** — efficient LoRA fine-tuning
- **PEFT, TRL, transformers** — HuggingFace ecosystem
- **sentence-transformers (BAAI/bge-m3)** — multilingual embeddings (works well with Russian/English)
- **BAAI/bge-reranker-v2-m3** — reranker model

### Agentic Framework
- **LangGraph** — agent orchestration (state machine pattern)
- **Langfuse** (self-hosted) — agent tracing and observability

### Data & Storage
- **Qdrant** — vector DB with native hybrid search
- **Redis** — cache and conversation history
- **PyMuPDF** — fast PDF text extraction
- **pdfplumber** — financial table extraction from PDFs

### Evaluation
- **RAGAS** — RAG quality metrics
- **Weights & Biases** — fine-tuning experiment tracking

### Quality & Infrastructure
- **uv** — Python package management (FAST — preferred over pip/poetry)
- **Ruff** — linting and formatting (replaces black + flake8 + isort)
- **pytest** + **pytest-asyncio** — testing
- **httpx.AsyncClient** for FastAPI testing (NOT TestClient — it's sync)
- **Docker + docker-compose** — local development
- **GitHub Actions** — CI

### Reliability & Observability
- **loguru** — structured logging
- **stamina** — retry with exponential backoff (modern alternative to tenacity)
- **slowapi** — rate limiting

---

## 📅 Project Stages

### Stage 1: FastAPI Service with LLM (5-7 days)
Production-ready backend with LLM integration, specialized for financial domain. Foundation for everything else.

### Stage 2: Agentic RAG with Self-Correction (14-18 days)
State-of-the-art agentic RAG system implementing Self-RAG and Corrective RAG patterns. Includes domain terminology handling, self-correction loops, and rigorous evaluation.

### Stage 3: LoRA Fine-tuning (10-14 days)
Fine-tune Qwen2.5-3B on financial domain for structured outputs. Integrate the fine-tuned model into the agentic RAG as the Answer Synthesis Agent.

---

## 📍 Current Stage

**Stage 2: Agentic RAG with Self-Correction**

**Current Task:** 2.4.1 — Langfuse observability

---

### Stage 1: FastAPI Service with LLM ✅ Complete

- [x] 1.1 — Project initialization (git, GitHub, CLAUDE.md, README.md, .gitignore)
- [x] 1.2 — Basic FastAPI skeleton with /health endpoint
- [x] 1.3 — LLMProvider abstraction (Strategy pattern)
- [x] 1.4 — Pydantic schemas for chat
- [x] 1.5 — Financial domain system prompt
- [x] 1.6 — /chat endpoint
- [x] 1.7 — Streaming via SSE (POST /chat/stream)
- [x] 1.8 — Conversation history with Redis
- [x] 1.9 — Reliability layer (retries, timeouts, circuit breaker)
- [x] 1.10 — Rate limiting (slowapi + Redis)
- [x] 1.11 — Basic prompt injection defense
- [x] 1.12 — Structured logging with request_id
- [x] 1.13 — Tests (18 tests, 73% coverage)
- [x] 1.14 — Docker + docker-compose (uv.lock, .dockerignore, Redis volume)
- [x] 1.15 — GitHub Actions CI (ruff + pytest --cov-fail-under=60)
- [x] 1.16 — README documentation

---

### Stage 2 — Agentic RAG

#### 2.1 Data Collection & Preparation
- [x] 2.1.1 — Download 20-30 SEC EDGAR filings (10-Q, 10-K) for 5-7 companies
- [x] 2.1.2 — HTML parsing: BeautifulSoup + lxml (EDGAR filings are HTML, not PDF)
- [x] 2.1.3 — Structured chunking by report sections (Income Statement, Balance Sheet, MD&A)
- [x] 2.1.4 — Doc-level metadata (company, ticker, year, quarter, report type, currency)
- [x] 2.1.5 — Contextual Retrieval: LLM-generated context prefix for each chunk

#### 2.2 Basic Retrieval
- [x] 2.2.1 — Embeddings: BAAI/bge-m3 via FlagEmbedding (dense 1024d + sparse)
- [x] 2.2.2 — Qdrant: Docker setup, collection schema with metadata filters
- [x] 2.2.3 — Hybrid Search: dense + sparse via Reciprocal Rank Fusion
- [x] 2.2.4 — Reranker: BAAI/bge-reranker-v2-m3
- [x] 2.2.5 — POST /documents/upload endpoint with background processing

#### 2.3 Agentic Layer
- [x] 2.3.1 — Domain Glossary: JSON with financial terms and synonyms (EBITDA, OPEX, FCF etc.)
- [x] 2.3.2 — Query Decomposition Agent: splits complex queries into sub-queries
- [x] 2.3.3 — Query Rewriting Agent: rephrases with glossary synonyms
- [x] 2.3.4 — Relevance Grader Agent: LLM validates each chunk (relevant/irrelevant/ambiguous)
- [x] 2.3.5 — Self-Correction Loop: rephrase + retry if low relevance (max 3 iterations)
- [x] 2.3.6 — Answer Synthesis Agent: generates answer with page citations
- [x] 2.3.7 — Hallucination Check: validates answer against context (Self-RAG)
- [x] 2.3.8 — LangGraph orchestration: assembles all agents into state-machine

#### 2.4 Observability & API
- [ ] 2.4.1 — Langfuse: self-hosted in docker-compose, tracing integration
- [ ] 2.4.2 — POST /agent/query endpoint with trace_id in response
- [ ] 2.4.3 — GET /agent/trace/{id} endpoint

#### 2.5 Evaluation
- [ ] 2.5.1 — Eval dataset: 50 QA pairs with ground truth answers and sources
- [ ] 2.5.2 — Retrieval metrics: Recall@k, MRR
- [ ] 2.5.3 — Generation metrics via RAGAS: Faithfulness, Answer Relevancy, Context Precision/Recall
- [ ] 2.5.4 — Agentic metrics: % rewrites, avg iterations, latency, cost per query
- [ ] 2.5.5 — Comparison table: naive RAG → hybrid+rerank → +contextual → full agentic

---

### Stage 3 — LoRA Fine-tuning
Будет выполнен отдельно на Kaggle как standalone notebook.
Результат интегрируется через LocalLoRAProvider после завершения.

---

## 📁 Target Project Structure
finanalyst-ai/
├── app/
│   ├── init.py
│   ├── main.py              # FastAPI app entrypoint
│   ├── config.py            # pydantic-settings configuration
│   ├── schemas.py           # Pydantic request/response models
│   ├── routers/
│   │   ├── init.py
│   │   ├── chat.py          # /chat, /chat/stream endpoints
│   │   ├── conversations.py # /conversations/{id}
│   │   └── health.py        # /health
│   ├── services/
│   │   ├── init.py
│   │   ├── llm/             # LLM provider abstraction
│   │   │   ├── init.py
│   │   │   ├── base.py      # LLMProvider abstract class
│   │   │   ├── openai_provider.py
│   │   │   └── ollama_provider.py
│   │   └── conversation.py  # Conversation history (Redis)
│   ├── agents/              # Stage 2: LangGraph agents
│   │   ├── init.py
│   │   ├── decomposer.py
│   │   ├── rewriter.py
│   │   ├── grader.py
│   │   └── synthesizer.py
│   ├── rag/                 # Stage 2: RAG pipeline
│   │   ├── init.py
│   │   ├── ingestion.py
│   │   ├── chunking.py
│   │   ├── retrieval.py
│   │   └── reranker.py
│   ├── data/                # Domain knowledge
│   │   └── financial_glossary.json
│   └── core/
│       ├── init.py
│       ├── logging.py       # loguru configuration
│       ├── exceptions.py    # Custom exceptions
│       └── middleware.py    # Request ID, etc.
├── tests/
│   ├── init.py
│   ├── conftest.py
│   └── test_*.py
├── notebooks/               # Stage 3: Fine-tuning notebooks
│   └── finetune_qwen_lora.ipynb
├── eval/                    # Evaluation datasets and scripts
│   ├── dataset.jsonl
│   └── run_eval.py
├── docs/                    # Architecture diagrams, screenshots
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
└── CLAUDE.md


---

## 📐 Code Conventions

### Python Style
- **Type hints everywhere** — every function signature, no exceptions
- **Async-first** — use `async def` for any I/O-bound code
- **Docstrings** in Google style for public functions/classes
- **Line length:** 100 characters (Ruff default)
- **Imports:** sorted by Ruff (isort rules)
- **f-strings** for formatting (never `.format()` or `%`)

### Pydantic
- Always use **Pydantic v2 syntax** (`model_config`, `Field(...)`, etc.)
- Validate at the API boundary, trust the data inside
- Use `Literal` for enums when possible

### Naming
- `snake_case` for variables, functions, modules
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants
- Private methods/functions: `_leading_underscore`

### Error Handling
- Custom exceptions for domain errors (in `app/core/exceptions.py`)
- Global exception handlers in FastAPI map them to HTTP codes
- NEVER catch bare `Exception` without re-raising or logging

### Testing
- pytest + pytest-asyncio
- Use `httpx.AsyncClient(transport=ASGITransport(app=app))` for endpoint tests
- Mock external services (LLM providers) with `pytest-mock` or `respx`
- Target 60%+ coverage on app code

---

## ⚠️ Critical Rules — What NOT To Do

These rules exist because violating them causes real problems:

1. **NEVER hardcode API keys.** All secrets through `pydantic-settings` + `.env`. The `.env` is in `.gitignore`.
2. **NEVER use sync `requests`.** Use `httpx.AsyncClient` always — sync HTTP in async code blocks the event loop.
3. **NEVER use Pydantic v1 syntax** (`@validator`, `Config` class, `.dict()`). Use v2 (`@field_validator`, `model_config`, `.model_dump()`).
4. **NEVER skip tests "because it's MVP".** Tests are the portfolio signal that separates juniors from middles.
5. **NEVER write giant `main.py`.** Always split into routers and services.
6. **NEVER commit `.env`, `__pycache__/`, `.venv/`, model weights, or PDFs.** The `.gitignore` should handle this — never override it.
7. **NEVER use `print()` for logging.** Use `loguru`.
8. **NEVER catch bare `Exception` silently.** Log it, re-raise it, or wrap it in a custom exception.
9. **NEVER use ChromaDB as the vector DB.** We chose Qdrant for production-grade hybrid search.
10. **NEVER use TestClient for FastAPI tests.** It's sync. Use `httpx.AsyncClient` + `ASGITransport`.

---

## 🎯 Key Architectural Decisions

Document these so I can defend them in interviews:

### Why FastAPI (not Flask/Django)?
- Async-first design fits LLM workloads (I/O bound, long-running requests)
- Automatic OpenAPI/Swagger docs generation
- Pydantic integration for type-safe validation
- Modern, performant, industry standard for AI services in 2025-2026

### Why Strategy Pattern for LLM Providers?
- Allows swapping OpenAI ↔ Anthropic ↔ Ollama ↔ Fine-tuned Qwen via a single env var
- Enables A/B testing of different models in production
- Decouples business logic from provider-specific SDKs

### Why SSE for Streaming (not WebSocket)?
- Server-to-client only — no need for full-duplex
- Simpler protocol, works over HTTP
- Browser-native support via EventSource API
- WebSocket reserved for future voice-assistant stage (bidirectional audio)

### Why Qdrant (not Pinecone/Weaviate/ChromaDB)?
- Open source — no vendor lock-in, free to self-host
- Native hybrid search (dense + sparse with RRF) — critical for financial domain
- Better metadata filtering than ChromaDB
- Production-grade performance (used by major AI companies)

### Why LangGraph (not LangChain agents)?
- Explicit state machine — easier to reason about and debug
- Better visualization of agent workflows
- Native support for cycles (self-correction loops)
- Industry trend in 2025-2026 for production agentic systems

### Why bge-m3 Embeddings (not OpenAI)?
- Multilingual (works with Russian financial reports)
- Generates dense + sparse embeddings simultaneously — enables hybrid search without separate BM25 index
- Free, runs locally, no API costs at retrieval scale
- Top-tier benchmark performance

### Why LoRA + QLoRA (not full fine-tuning)?
- 1000× fewer trainable parameters → fits on consumer GPUs (T4, A100)
- 4-bit base model quantization saves memory
- Multiple LoRA adapters can be hot-swapped for different tasks
- Standard approach for resource-efficient adaptation in 2025-2026

---

## 💬 How to Work With Me (Claude Code)

When the user gives you a task:

1. **Read this CLAUDE.md fully first** if it's the start of a session.
2. **Ask 1-3 clarifying questions** before writing code if anything is ambiguous.
3. **Explain your approach** before implementing — what files you'll touch and why.
4. **Implement incrementally** — small steps the user can verify.
5. **Explain key decisions inline** — why this library, why this pattern, why this trade-off.
6. **After each task, suggest an interview question** the user might be asked about what was just built — this is a learning project.
7. **Suggest a git commit message** at the end of each task in conventional commits format (e.g., `feat: add LLMProvider abstraction with OpenAI implementation`).
8. **Update the "Current Stage" section** of CLAUDE.md after completing each task.

The user is **learning** through this project. Optimize for **understanding**, not just shipped code. If a concept is non-trivial, explain the intuition first, then the formal details.

---

## 📚 Reference Patterns to Use

### LLMProvider Abstraction (Strategy Pattern)
```python
from abc import ABC, abstractmethod
from typing import AsyncIterator

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, messages: list[Message], **kwargs) -> str: ...
    
    @abstractmethod
    async def stream_generate(self, messages: list[Message], **kwargs) -> AsyncIterator[str]: ...

class OpenAIProvider(LLMProvider): ...
class OllamaProvider(LLMProvider): ...
class LocalLoRAProvider(LLMProvider): ...  # Stage 3
```

### Settings via pydantic-settings
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    
    openai_api_key: str
    llm_provider: Literal["openai", "ollama"] = "openai"
    redis_url: str = "redis://localhost:6379"
```

### FastAPI Endpoint Structure
```python
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm: LLMProvider = Depends(get_llm_provider),
) -> ChatResponse:
    ...
```

---

## 🔗 External Resources

- Project repo: https://github.com/StepanovArt/finanalyst-ai
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- LangGraph docs: https://langchain-ai.github.io/langgraph/
- Qdrant docs: https://qdrant.tech/documentation/
- Unsloth docs: https://github.com/unslothai/unsloth
- RAGAS docs: https://docs.ragas.io/

---

## 📝 Session Log

> Each session, append a brief note here about what was accomplished.
> Format: `YYYY-MM-DD: [Stage X.Y] What was done`

- 2026-05-14: [Stage 1.1] Project initialization — git, GitHub repo, CLAUDE.md, README.md, .gitignore. First commit pushed to origin/main.