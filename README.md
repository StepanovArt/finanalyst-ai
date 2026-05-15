# FinAnalyst AI

Agentic AI assistant for financial analysts working with SEC filings (10-Q, 10-K).
Answers complex financial questions in seconds instead of hours of manual reading.

> **Portfolio project** built to demonstrate production-grade AI engineering practices:
> agentic architecture, RAG pipeline, reliability patterns, and rigorous testing.

---

## Problem

Financial analysts spend 3–5 hours per day reading company filings. Generic chatbots
don't understand financial terminology, confuse GAAP/non-GAAP metrics, and can't
compare metrics across companies or periods.

## Solution

Domain-specific agentic RAG system with:
- Financial terminology understanding (EBITDA, FCF, OPEX, non-GAAP adjustments)
- Self-correction loops — rewrites queries when retrieved context is irrelevant
- Hallucination validation before returning answers
- Source citations with page references

---

## Architecture

```
Client
  │
  ▼
FastAPI  ──► Guard (prompt injection defense)
  │
  ├──► POST /chat        → LLMProvider → OpenAI / Ollama
  ├──► POST /chat/stream → SSE streaming
  └──► GET  /health
  │
  ├── Rate limiting (slowapi + Redis)
  ├── Circuit breaker (Redis-backed)
  ├── Retry with backoff (stamina)
  └── Structured logging (loguru + request_id)

Redis  ── conversation history (TTL 24h)
       └─ rate limit counters
       └─ circuit breaker state
```

**Stage 2 (coming):** LangGraph agents + Qdrant RAG + bge-m3 embeddings
**Stage 3 (coming):** LoRA fine-tuning on Qwen2.5-3B with Unsloth

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + uvicorn |
| Validation | Pydantic v2 + pydantic-settings |
| LLM | OpenAI API (gpt-4o-mini) / Ollama (local) |
| Storage | Redis (conversation history + rate limiting) |
| Reliability | stamina (retries), slowapi (rate limiting), custom circuit breaker |
| Logging | loguru (structured, request_id context) |
| Packaging | uv + uv.lock (reproducible builds) |
| Testing | pytest-asyncio, httpx.AsyncClient, fakeredis |
| Containers | Docker multi-stage + docker-compose |
| CI | GitHub Actions (ruff + pytest --cov-fail-under=60) |

---

## Quick Start

### Option 1: Docker Compose (recommended)

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env

docker-compose up --build
```

### Option 2: Local with uv

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env

uv sync
uv run uvicorn app.main:app --reload
```

Service runs at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

---

## Configuration

All settings are loaded from `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `OPENAI_API_KEY` | — | Required when using OpenAI |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Local model name |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection |
| `LLM_TIMEOUT_SECONDS` | `30.0` | Request timeout |
| `LLM_MAX_ATTEMPTS` | `3` | Retry attempts |
| `RATE_LIMIT_CHAT` | `20/minute` | Per-IP limit for /chat |
| `RATE_LIMIT_STREAM` | `10/minute` | Per-IP limit for /chat/stream |

---

## API

### `POST /chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What was Apple revenue in Q3 2024?"}],
    "conversation_id": "my-session-123"
  }'
```

```json
{
  "content": "Apple reported total net sales of $85.8B in Q3 FY2024...",
  "model": "OpenAIProvider",
  "usage": null,
  "created_at": "2026-05-15T10:00:00Z"
}
```

### `POST /chat/stream`

Same request body — returns `text/event-stream`:

```
data: {"content": "Apple"}
data: {"content": " reported"}
data: {"content": " total"}
...
data: [DONE]
```

### `GET /health`

```json
{"status": "ok", "service": "finanalyst-ai"}
```

---

## Key Design Decisions

**Why Strategy Pattern for LLM providers?**
Allows switching OpenAI ↔ Ollama via a single env var. Enables A/B testing.
Decouples business logic from provider SDKs.

**Why SSE instead of WebSocket for streaming?**
Server-to-client only — no need for full-duplex. Simpler protocol over plain HTTP.
WebSocket reserved for future voice-assistant stage.

**Why Redis-backed circuit breaker?**
In-memory state breaks with multiple uvicorn workers — each process has independent
counters. Redis gives shared state across all workers.

**Why `httpx.AsyncClient` + `ASGITransport` in tests, not `TestClient`?**
`TestClient` is synchronous and blocks the event loop. All FastAPI endpoints are
`async def` — only an async client tests them correctly.

---

## Development

```bash
# Install dev dependencies
uv sync

# Run tests with coverage
uv run pytest tests/ --cov=app --cov-report=term-missing

# Lint and format
uv run ruff check app/ tests/
uv run ruff format app/ tests/
```

CI runs automatically on push to `main` and on all pull requests.

---

## Project Status

| Stage | Status | Description |
|---|---|---|
| Stage 1 | ✅ Complete | FastAPI service with LLM, reliability, tests |
| Stage 2 | Planned | Agentic RAG (LangGraph + Qdrant + bge-m3) |
| Stage 3 | Planned | LoRA fine-tuning (Qwen2.5-3B + Unsloth) |
