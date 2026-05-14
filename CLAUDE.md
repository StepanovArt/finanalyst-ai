# FinAnalyst AI — Project Context for Claude

## Project
Agentic AI assistant for financial analysts working with SEC reports (10-Q, 10-K).
Parses, chunks, embeds, and answers questions about quarterly/annual filings via RAG + LLM.

## Tech Stack
- **Runtime**: Python 3.11+, uv (package manager)
- **API**: FastAPI + uvicorn
- **Agents**: LangGraph
- **Vector DB**: Qdrant
- **LLM fine-tuning**: Unsloth + LoRA
- **Validation**: Pydantic v2 + pydantic-settings
- **Logging**: loguru
- **Testing**: pytest + pytest-asyncio
- **Linting**: ruff
- **Containers**: Docker (multi-stage, uv-based) + docker-compose

## Project Layout
```
finanalyst-ai/
├── app/
│   ├── main.py            # FastAPI app factory
│   ├── core/              # config, logging, lifespan
│   ├── routers/           # API route handlers
│   └── services/          # business logic
├── tests/
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

## Development Rules (agreed with user)
- Use `uv` for all package management (not pip/poetry)
- Stop after each task step and wait for user confirmation before proceeding
- No unnecessary comments — only non-obvious WHY
- No mocking internal services in tests; prefer real dependencies where feasible

## Task Roadmap

### Stage 1 — FastAPI service with LLM
- [x] 1.1 Repo init & CLAUDE.md
- [x] 1.2 Basic FastAPI skeleton (pyproject.toml, app layout, /health, Docker)
- [ ] 1.3 LLM Provider abstraction
  - [ ] 1.3.1 base.py — abstract LLMProvider
  - [ ] 1.3.2 openai_provider.py
  - [ ] 1.3.3 ollama_provider.py
  - [ ] 1.3.4 Dependency injection via Depends() + LLM_PROVIDER env var
- [ ] 1.4 Chat endpoint (POST /chat)
- [ ] 1.5 LangGraph agent integration

### Stage 2 — RAG pipeline
- [ ] 2.1 SEC EDGAR ingestion (fetch 10-Q / 10-K)
- [ ] 2.2 Parsing & chunking
- [ ] 2.3 Qdrant integration (embeddings + vector search)
- [ ] 2.4 RAG agent with LangGraph

### Stage 3 — Fine-tuning
- [ ] 3.1 LoRA fine-tuning with Unsloth
