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

## Task Roadmap — Stage 1
- [x] 1.1 Repo init & CLAUDE.md
- [ ] 1.2 Basic FastAPI skeleton
  - [x] 1.2.1 pyproject.toml with uv
  - [ ] 1.2.2 app/ folder structure
  - [ ] 1.2.3 GET /health endpoint
  - [ ] 1.2.4 Dockerfile + docker-compose
- [ ] 1.3 SEC filing ingestion pipeline
- [ ] 1.4 Qdrant integration (RAG)
- [ ] 1.5 LangGraph agent
- [ ] 1.6 LoRA fine-tuning pipeline
