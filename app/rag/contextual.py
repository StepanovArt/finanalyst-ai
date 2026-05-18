"""
Contextual Retrieval: enrich each chunk with a context prefix.

Technique from Anthropic: https://www.anthropic.com/news/contextual-retrieval
Adding context dramatically improves retrieval because raw chunks often lack
information about which company/period they belong to.

Two modes:
- deterministic: build prefix from structured metadata (free, instant)
- llm: generate 1-sentence summary via LLM (better quality, costs tokens)

The final contextualized_text = context_prefix + "\n\n" + text
This is what gets embedded and stored in Qdrant.
"""

import asyncio
from dataclasses import dataclass

import httpx

from app.rag.chunking import Chunk

_LLM_PROMPT = """\
You are indexing SEC financial filings for semantic search.

Document: {company} ({ticker}) {filing_type} {quarter} {year}, section: {section}.

Chunk text (first 600 chars):
{text_preview}

Write exactly 1 sentence describing the specific financial information in this chunk \
(metrics, comparisons, time periods). No preamble."""

_OLLAMA_URL = "http://localhost:11434/api/generate"
_OLLAMA_MODEL = "llama3.1"
_REQUEST_TIMEOUT = 60.0


@dataclass
class ContextualChunk:
    """Chunk enriched with a context prefix for embedding."""

    chunk: Chunk
    context_prefix: str

    @property
    def contextualized_text(self) -> str:
        return f"{self.context_prefix}\n\n{self.chunk.text}"

    def to_dict(self) -> dict[str, str | int]:
        d = self.chunk.to_dict()
        d["context_prefix"] = self.context_prefix
        d["contextualized_text"] = self.contextualized_text
        return d


def build_deterministic_prefix(chunk: Chunk) -> str:
    """Build context prefix purely from chunk metadata — no LLM needed."""
    quarter_label = chunk.quarter if chunk.quarter == "FY" else f"{chunk.quarter}"
    return (
        f"{chunk.company} ({chunk.ticker}) {chunk.filing_type} "
        f"{quarter_label} {chunk.year}. "
        f"Section: {chunk.section}. "
        f"Currency: {chunk.currency}."
    )


async def _call_ollama(prompt: str) -> str:
    """Call Ollama API directly (no circuit breaker — this is a data pipeline)."""
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        resp = await client.post(
            _OLLAMA_URL,
            json={"model": _OLLAMA_MODEL, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()


async def build_llm_prefix(chunk: Chunk) -> str:
    """Generate context prefix via LLM. Falls back to deterministic on error."""
    prompt = _LLM_PROMPT.format(
        company=chunk.company,
        ticker=chunk.ticker,
        filing_type=chunk.filing_type,
        quarter=chunk.quarter,
        year=chunk.year,
        section=chunk.section,
        text_preview=chunk.text[:600],
    )
    try:
        llm_sentence = await _call_ollama(prompt)
        base = build_deterministic_prefix(chunk)
        return f"{base} {llm_sentence}"
    except Exception:
        return build_deterministic_prefix(chunk)


def add_deterministic_context(chunks: list[Chunk]) -> list[ContextualChunk]:
    """Enrich all chunks with deterministic context prefixes (synchronous)."""
    return [ContextualChunk(chunk=c, context_prefix=build_deterministic_prefix(c)) for c in chunks]


async def add_llm_context(
    chunks: list[Chunk],
    concurrency: int = 4,
) -> list[ContextualChunk]:
    """Enrich chunks with LLM-generated context prefixes (async, batched)."""
    semaphore = asyncio.Semaphore(concurrency)

    async def process(chunk: Chunk) -> ContextualChunk:
        async with semaphore:
            prefix = await build_llm_prefix(chunk)
            return ContextualChunk(chunk=chunk, context_prefix=prefix)

    return await asyncio.gather(*[process(c) for c in chunks])
