"""
End-to-end answer accuracy: LLM-only vs LLM+RAG (the headline value test).

For each question we generate an answer under three conditions and check whether
it contains the ground-truth fact (digit-normalized for figures, substring for
text). This answers the decisive question — does retrieval actually make the
model answer correctly, versus just asking it cold?

Conditions:
  - no-RAG:   the question alone (parametric memory)
  - dense:    top-k dense retrieval as context
  - hybrid:   top-k hybrid RRF retrieval as context (production path)

The LLM is called directly (Redis-free, no circuit breaker) — this is offline
eval tooling, not the serving path. Generator = settings.llm_provider model.

Requires: Qdrant up, the LLM reachable, transformers==4.56.x for bge-m3.

Usage:
    uv run python -m eval.answer_accuracy [--k 5] [--limit N] [--concurrency 3]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from openai import AsyncOpenAI
from qdrant_client.models import Fusion, FusionQuery, Prefetch

from app.core.config import settings
from app.rag.retrieval import HybridRetriever
from app.rag.vector_store import COLLECTION_NAME

DATASET = Path(__file__).parent / "dataset.jsonl"

_NORAG_SYS = (
    "You are a financial analyst assistant. Answer the question with the specific figure or "
    "fact from SEC filings. Be concise (1-2 sentences). If you don't know the exact figure, "
    "say so."
)
_RAG_SYS = (
    "You are a financial analyst assistant. Answer the question using ONLY the provided "
    "context. Include the exact figure from the context. Be concise (1-2 sentences)."
)


def _hit(anchor: str, answer: str) -> bool:
    """True if the answer contains the gold fact (substring, or digit-normalized for numbers)."""
    al, ansl = anchor.lower(), (answer or "").lower()
    if al in ansl:
        return True
    na, nans = re.sub(r"[,\s]", "", al), re.sub(r"[,\s]", "", ansl)
    return bool(na) and any(c.isdigit() for c in na) and na in nans


def _build_llm() -> tuple[AsyncOpenAI, str]:
    if settings.llm_provider.lower() == "ollama":
        return AsyncOpenAI(base_url=settings.ollama_base_url, api_key="ollama", timeout=60.0), (
            settings.ollama_model
        )
    return AsyncOpenAI(api_key=settings.openai_api_key, timeout=60.0), settings.openai_model


async def main(k: int, limit: int | None, concurrency: int) -> None:
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    print(f"Loaded {len(rows)} questions")

    retriever = HybridRetriever()
    client = retriever._client
    llm, model = _build_llm()
    print(f"Generator: {model} (provider={settings.llm_provider})")

    def dense_ctx(vec: list[float]) -> str:
        res = client.query_points(
            collection_name=COLLECTION_NAME, query=vec, using="dense", limit=k, with_payload=True
        )
        return "\n\n".join(p.payload.get("text", "")[:600] for p in res.points)

    def hybrid_ctx(d: list[float], s: object) -> str:
        res = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=d, using="dense", limit=20),
                Prefetch(query=s, using="sparse", limit=20),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
            with_payload=True,
        )
        return "\n\n".join(p.payload.get("text", "")[:600] for p in res.points)

    async def ask(system: str, user: str) -> str:
        resp = await llm.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    sem = asyncio.Semaphore(concurrency)
    scores = {"LLM-only (no RAG)": 0, "LLM + dense RAG": 0, "LLM + hybrid RAG": 0}
    done = [0]

    async def run_one(s: dict) -> None:
        d, sp = retriever._embed_query(s["question"])
        dctx, hctx = dense_ctx(d), hybrid_ctx(d, sp)
        anchor = s["contexts"][0]
        async with sem:
            a_no = await ask(_NORAG_SYS, s["question"])
            a_d = await ask(_RAG_SYS, f"Context:\n{dctx}\n\nQuestion: {s['question']}")
            a_h = await ask(_RAG_SYS, f"Context:\n{hctx}\n\nQuestion: {s['question']}")
        if _hit(anchor, a_no):
            scores["LLM-only (no RAG)"] += 1
        if _hit(anchor, a_d):
            scores["LLM + dense RAG"] += 1
        if _hit(anchor, a_h):
            scores["LLM + hybrid RAG"] += 1
        done[0] += 1
        print(f"  [{done[0]}/{len(rows)}] {s['id']}", flush=True)

    await asyncio.gather(*[run_one(s) for s in rows])

    n = len(rows)
    print(f"\n| {'Condition':<22} | accuracy |")
    print(f"|{'-' * 24}|{'-' * 10}|")
    for name, v in scores.items():
        print(f"| {name:<22} | {v / n:>6.3f} ({v}/{n}) |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main(k=args.k, limit=args.limit, concurrency=args.concurrency))
