"""
Generate a synthetic evaluation dataset from indexed chunks.

Why synthetic generation?
  The hand-curated set has 15 high-quality QA pairs but is skewed toward AAPL
  and Income Statements. To get a statistically meaningful eval (50+) with
  balanced coverage, we sample chunks across every ticker and section type and
  ask the LLM to write a grounded question + answer for each.

Reliability guardrails:
  - The LLM must return an "anchor": a verbatim phrase from the chunk. We REJECT
    any pair whose anchor is not actually a substring of the source chunk. This
    keeps the retrieval metric (substring match) honest — every ground-truth
    context provably exists in the corpus.
  - Chunks too short or with no concrete content yield null → skipped.
  - Existing dataset is preserved; new pairs are appended with continuing IDs.

Does NOT require Qdrant or Redis — reads chunks straight from
data/processed/chunks.jsonl and calls the LLM directly (no circuit breaker).
Like the contextual-retrieval indexer, this is offline data tooling: it should
not depend on the production serving stack's reliability layer.
Requires only the configured LLM (Ollama or OpenAI) to be reachable.

Usage:
    uv run python scripts/generate_eval_dataset.py --dry-run        # show plan
    uv run python scripts/generate_eval_dataset.py --limit 5        # small test
    uv run python scripts/generate_eval_dataset.py                  # full run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DATASET_PATH = Path("eval/dataset.jsonl")

SAMPLES_PER_CELL = 3  # chunks per (ticker, section_group) cell
MIN_WORDS = 60  # skip chunks shorter than this

SECTION_GROUPS: dict[str, list[str]] = {
    "financials": ["Income Statement", "Balance Sheet", "Cash Flow Statement"],
    "narrative": ["MD&A", "Business Overview"],
    "risk": ["Risk Factors", "Market Risk"],
    "notes": ["Notes to FS", "Equity Statement"],
}

TICKERS = ["AAPL", "AMZN", "META", "MSFT", "NVDA"]

_SYSTEM_PROMPT = """\
You are building an evaluation dataset for a financial RAG system over SEC filings.

Given a document chunk from a 10-K or 10-Q filing, generate exactly ONE high-quality
question-answer pair that tests retrieval and answer quality.

Rules:
- The question must be specific and answerable ONLY from this chunk.
- Include the company name AND year/quarter in the question when relevant.
- The answer must be precise: use exact figures, percentages, or dates from the text.
- The anchor must be a verbatim 4-12 word phrase copied EXACTLY from the chunk text,
  ideally containing a number. It is used for substring matching, so it must appear
  in the chunk character-for-character.
- If the chunk has no specific answerable content (pure headers, boilerplate),
  return null for all fields.

Respond ONLY with a JSON object, no markdown:
{"question": "...", "ground_truth": "...", "anchor": "..."}
Or if unsuitable:
{"question": null, "ground_truth": null, "anchor": null}
"""

_USER_TEMPLATE = """\
Chunk metadata: {company} ({ticker}) {filing_type} {quarter} {year}, section: {section}

Chunk text:
{text}

Generate a question-answer pair as JSON:"""


def _provider_name() -> str:
    from app.core.config import settings

    return settings.llm_provider.lower()


def _section_group(section: str) -> str:
    for group, sections in SECTION_GROUPS.items():
        if section in sections:
            return group
    return "other"


def sample_chunks(chunks: list[dict], rng: random.Random) -> list[dict]:
    """Sample chunks balanced across ticker x section_group cells."""
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in chunks:
        if len(c["text"].split()) < MIN_WORDS:
            continue
        group = _section_group(c["section"])
        if c["ticker"] not in TICKERS or group == "other":
            continue
        cells[(c["ticker"], group)].append(c)

    sampled: list[dict] = []
    for key, cell_chunks in sorted(cells.items()):
        n = min(SAMPLES_PER_CELL, len(cell_chunks))
        sampled.extend(rng.sample(cell_chunks, n))

    rng.shuffle(sampled)
    return sampled


def _build_llm_client() -> tuple[object, str]:
    """Build a Redis-free AsyncOpenAI client for the configured provider.

    Mirrors the production provider's wire config (base_url/key/model) but skips
    the circuit breaker — this is an offline pipeline, not the serving path.
    """
    from openai import AsyncOpenAI

    from app.core.config import settings

    if settings.llm_provider.lower() == "ollama":
        client = AsyncOpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",
            timeout=settings.llm_timeout_seconds,
        )
        return client, settings.ollama_model

    client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=settings.llm_timeout_seconds)
    return client, settings.openai_model


async def _generate_qa(chunk: dict, client, model: str) -> dict | None:  # type: ignore[no-untyped-def]
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(
                company=chunk["company"],
                ticker=chunk["ticker"],
                filing_type=chunk["filing_type"],
                quarter=chunk["quarter"],
                year=chunk["year"],
                section=chunk["section"],
                text=chunk["text"][:1200],
            ),
        },
    ]

    try:
        resp = await client.chat.completions.create(model=model, messages=messages)
        raw = resp.choices[0].message.content or ""
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)

        question = data.get("question")
        anchor = data.get("anchor")
        if not question or not anchor:
            return None

        # CRITICAL: reject anchors not verbatim-present in the chunk.
        if anchor.lower() not in chunk["text"].lower():
            logger.warning(f"  anchor not in chunk, reject: {anchor!r}")
            return None

        return {
            "question": question.strip(),
            "ground_truth": (data.get("ground_truth") or "").strip(),
            "anchor": anchor.strip(),
        }
    except Exception as exc:
        logger.warning(f"  generation failed for {chunk['id']}: {exc}")
        return None


def _next_id_num() -> int:
    if not DATASET_PATH.exists():
        return 1
    nums = []
    for line in DATASET_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.search(r"(\d+)$", json.loads(line)["id"])
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


async def main(limit: int | None, dry_run: bool, seed: int) -> None:
    if not CHUNKS_PATH.exists():
        logger.error(f"Chunks file not found: {CHUNKS_PATH}")
        sys.exit(1)

    chunks = [json.loads(line) for line in CHUNKS_PATH.read_text().splitlines() if line.strip()]
    logger.info(f"Loaded {len(chunks)} chunks")

    rng = random.Random(seed)
    sampled = sample_chunks(chunks, rng)
    if limit:
        sampled = sampled[:limit]
    logger.info(f"Sampled {len(sampled)} chunks")

    if dry_run:
        dist = Counter(f"{c['ticker']}/{_section_group(c['section'])}" for c in sampled)
        print("\nSampling plan (ticker/group: count):")
        for k, v in sorted(dist.items()):
            print(f"  {k}: {v}")
        print(f"\nTotal to generate: {len(sampled)}")
        return

    client, model = _build_llm_client()
    logger.info(f"LLM: {model} (provider={_provider_name()})")
    semaphore = asyncio.Semaphore(4)
    results: list[tuple[dict, dict | None]] = []

    async def process(chunk: dict) -> None:
        async with semaphore:
            qa = await _generate_qa(chunk, client, model)
            results.append((chunk, qa))
            logger.info(f"[{'OK' if qa else 'SKIP'}] {chunk['ticker']} {chunk['section']}")

    await asyncio.gather(*[process(c) for c in sampled])

    next_num = _next_id_num()
    new_records: list[dict] = []
    for chunk, qa in results:
        if qa is None:
            continue
        new_records.append(
            {
                "id": f"q{next_num:03d}",
                "question": qa["question"],
                "ground_truth": qa["ground_truth"],
                "contexts": [qa["anchor"]],
                "metadata": {
                    "ticker": chunk["ticker"],
                    "company": chunk["company"],
                    "year": chunk["year"],
                    "quarter": chunk["quarter"],
                    "filing_type": chunk["filing_type"],
                    "section": chunk["section"],
                },
            }
        )
        next_num += 1

    n_skip = len(sampled) - len(new_records)
    logger.info(f"Generated {len(new_records)} valid pairs, skipped {n_skip}")

    with DATASET_PATH.open("a") as f:
        for rec in new_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_ticker = Counter(r["metadata"]["ticker"] for r in new_records)
    by_section = Counter(r["metadata"]["section"] for r in new_records)
    print(f"\nAppended {len(new_records)} pairs to {DATASET_PATH}")
    print("By ticker:", dict(sorted(by_ticker.items())))
    print("By section:", dict(sorted(by_section.items())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic eval dataset from chunks")
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to process")
    parser.add_argument("--dry-run", action="store_true", help="Show sampling plan only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    asyncio.run(main(limit=args.limit, dry_run=args.dry_run, seed=args.seed))
