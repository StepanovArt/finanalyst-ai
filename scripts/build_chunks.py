"""
Parse all EDGAR filings, chunk them, add context, save to chunks.jsonl.

Usage:
    uv run --extra data python scripts/build_chunks.py           # deterministic context
    uv run --extra data python scripts/build_chunks.py --llm     # LLM context via Ollama
"""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from app.rag.chunking import chunk_document
from app.rag.contextual import add_deterministic_context, add_llm_context
from app.rag.ingestion import parse_all_filings

FILINGS_DIR = Path("data/filings/sec-edgar-filings")
OUTPUT_PATH = Path("data/processed/chunks.jsonl")


async def main(use_llm: bool) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Parsing filings...")
    docs = parse_all_filings(FILINGS_DIR)
    print(f"  Parsed {len(docs)} documents")

    print("Chunking...")
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    print(f"  {len(all_chunks)} chunks total")

    print(f"Adding context ({'LLM via Ollama' if use_llm else 'deterministic'})...")
    if use_llm:
        contextual = await add_llm_context(all_chunks)
    else:
        contextual = add_deterministic_context(all_chunks)

    print(f"Saving → {OUTPUT_PATH}")
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for c in contextual:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    # Summary
    section_counts: Counter[str] = Counter(c.chunk.section for c in contextual)
    print("\nChunks by section:")
    for section, count in section_counts.most_common():
        print(f"  {section:<30} {count:>5}")

    print("\nSample context prefix:")
    sample = contextual[0]
    print(f"  {sample.context_prefix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="Use Ollama to generate context")
    args = parser.parse_args()
    asyncio.run(main(use_llm=args.llm))
