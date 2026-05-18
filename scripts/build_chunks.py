"""
Parse all EDGAR filings and save chunks to data/processed/chunks.jsonl.

Usage:
    uv run --extra data python scripts/build_chunks.py
"""

import json
from collections import Counter
from pathlib import Path

from app.rag.chunking import chunk_document
from app.rag.ingestion import parse_all_filings

FILINGS_DIR = Path("data/filings/sec-edgar-filings")
OUTPUT_PATH = Path("data/processed/chunks.jsonl")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Parsing filings...")
    docs = parse_all_filings(FILINGS_DIR)
    print(f"  Parsed {len(docs)} documents")

    print("Chunking...")
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc.ticker} {doc.filing_type} {doc.period}: {len(chunks)} chunks")

    print(f"\nSaving {len(all_chunks)} chunks → {OUTPUT_PATH}")
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    # Summary
    section_counts: Counter[str] = Counter(c.section for c in all_chunks)
    print("\nChunks by section:")
    for section, count in section_counts.most_common():
        print(f"  {section:<30} {count:>5}")


if __name__ == "__main__":
    main()
