"""
Validate Opus-authored QA pairs against raw chunk text and append to the dataset.

For each authored (chunk_id, question, ground_truth, anchor):
  - look up the source chunk
  - REQUIRE the anchor to be a verbatim (case-insensitive) substring of the raw
    chunk text — the same matching the retrieval eval uses. This guarantees every
    ground-truth context provably exists in the indexed corpus.
  - on pass: emit a dataset record with metadata copied from the chunk
  - on fail: report and skip (so a bad anchor never silently corrupts the eval)

Usage:
    uv run python scripts/append_authored_qa.py --dry-run   # validate only
    uv run python scripts/append_authored_qa.py             # validate + append
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from scripts._authored_qa import AUTHORED

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DATASET_PATH = Path("eval/dataset.jsonl")


def _load_chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS_PATH.read_text().splitlines() if line.strip()]


def _ticker_of(chunk_id: str) -> str:
    return chunk_id.split("_", 1)[0]


def _find_containing_chunk(chunks: list[dict], chunk_id: str, anchor: str) -> dict | None:
    """Find a chunk whose raw text contains the anchor.

    Semantic chunk IDs collide in chunks.jsonl (sections sharing a canonical name
    restart chunk_index at 0), so a dict keyed by id is unreliable. We instead
    search by content, scoped to the intended ticker, and prefer the exact id match
    when it actually contains the anchor.
    """
    ticker = _ticker_of(chunk_id)
    anchor_l = anchor.lower()

    # Prefer a chunk with the exact intended id that contains the anchor
    for c in chunks:
        if c["id"] == chunk_id and anchor_l in c["text"].lower():
            return c
    # Fall back to any same-ticker chunk containing the anchor
    for c in chunks:
        if c["ticker"] == ticker and anchor_l in c["text"].lower():
            return c
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


def main(dry_run: bool) -> None:
    chunks = _load_chunks()
    records: list[dict] = []
    failures: list[tuple[str, str]] = []
    next_num = _next_id_num()

    for chunk_id, question, ground_truth, anchor in AUTHORED:
        chunk = _find_containing_chunk(chunks, chunk_id, anchor)
        if chunk is None:
            ticker = _ticker_of(chunk_id)
            failures.append((chunk_id, f"anchor not found in any {ticker} chunk: {anchor!r}"))
            continue

        records.append(
            {
                "id": f"q{next_num:03d}",
                "question": question,
                "ground_truth": ground_truth,
                "contexts": [anchor],
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

    print(f"PASS: {len(records)}  FAIL: {len(failures)}  (of {len(AUTHORED)} authored)\n")
    if failures:
        print("Failures (skipped):")
        for cid, reason in failures:
            print(f"  - {cid}: {reason}")
        print()

    if dry_run:
        by_ticker = Counter(r["metadata"]["ticker"] for r in records)
        by_section = Counter(r["metadata"]["section"] for r in records)
        print("Would append (by ticker):", dict(sorted(by_ticker.items())))
        print("Would append (by section):", dict(sorted(by_section.items())))
        return

    with DATASET_PATH.open("a") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = sum(1 for line in DATASET_PATH.read_text().splitlines() if line.strip())
    print(f"Appended {len(records)} pairs. Dataset now has {total} QA pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not CHUNKS_PATH.exists():
        print(f"Chunks file not found: {CHUNKS_PATH}", file=sys.stderr)
        sys.exit(1)
    main(dry_run=args.dry_run)
