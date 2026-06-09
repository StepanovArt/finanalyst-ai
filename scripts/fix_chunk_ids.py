"""
One-off migration: re-assign globally-unique ids to an existing chunks.jsonl.

Background
----------
The original _make_id keyed only on (ticker, type, period, section, per-block
index). Sections sharing a canonical name (e.g. several "Income Statement" or
"Notes to FS" blocks) each restarted the index at 0, and the accession was not
part of the key — so ~39% of chunk ids collided. This did not corrupt the
Qdrant index (indexing uses positional point ids and does not store the chunk
id in the payload), but the chunk file itself had non-unique keys.

This migration recomputes ids using the fixed scheme (accession + document-global
sequence) WITHOUT re-parsing or re-chunking — every other field, including the
LLM/deterministic context_prefix, is preserved byte-for-byte. The sequence is
reconstructed per (ticker, accession) in file order, which matches the emission
order of chunk_document.

Usage:
    uv run python scripts/fix_chunk_ids.py --dry-run
    uv run python scripts/fix_chunk_ids.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from app.rag.chunking import _make_id

CHUNKS_PATH = Path("data/processed/chunks.jsonl")


def main(dry_run: bool) -> None:
    rows = [json.loads(line) for line in CHUNKS_PATH.read_text().splitlines() if line.strip()]
    before_unique = len({r["id"] for r in rows})

    seq_by_doc: dict[tuple[str, str], int] = defaultdict(int)
    new_ids: list[str] = []
    for r in rows:
        key = (r["ticker"], r["accession"])
        seq = seq_by_doc[key]
        seq_by_doc[key] += 1
        new_ids.append(
            _make_id(
                r["ticker"], r["period"], r["filing_type"], r["accession"], r["section"], seq
            )
        )

    after_unique = len(set(new_ids))
    dup_after = len(new_ids) - after_unique

    print(f"Total chunks:        {len(rows)}")
    print(f"Unique ids before:   {before_unique}  (collisions: {len(rows) - before_unique})")
    print(f"Unique ids after:    {after_unique}  (collisions: {dup_after})")

    if dup_after:
        # Should be 0; if not, report the offenders
        dupes = [i for i, c in Counter(new_ids).items() if c > 1]
        print(f"WARNING: {len(dupes)} residual collisions, e.g. {dupes[:3]}")

    if dry_run:
        print("\n(dry run — no file written)")
        return

    backup = CHUNKS_PATH.with_suffix(".jsonl.bak")
    shutil.copy(CHUNKS_PATH, backup)
    print(f"\nBackup written: {backup}")

    for r, new_id in zip(rows, new_ids):
        r["id"] = new_id

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Rewrote {CHUNKS_PATH} with unique ids.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
