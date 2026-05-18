"""
Generate dense + sparse embeddings for all chunks via BAAI/bge-m3.

Usage:
    uv run --extra data python scripts/embed_chunks.py

Input:  data/processed/chunks.jsonl
Output: data/processed/embeddings.jsonl

Note: First run downloads ~2GB model from HuggingFace.
      CPU-only: ~20-40 min for 2092 chunks.
      MPS (Apple Silicon) or GPU: ~2-5 min.
"""

import json
import time
from pathlib import Path

from app.rag.embeddings import embed_chunks

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/embeddings.jsonl")


def main() -> None:
    print(f"Loading chunks from {CHUNKS_PATH}...")
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    print(f"  {len(chunks)} chunks loaded")

    print("\nLoading bge-m3 model (downloads ~2GB on first run)...")
    t0 = time.time()

    results = embed_chunks(chunks, EMBEDDINGS_PATH)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Saved {len(results)} embeddings → {EMBEDDINGS_PATH}")

    # Sanity check
    r = results[0]
    print(f"\nSample embedding:")
    print(f"  chunk_id:       {r.chunk_id}")
    print(f"  dense shape:    {len(r.dense)} floats")
    print(f"  sparse entries: {len(r.sparse_indices)} tokens")
    print(f"  dense[:3]:      {r.dense[:3]}")


if __name__ == "__main__":
    main()
