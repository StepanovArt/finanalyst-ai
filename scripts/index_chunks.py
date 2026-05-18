"""
Embed all chunks with bge-m3 and upload to Qdrant.

Processes chunks in batches: embed → upload → next batch.
No intermediate file needed — vectors go straight to Qdrant.

Usage:
    # Qdrant must be running:
    docker compose up qdrant -d

    uv run --extra data python scripts/index_chunks.py
    uv run --extra data python scripts/index_chunks.py --recreate  # wipe and reindex
"""

import argparse
import json
import time
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel
from qdrant_client.models import PointStruct, SparseVector

from app.rag.vector_store import (
    COLLECTION_NAME,
    create_collection,
    get_client,
)

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
BATCH_SIZE = 16
MAX_LENGTH = 512

# Payload fields stored in Qdrant (everything except the raw text body)
PAYLOAD_FIELDS = ["ticker", "company", "filing_type", "period", "year",
                  "quarter", "accession", "section", "currency", "chunk_index",
                  "context_prefix"]


def build_points(
    chunks: list[dict],
    chunk_offset: int,
    model: BGEM3FlagModel,
) -> list[PointStruct]:
    texts = [c["contextualized_text"] for c in chunks]

    output = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    points = []
    for i, chunk in enumerate(chunks):
        dense = output["dense_vecs"][i].tolist()
        sparse_raw = output["lexical_weights"][i]
        indices = [int(k) for k in sparse_raw]
        values = [float(sparse_raw[k]) for k in sparse_raw]

        payload = {f: chunk[f] for f in PAYLOAD_FIELDS if f in chunk}
        # Store original text (not contextualized) for display in results
        payload["text"] = chunk["text"]

        points.append(
            PointStruct(
                id=chunk_offset + i,
                vector={
                    "dense": dense,
                    "sparse": SparseVector(indices=indices, values=values),
                },
                payload=payload,
            )
        )
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="Wipe collection and reindex")
    args = parser.parse_args()

    print(f"Loading chunks from {CHUNKS_PATH}...")
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    print(f"  {len(chunks)} chunks")

    client = get_client()
    create_collection(client, recreate=args.recreate)

    print("\nLoading bge-m3 model...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    print(f"\nIndexing in batches of {BATCH_SIZE}...")
    t0 = time.time()
    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        points = build_points(batch, chunk_offset=start, model=model)
        client.upsert(collection_name=COLLECTION_NAME, points=points)

        done = min(start + BATCH_SIZE, total)
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  {done}/{total}  {elapsed:.0f}s elapsed  ETA {eta:.0f}s", end="\r")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    info = client.get_collection(COLLECTION_NAME)
    print(f"Collection points: {info.points_count}")


if __name__ == "__main__":
    main()
