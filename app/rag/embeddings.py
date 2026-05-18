"""
Dense + sparse embeddings via BAAI/bge-m3.

bge-m3 is a multilingual model that generates both vector types in a single
forward pass — no separate BM25 index needed for hybrid search.

Dense vector:  1024 floats — captures semantic meaning
Sparse vector: {token_id: weight} — captures exact keyword matches

Both are embedded from contextualized_text (context prefix + chunk text)
so the company/period context is baked into the vector representation.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 8
MAX_LENGTH = 512  # bge-m3 supports up to 8192, but 512 covers ~400 words


@dataclass
class EmbeddingResult:
    chunk_id: str
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "dense": self.dense,
            "sparse_indices": self.sparse_indices,
            "sparse_values": self.sparse_values,
        }


@lru_cache(maxsize=1)
def _load_model():
    """Load bge-m3 model once and cache it (download ~2GB on first run)."""
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(MODEL_NAME, use_fp16=True)


def embed_texts(texts: list[str]) -> list[dict]:
    """Embed a batch of texts, return dense + sparse vectors."""
    model = _load_model()
    output = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    return output


def embed_chunks(
    chunks: list[dict],
    output_path: Path,
    text_field: str = "contextualized_text",
) -> list[EmbeddingResult]:
    """Embed all chunks and save results to output_path as JSONL.

    Args:
        chunks: list of chunk dicts (from chunks.jsonl)
        output_path: where to save EmbeddingResult JSONL
        text_field: which field to embed (default: contextualized_text)
    """
    import json

    texts = [c[text_field] for c in chunks]
    ids = [c["id"] for c in chunks]

    results: list[EmbeddingResult] = []
    total = len(texts)

    # Process in batches for progress visibility
    for start in range(0, total, BATCH_SIZE):
        batch_texts = texts[start : start + BATCH_SIZE]
        batch_ids = ids[start : start + BATCH_SIZE]

        output = embed_texts(batch_texts)
        dense_vecs = output["dense_vecs"]
        sparse_vecs = output["lexical_weights"]

        for i, (chunk_id, dense, sparse) in enumerate(zip(batch_ids, dense_vecs, sparse_vecs)):
            # sparse is {token_id_str: weight} — convert to parallel lists
            indices = [int(k) for k in sparse]
            values = [float(sparse[k]) for k in sparse]

            results.append(
                EmbeddingResult(
                    chunk_id=chunk_id,
                    dense=dense.tolist(),
                    sparse_indices=indices,
                    sparse_values=values,
                )
            )

        done = min(start + BATCH_SIZE, total)
        print(f"  {done}/{total} chunks embedded", end="\r")

    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")

    return results
