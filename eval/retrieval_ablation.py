"""
Retrieval ablation: dense-only vs sparse-only vs hybrid RRF (+ optional rerank).

Isolates the contribution of each retrieval signal on the same 53-question set,
so the "was hybrid search worth it?" question is answered with data rather than
assumption. Unlike eval/run_eval.py (which always instantiates the full
HybridRetriever), this queries Qdrant directly per-mode.

Requires: Qdrant up with the `sec_filings` collection, and transformers==4.56.x
(see README reproducibility note).

Usage:
    uv run python -m eval.retrieval_ablation [--k 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qdrant_client.models import Fusion, FusionQuery, Prefetch

from app.rag.retrieval import HybridRetriever
from app.rag.vector_store import COLLECTION_NAME
from eval.metrics.retrieval import compute_retrieval_metrics

DATASET = Path(__file__).parent / "dataset.jsonl"
PREFETCH_LIMIT = 20


def main(k: int) -> None:
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(rows)} questions")

    retriever = HybridRetriever()
    client = retriever._client

    def _texts(points: object) -> list[str]:
        return [p.payload.get("text", "") for p in points.points]  # type: ignore[union-attr]

    def dense(vec: list[float]) -> list[str]:
        res = client.query_points(
            collection_name=COLLECTION_NAME, query=vec, using="dense", limit=k, with_payload=True
        )
        return _texts(res)

    def sparse(vec: object) -> list[str]:
        res = client.query_points(
            collection_name=COLLECTION_NAME, query=vec, using="sparse", limit=k, with_payload=True
        )
        return _texts(res)

    def hybrid(d: list[float], s: object) -> list[str]:
        res = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=d, using="dense", limit=PREFETCH_LIMIT),
                Prefetch(query=s, using="sparse", limit=PREFETCH_LIMIT),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
            with_payload=True,
        )
        return _texts(res)

    modes: dict[str, list[tuple[list[str], list[str]]]] = {
        "Dense-only": [],
        "Sparse-only": [],
        "Hybrid RRF": [],
        "Hybrid + rerank": [],
    }
    for s in rows:
        d, sp = retriever._embed_query(s["question"])
        modes["Dense-only"].append((dense(d), s["contexts"]))
        modes["Sparse-only"].append((sparse(sp), s["contexts"]))
        modes["Hybrid RRF"].append((hybrid(d, sp), s["contexts"]))
        hits = retriever.search(s["question"], limit=k, rerank=True)
        modes["Hybrid + rerank"].append(([h.text for h in hits], s["contexts"]))

    print(f"\n| {'Retrieval':<18} | Recall@{k} |  MRR  |")
    print(f"|{'-' * 20}|{'-' * 10}|{'-' * 7}|")
    for name, pairs in modes.items():
        m = compute_retrieval_metrics(pairs, k=k)
        print(f"| {name:<18} | {m.recall_at_k:>8.3f} | {m.mrr:>5.3f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    main(k=args.k)
