"""
Retrieval evaluation metrics: Recall@k and MRR.

Both metrics operate on text-based relevance rather than chunk IDs: a
retrieved chunk is considered relevant if it contains any ground-truth
context string as a substring (case-insensitive). This avoids coupling
the eval to ephemeral Qdrant point IDs and works with the dataset.jsonl
format where contexts are raw text passages.

Recall@k
--------
Fraction of questions for which at least one relevant chunk appears in
the top-k retrieved results. Standard for retrieval systems where a
single good chunk is sufficient to answer the question.

MRR (Mean Reciprocal Rank)
--------------------------
Average of 1/rank for the first relevant chunk. Rewards systems that
surface the best chunk early (rank 1 = 1.0, rank 2 = 0.5, rank 5 = 0.2).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    mrr: float
    k: int
    num_questions: int

    def __str__(self) -> str:
        return (
            f"Recall@{self.k}: {self.recall_at_k:.3f}  "
            f"MRR: {self.mrr:.3f}  "
            f"(n={self.num_questions})"
        )


def _is_relevant(chunk_text: str, ground_truth_contexts: list[str]) -> bool:
    """True if any ground-truth context appears (case-insensitive) in the chunk."""
    chunk_lower = chunk_text.lower()
    return any(ctx.lower() in chunk_lower for ctx in ground_truth_contexts)


def recall_at_k(
    retrieved_texts: list[str],
    ground_truth_contexts: list[str],
    k: int,
) -> float:
    """Return 1.0 if a relevant chunk is in the top-k, 0.0 otherwise."""
    for text in retrieved_texts[:k]:
        if _is_relevant(text, ground_truth_contexts):
            return 1.0
    return 0.0


def reciprocal_rank(
    retrieved_texts: list[str],
    ground_truth_contexts: list[str],
) -> float:
    """Return 1/rank of the first relevant chunk, or 0 if none found."""
    for rank, text in enumerate(retrieved_texts, start=1):
        if _is_relevant(text, ground_truth_contexts):
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics(
    results: list[tuple[list[str], list[str]]],
    k: int = 5,
) -> RetrievalMetrics:
    """Compute Recall@k and MRR across a list of (retrieved_texts, ground_truth_contexts) pairs.

    Args:
        results: each element is (retrieved_chunk_texts, ground_truth_context_strings)
        k: cutoff for Recall@k

    Returns:
        RetrievalMetrics with aggregated scores
    """
    if not results:
        return RetrievalMetrics(recall_at_k=0.0, mrr=0.0, k=k, num_questions=0)

    recall_scores = [recall_at_k(retrieved, gt, k) for retrieved, gt in results]
    rr_scores = [reciprocal_rank(retrieved, gt) for retrieved, gt in results]

    return RetrievalMetrics(
        recall_at_k=sum(recall_scores) / len(recall_scores),
        mrr=sum(rr_scores) / len(rr_scores),
        k=k,
        num_questions=len(results),
    )
