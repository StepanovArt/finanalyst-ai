"""
Generation evaluation via RAGAS.

Wraps RAGAS evaluate() to compute four metrics for a set of QA pairs:

  Faithfulness       — are all claims in the answer grounded in the contexts?
  Answer Relevancy   — does the answer address the question?
  Context Precision  — are the retrieved contexts ranked by relevance?
  Context Recall     — does the retrieved context cover the ground truth?

Usage (requires: uv run --extra eval python -m eval.run_eval):

    from eval.metrics.generation import compute_ragas_metrics

    scores = await compute_ragas_metrics([
        {"question": "...", "answer": "...", "contexts": ["..."], "ground_truth": "..."},
        ...
    ])
    print(scores)

RAGAS expects contexts as list[str] per question (all retrieved chunk texts).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationMetrics:
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None
    num_questions: int

    def __str__(self) -> str:
        def _fmt(v: float | None) -> str:
            return f"{v:.3f}" if v is not None else "N/A"

        return (
            f"Faithfulness: {_fmt(self.faithfulness)}  "
            f"Answer Relevancy: {_fmt(self.answer_relevancy)}  "
            f"Context Precision: {_fmt(self.context_precision)}  "
            f"Context Recall: {_fmt(self.context_recall)}  "
            f"(n={self.num_questions})"
        )


def compute_ragas_metrics(
    samples: list[dict],
) -> GenerationMetrics:
    """Evaluate generation quality using RAGAS.

    Args:
        samples: list of dicts with keys:
            question (str), answer (str), contexts (list[str]), ground_truth (str)

    Returns:
        GenerationMetrics — None values for individual metrics on RAGAS error
    """
    try:
        from datasets import Dataset  # type: ignore[import-untyped]
        from ragas import evaluate  # type: ignore[import-untyped]
        from ragas.metrics import (  # type: ignore[import-untyped]
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise ImportError(
            "RAGAS and datasets are required for generation eval. "
            "Install with: uv sync --extra eval"
        ) from exc

    dataset = Dataset.from_list(samples)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    scores = result.to_pandas().mean(numeric_only=True)

    return GenerationMetrics(
        faithfulness=float(scores.get("faithfulness", float("nan"))) or None,
        answer_relevancy=float(scores.get("answer_relevancy", float("nan"))) or None,
        context_precision=float(scores.get("context_precision", float("nan"))) or None,
        context_recall=float(scores.get("context_recall", float("nan"))) or None,
        num_questions=len(samples),
    )
