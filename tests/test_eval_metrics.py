"""
Unit tests for eval/metrics — retrieval and agentic metrics only.

RAGAS (generation.py) requires heavy ML deps and is tested manually;
retrieval.py and agentic.py are pure Python and fully unit-testable.
"""

import time

import pytest

from eval.metrics.agentic import (
    AgenticMetrics,
    LatencyTimer,
    QueryRecord,
    compute_agentic_metrics,
)
from eval.metrics.retrieval import (
    RetrievalMetrics,
    compute_retrieval_metrics,
    recall_at_k,
    reciprocal_rank,
)

# ---------------------------------------------------------------------------
# Retrieval — recall_at_k
# ---------------------------------------------------------------------------


def test_recall_at_k_hit_at_first_position() -> None:
    retrieved = ["Net sales were $391 billion.", "Other text."]
    contexts = ["net sales were $391"]
    assert recall_at_k(retrieved, contexts, k=5) == 1.0


def test_recall_at_k_hit_beyond_cutoff_returns_zero() -> None:
    retrieved = ["irrelevant chunk", "another irrelevant", "Net sales were $391 billion."]
    contexts = ["Net sales were $391"]
    assert recall_at_k(retrieved, contexts, k=2) == 0.0


def test_recall_at_k_no_match_returns_zero() -> None:
    retrieved = ["chunk a", "chunk b"]
    contexts = ["not present anywhere"]
    assert recall_at_k(retrieved, contexts, k=5) == 0.0


def test_recall_at_k_empty_retrieved_returns_zero() -> None:
    assert recall_at_k([], ["some context"], k=5) == 0.0


def test_recall_at_k_case_insensitive() -> None:
    retrieved = ["NET SALES WERE $391 BILLION"]
    contexts = ["net sales were $391"]
    assert recall_at_k(retrieved, contexts, k=1) == 1.0


def test_recall_at_k_multiple_contexts_any_match() -> None:
    retrieved = ["operating income was $123B"]
    contexts = ["not here", "operating income was $123"]
    assert recall_at_k(retrieved, contexts, k=1) == 1.0


# ---------------------------------------------------------------------------
# Retrieval — reciprocal_rank
# ---------------------------------------------------------------------------


def test_reciprocal_rank_first_position() -> None:
    retrieved = ["revenue $391B", "other"]
    assert reciprocal_rank(retrieved, ["revenue $391"]) == 1.0


def test_reciprocal_rank_second_position() -> None:
    retrieved = ["irrelevant", "revenue $391B"]
    assert reciprocal_rank(retrieved, ["revenue $391"]) == pytest.approx(0.5)


def test_reciprocal_rank_no_match() -> None:
    retrieved = ["a", "b", "c"]
    assert reciprocal_rank(retrieved, ["xyz"]) == 0.0


# ---------------------------------------------------------------------------
# Retrieval — compute_retrieval_metrics
# ---------------------------------------------------------------------------


def test_compute_retrieval_metrics_all_hits() -> None:
    results = [
        (["Net sales $391B", "other"], ["net sales $391"]),
        (["Operating income $123B"], ["operating income $123"]),
    ]
    m = compute_retrieval_metrics(results, k=5)
    assert m.recall_at_k == 1.0
    assert m.mrr == 1.0
    assert m.num_questions == 2


def test_compute_retrieval_metrics_no_hits() -> None:
    results = [
        (["irrelevant"], ["not found"]),
        (["also irrelevant"], ["missing"]),
    ]
    m = compute_retrieval_metrics(results, k=5)
    assert m.recall_at_k == 0.0
    assert m.mrr == 0.0


def test_compute_retrieval_metrics_partial() -> None:
    results = [
        (["Revenue $391B"], ["revenue $391"]),  # hit
        (["irrelevant"], ["not here"]),  # miss
    ]
    m = compute_retrieval_metrics(results, k=5)
    assert m.recall_at_k == pytest.approx(0.5)
    assert m.mrr == pytest.approx(0.5)


def test_compute_retrieval_metrics_empty_input() -> None:
    m = compute_retrieval_metrics([], k=5)
    assert m.recall_at_k == 0.0
    assert m.num_questions == 0


def test_compute_retrieval_metrics_str_representation() -> None:
    m = RetrievalMetrics(recall_at_k=0.8, mrr=0.65, k=5, num_questions=15)
    s = str(m)
    assert "0.800" in s
    assert "0.650" in s


# ---------------------------------------------------------------------------
# Agentic — compute_agentic_metrics
# ---------------------------------------------------------------------------


def test_agentic_metrics_empty_records() -> None:
    m = compute_agentic_metrics([])
    assert m.num_questions == 0
    assert m.rewrites_pct == 0.0


def test_agentic_metrics_no_rewrites() -> None:
    records = [
        QueryRecord(latency_s=1.0, iterations=1, rewrites=0),
        QueryRecord(latency_s=2.0, iterations=1, rewrites=0),
    ]
    m = compute_agentic_metrics(records)
    assert m.rewrites_pct == 0.0
    assert m.avg_iterations == 1.0
    assert m.avg_latency_s == pytest.approx(1.5)


def test_agentic_metrics_some_rewrites() -> None:
    records = [
        QueryRecord(latency_s=1.0, iterations=1, rewrites=0),
        QueryRecord(latency_s=3.0, iterations=3, rewrites=2),
        QueryRecord(latency_s=2.0, iterations=2, rewrites=1),
    ]
    m = compute_agentic_metrics(records)
    assert m.rewrites_pct == pytest.approx(2 / 3)
    assert m.avg_iterations == pytest.approx(2.0)


def test_agentic_metrics_all_rewrites() -> None:
    records = [QueryRecord(latency_s=2.0, iterations=2, rewrites=1) for _ in range(5)]
    m = compute_agentic_metrics(records)
    assert m.rewrites_pct == 1.0
    assert m.num_questions == 5


def test_agentic_metrics_cost_computed() -> None:
    records = [
        QueryRecord(latency_s=1.0, iterations=1, rewrites=0, input_tokens=1000, output_tokens=200)
    ]
    m = compute_agentic_metrics(records)
    # $0.15/1M input + $0.60/1M output for 1000+200 tokens
    expected = 1000 * 0.15 / 1_000_000 + 200 * 0.60 / 1_000_000
    assert m.avg_cost_usd == pytest.approx(expected)


def test_agentic_metrics_str_representation() -> None:
    m = AgenticMetrics(
        rewrites_pct=0.4,
        avg_iterations=1.8,
        avg_latency_s=2.5,
        avg_cost_usd=0.0012,
        num_questions=10,
    )
    s = str(m)
    assert "40.0%" in s
    assert "2.50s" in s


# ---------------------------------------------------------------------------
# LatencyTimer
# ---------------------------------------------------------------------------


def test_latency_timer_measures_elapsed() -> None:
    timer = LatencyTimer()
    with timer:
        time.sleep(0.05)
    assert timer.elapsed_s >= 0.04
    assert timer.elapsed_s < 0.5
