"""
Agentic pipeline metrics.

Measures the self-correction loop behaviour and full-pipeline costs:

  rewrites_pct    — % of queries that triggered at least one rewrite
  avg_iterations  — mean number of retrieval attempts per query
  avg_latency_s   — mean wall-clock latency of graph.run() in seconds
  avg_cost_usd    — estimated LLM cost per query based on token counts

These are collected at runtime by wrapping each graph.run() call with a
timer and by reading SelfCorrectionResult.rewrites_triggered from the
graph internals. For cost estimation we use rough GPT-4o-mini token
prices (updated 2025-01: $0.15/1M input, $0.60/1M output).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

_INPUT_COST_PER_TOKEN = 0.15 / 1_000_000  # GPT-4o-mini input
_OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000  # GPT-4o-mini output


@dataclass
class QueryRecord:
    """Data captured for a single eval query run."""

    latency_s: float
    iterations: int  # SelfCorrectionResult.total_attempts
    rewrites: int  # SelfCorrectionResult.rewrites_triggered
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AgenticMetrics:
    rewrites_pct: float
    avg_iterations: float
    avg_latency_s: float
    avg_cost_usd: float
    num_questions: int
    records: list[QueryRecord] = field(default_factory=list, repr=False)

    def __str__(self) -> str:
        return (
            f"Rewrites: {self.rewrites_pct:.1%}  "
            f"Avg iterations: {self.avg_iterations:.2f}  "
            f"Avg latency: {self.avg_latency_s:.2f}s  "
            f"Avg cost: ${self.avg_cost_usd:.4f}  "
            f"(n={self.num_questions})"
        )


def compute_agentic_metrics(records: list[QueryRecord]) -> AgenticMetrics:
    """Aggregate per-query records into agentic metrics.

    Args:
        records: one QueryRecord per evaluated question

    Returns:
        AgenticMetrics with aggregated statistics
    """
    if not records:
        return AgenticMetrics(
            rewrites_pct=0.0,
            avg_iterations=0.0,
            avg_latency_s=0.0,
            avg_cost_usd=0.0,
            num_questions=0,
        )

    n = len(records)
    rewrites_pct = sum(1 for r in records if r.rewrites > 0) / n
    avg_iterations = sum(r.iterations for r in records) / n
    avg_latency_s = sum(r.latency_s for r in records) / n
    avg_cost_usd = (
        sum(
            r.input_tokens * _INPUT_COST_PER_TOKEN + r.output_tokens * _OUTPUT_COST_PER_TOKEN
            for r in records
        )
        / n
    )

    return AgenticMetrics(
        rewrites_pct=rewrites_pct,
        avg_iterations=avg_iterations,
        avg_latency_s=avg_latency_s,
        avg_cost_usd=avg_cost_usd,
        num_questions=n,
        records=records,
    )


class LatencyTimer:
    """Context manager that records elapsed wall-clock time."""

    def __init__(self) -> None:
        self.elapsed_s: float = 0.0

    def __enter__(self) -> LatencyTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_s = time.perf_counter() - self._start
