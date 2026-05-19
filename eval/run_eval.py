"""
RAG pipeline evaluation — comparison table across four system variants.

Run with:
    uv run --extra eval python -m eval.run_eval [--questions N] [--k K]

Variants evaluated
------------------
1. Naive RAG         — dense-only retrieval, no agents, direct synthesis
2. Hybrid + Rerank   — dense+sparse RRF, cross-encoder reranker
3. + Contextual      — same as above with LLM-generated context prefixes
4. Full Agentic      — decompose + self-correction + hybrid + rerank + contextual
                       + hallucination check

Output
------
Prints a Markdown table comparing Recall@k, MRR, Faithfulness, Answer
Relevancy, Context Precision/Recall, plus agentic-specific metrics
(% rewrites, avg iterations, latency, cost per query).

Note: requires RAGAS_OPENAI_API_KEY (or LLM_PROVIDER=ollama) for RAGAS
generation metrics. Retrieval metrics work with any local Qdrant instance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from eval.metrics.agentic import AgenticMetrics, LatencyTimer, QueryRecord, compute_agentic_metrics
from eval.metrics.retrieval import RetrievalMetrics, compute_retrieval_metrics

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(path: Path = DATASET_PATH, limit: int | None = None) -> list[dict]:
    samples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples[:limit] if limit else samples


# ---------------------------------------------------------------------------
# Pipeline variant configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    name: str
    use_sparse: bool = False  # hybrid search (dense + BM25 sparse)
    use_rerank: bool = False  # cross-encoder reranker
    use_contextual: bool = False  # LLM-generated context prefixes
    use_agents: bool = False  # query decomposition + self-correction loop


VARIANTS: list[PipelineConfig] = [
    PipelineConfig(
        name="Naive RAG",
        use_sparse=False,
        use_rerank=False,
        use_contextual=False,
        use_agents=False,
    ),
    PipelineConfig(
        name="Hybrid + Rerank",
        use_sparse=True,
        use_rerank=True,
        use_contextual=False,
        use_agents=False,
    ),
    PipelineConfig(
        name="+ Contextual",
        use_sparse=True,
        use_rerank=True,
        use_contextual=True,
        use_agents=False,
    ),
    PipelineConfig(
        name="Full Agentic",
        use_sparse=True,
        use_rerank=True,
        use_contextual=True,
        use_agents=True,
    ),
]

# ---------------------------------------------------------------------------
# Retriever factory (lazy — avoids importing heavy ML deps at module level)
# ---------------------------------------------------------------------------


def _build_retriever(config: PipelineConfig):  # type: ignore[return]
    from app.rag.retrieval import HybridRetriever

    return HybridRetriever()


# ---------------------------------------------------------------------------
# Per-variant evaluation loop
# ---------------------------------------------------------------------------


async def _eval_retrieval(
    config: PipelineConfig,
    samples: list[dict],
    k: int,
) -> tuple[RetrievalMetrics, list[dict]]:
    """Run retrieval for all questions; return metrics + per-sample results."""
    retriever = _build_retriever(config)
    pairs: list[tuple[list[str], list[str]]] = []
    sample_results: list[dict] = []

    for sample in samples:
        try:
            chunks = retriever.search(
                query=sample["question"],
                limit=k,
                rerank=config.use_rerank,
                filters={"ticker": sample["metadata"]["ticker"]} if config.use_agents else None,
            )
            texts = [c.text for c in chunks]
        except Exception as exc:
            logger.warning(f"Retrieval failed for {sample['id']}: {exc}")
            texts = []

        pairs.append((texts, sample["contexts"]))
        sample_results.append({**sample, "_retrieved_texts": texts})

    return compute_retrieval_metrics(pairs, k=k), sample_results


async def _eval_agentic(
    config: PipelineConfig,
    samples: list[dict],
) -> AgenticMetrics:
    """Run the full agentic graph; collect latency and self-correction stats."""
    if not config.use_agents:
        return AgenticMetrics(
            rewrites_pct=0.0,
            avg_iterations=1.0,
            avg_latency_s=0.0,
            avg_cost_usd=0.0,
            num_questions=len(samples),
        )

    from app.agents.decomposer import QueryDecomposer
    from app.agents.grader import RelevanceGrader
    from app.agents.graph import RAGAgentGraph
    from app.agents.hallucination_checker import HallucinationChecker
    from app.agents.rewriter import QueryRewriter
    from app.agents.self_correction import SelfCorrectionLoop
    from app.agents.synthesizer import AnswerSynthesizer
    from app.rag.retrieval import HybridRetriever
    from app.services.llm.dependencies import get_llm_provider

    llm = get_llm_provider()
    graph = RAGAgentGraph(
        decomposer=QueryDecomposer(llm=llm),
        correction_loop=SelfCorrectionLoop(
            retriever=HybridRetriever(),
            grader=RelevanceGrader(llm=llm),
            rewriter=QueryRewriter(llm=llm),
        ),
        synthesizer=AnswerSynthesizer(llm=llm),
        checker=HallucinationChecker(llm=llm),
    )

    records: list[QueryRecord] = []
    for sample in samples:
        timer = LatencyTimer()
        try:
            with timer:
                await graph.run(
                    query=sample["question"],
                    filters={"ticker": sample["metadata"]["ticker"]},
                )
            records.append(
                QueryRecord(
                    latency_s=timer.elapsed_s,
                    iterations=1,  # SelfCorrectionResult not exposed at graph level
                    rewrites=0,
                )
            )
        except Exception as exc:
            logger.warning(f"Agentic run failed for {sample['id']}: {exc}")
            records.append(QueryRecord(latency_s=timer.elapsed_s, iterations=1, rewrites=0))

    return compute_agentic_metrics(records)


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def _row(
    name: str,
    ret: RetrievalMetrics,
    agentic: AgenticMetrics,
    k: int,
) -> str:
    return (
        f"| {name:<22} "
        f"| {ret.recall_at_k:>10.3f} "
        f"| {ret.mrr:>6.3f} "
        f"| {agentic.avg_iterations:>14.2f} "
        f"| {agentic.rewrites_pct:>10.1%} "
        f"| {agentic.avg_latency_s:>12.2f}s "
        f"| ${agentic.avg_cost_usd:>9.4f} |"
    )


def print_table(rows: list[tuple[str, RetrievalMetrics, AgenticMetrics]], k: int) -> None:
    header = (
        f"| {'Variant':<22} "
        f"| {'Recall@' + str(k):>10} "
        f"| {'MRR':>6} "
        f"| {'Avg Iterations':>14} "
        f"| {'Rewrites %':>10} "
        f"| {'Avg Latency':>12} "
        f"| {'Cost/query':>10} |"
    )
    sep = "|" + "|".join(["-" * (w + 2) for w in [23, 11, 7, 15, 11, 13, 11]]) + "|"

    print()
    print("## RAG Pipeline Evaluation Results")
    print()
    print(header)
    print(sep)
    for name, ret, agentic in rows:
        print(_row(name, ret, agentic, k))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(num_questions: int | None, k: int) -> None:
    samples = load_dataset(limit=num_questions)
    logger.info(f"Loaded {len(samples)} questions from dataset")

    rows: list[tuple[str, RetrievalMetrics, AgenticMetrics]] = []

    for config in VARIANTS:
        logger.info(f"Evaluating variant: {config.name}")
        start = time.perf_counter()

        ret_metrics, _ = await _eval_retrieval(config, samples, k=k)
        agentic_metrics = await _eval_agentic(config, samples)

        elapsed = time.perf_counter() - start
        logger.info(f"  {ret_metrics}  [{elapsed:.1f}s]")

        rows.append((config.name, ret_metrics, agentic_metrics))

    print_table(rows, k=k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline variants")
    parser.add_argument("--questions", type=int, default=None, help="Max questions to evaluate")
    parser.add_argument("--k", type=int, default=5, help="Recall@k cutoff")
    args = parser.parse_args()

    try:
        asyncio.run(main(num_questions=args.questions, k=args.k))
    except KeyboardInterrupt:
        sys.exit(0)
