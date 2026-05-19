"""
Self-Correction Loop.

Orchestrates retrieval → grading → rewriting into a retry loop.
Stops early when grader reports sufficient context; falls back to the
best result seen so far after max_iterations regardless.

Iteration flow:
  attempt 1: retrieve(original_query) → grade → done if sufficient
  attempt 2: rewrite(query, attempt=1) → retrieve → grade → done if sufficient
  attempt 3: rewrite(query, attempt=2) → retrieve → grade → done regardless
"""

from dataclasses import dataclass, field

from loguru import logger

from app.agents.grader import GradingResult, RelevanceGrader
from app.agents.rewriter import QueryRewriter
from app.rag.retrieval import HybridRetriever, SearchResult

DEFAULT_MAX_ITERATIONS = 3
DEFAULT_RETRIEVE_LIMIT = 5


@dataclass
class IterationResult:
    attempt: int
    query: str
    chunks: list[SearchResult]
    grading: GradingResult


@dataclass
class SelfCorrectionResult:
    final_chunks: list[SearchResult]
    final_query: str
    iterations: list[IterationResult] = field(default_factory=list)

    @property
    def total_attempts(self) -> int:
        return len(self.iterations)

    @property
    def succeeded(self) -> bool:
        """True if at least one retrieved chunk was graded relevant."""
        if not self.iterations:
            return False
        return self.iterations[-1].grading.has_sufficient_context

    @property
    def rewrites_triggered(self) -> int:
        return max(0, self.total_attempts - 1)


class SelfCorrectionLoop:
    """Retrieval loop with automatic query rewriting on low-relevance results."""

    def __init__(
        self,
        retriever: HybridRetriever,
        grader: RelevanceGrader,
        rewriter: QueryRewriter,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        retrieve_limit: int = DEFAULT_RETRIEVE_LIMIT,
    ) -> None:
        self._retriever = retriever
        self._grader = grader
        self._rewriter = rewriter
        self._max_iterations = max_iterations
        self._retrieve_limit = retrieve_limit

    async def run(
        self,
        query: str,
        filters: dict | None = None,
        rerank: bool = False,
    ) -> SelfCorrectionResult:
        """Run the self-correction retrieval loop.

        Args:
            query: original user question
            filters: optional Qdrant metadata filters
            rerank: whether to apply cross-encoder reranking on each attempt

        Returns:
            SelfCorrectionResult with best chunks and full iteration history
        """
        current_query = query
        iterations: list[IterationResult] = []

        for attempt in range(1, self._max_iterations + 1):
            logger.info(
                f"Self-correction attempt {attempt}/{self._max_iterations}: '{current_query}'"
            )

            chunks = self._retriever.search(
                current_query,
                limit=self._retrieve_limit,
                filters=filters,
                rerank=rerank,
            )

            grading = await self._grader.grade(current_query, chunks)
            iterations.append(
                IterationResult(
                    attempt=attempt,
                    query=current_query,
                    chunks=chunks,
                    grading=grading,
                )
            )

            if grading.has_sufficient_context:
                logger.info(
                    f"Sufficient context found at attempt {attempt} "
                    f"({grading.relevant_count} relevant chunks)"
                )
                break

            if attempt < self._max_iterations:
                rewrite = await self._rewriter.rewrite(current_query, attempt=attempt)
                current_query = rewrite.rewritten_query
                logger.info(f"Rewriting query for attempt {attempt + 1}: '{current_query}'")

        # Use the last iteration's chunks (best available, even if grading failed)
        last = iterations[-1]
        return SelfCorrectionResult(
            final_chunks=last.chunks,
            final_query=last.query,
            iterations=iterations,
        )
