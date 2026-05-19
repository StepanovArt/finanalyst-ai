"""
Tests for SelfCorrectionLoop.

Retriever, grader, and rewriter are all mocked.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.grader import ChunkGrade, GradeLabel, GradingResult
from app.agents.rewriter import RewriteResult
from app.agents.self_correction import SelfCorrectionLoop, SelfCorrectionResult
from app.rag.retrieval import SearchResult


def _chunk(chunk_id: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=0.9,
        ticker="AAPL",
        company="Apple Inc.",
        filing_type="10-K",
        year=2024,
        quarter="FY",
        section="MD&A",
        text="revenue text",
        context_prefix="prefix",
    )


def _sufficient_grading(chunk_ids: list[str]) -> GradingResult:
    return GradingResult(
        grades=[ChunkGrade(cid, GradeLabel.RELEVANT, "relevant") for cid in chunk_ids]
    )


def _insufficient_grading(chunk_ids: list[str]) -> GradingResult:
    return GradingResult(
        grades=[ChunkGrade(cid, GradeLabel.IRRELEVANT, "not relevant") for cid in chunk_ids]
    )


def _make_loop(
    chunks_per_attempt: list[list[SearchResult]],
    gradings: list[GradingResult],
    rewrite_response: str = "rewritten query",
    max_iterations: int = 3,
) -> SelfCorrectionLoop:
    retriever = MagicMock()
    retriever.search = MagicMock(side_effect=chunks_per_attempt)

    grader = MagicMock()
    grader.grade = AsyncMock(side_effect=gradings)

    rewriter = MagicMock()
    rewriter.rewrite = AsyncMock(
        return_value=RewriteResult(
            original_query="original",
            rewritten_query=rewrite_response,
            synonyms_added=[],
        )
    )

    return SelfCorrectionLoop(
        retriever=retriever,
        grader=grader,
        rewriter=rewriter,
        max_iterations=max_iterations,
    )


# ---------------------------------------------------------------------------
# Happy path: sufficient on first attempt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stops_on_first_attempt_when_sufficient() -> None:
    chunks = [_chunk("1"), _chunk("2")]
    grading = _sufficient_grading(["1", "2"])

    loop = _make_loop(
        chunks_per_attempt=[chunks],
        gradings=[grading],
    )
    result = await loop.run("Apple revenue 2024")

    assert result.total_attempts == 1
    assert result.succeeded is True
    assert result.rewrites_triggered == 0
    assert result.final_chunks == chunks


# ---------------------------------------------------------------------------
# Self-correction: insufficient → rewrite → sufficient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rewrites_once_when_first_attempt_insufficient() -> None:
    chunks1 = [_chunk("1")]
    chunks2 = [_chunk("2"), _chunk("3")]

    loop = _make_loop(
        chunks_per_attempt=[chunks1, chunks2],
        gradings=[_insufficient_grading(["1"]), _sufficient_grading(["2", "3"])],
        rewrite_response="Apple total revenue fiscal year 2024",
    )
    result = await loop.run("Apple revenue")

    assert result.total_attempts == 2
    assert result.rewrites_triggered == 1
    assert result.succeeded is True
    assert result.final_query == "Apple total revenue fiscal year 2024"


# ---------------------------------------------------------------------------
# Max iterations: never finds sufficient, returns last result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_last_result_after_max_iterations() -> None:
    chunks = [[_chunk(str(i))] for i in range(3)]
    gradings = [_insufficient_grading([str(i)]) for i in range(3)]

    loop = _make_loop(
        chunks_per_attempt=chunks,
        gradings=gradings,
        max_iterations=3,
    )
    result = await loop.run("obscure query")

    assert result.total_attempts == 3
    assert result.succeeded is False
    assert result.rewrites_triggered == 2


# ---------------------------------------------------------------------------
# SelfCorrectionResult properties
# ---------------------------------------------------------------------------


def test_result_succeeded_false_when_no_iterations() -> None:
    result = SelfCorrectionResult(final_chunks=[], final_query="q")
    assert result.succeeded is False


def test_result_total_attempts_matches_iterations_length() -> None:
    from app.agents.self_correction import IterationResult

    grading = _insufficient_grading([])
    iters = [IterationResult(attempt=i, query="q", chunks=[], grading=grading) for i in range(1, 3)]
    result = SelfCorrectionResult(final_chunks=[], final_query="q", iterations=iters)
    assert result.total_attempts == 2
    assert result.rewrites_triggered == 1
