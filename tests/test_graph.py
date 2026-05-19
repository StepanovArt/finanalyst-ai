"""
Tests for RAGAgentGraph (LangGraph orchestration).

All agents and the retriever are mocked — tests verify graph wiring and
state flow, not individual agent behaviour.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.decomposer import DecompositionResult
from app.agents.graph import AgentResponse, RAGAgentGraph
from app.agents.hallucination_checker import GroundednessLabel, HallucinationCheckResult
from app.agents.self_correction import SelfCorrectionResult
from app.agents.synthesizer import SynthesisResult
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


def _make_graph(
    sub_queries: list[str] = None,
    chunks: list[SearchResult] = None,
    answer: str = "Apple revenue was $391B.",
    citations: list = None,
    groundedness: GroundednessLabel = GroundednessLabel.GROUNDED,
) -> RAGAgentGraph:
    sub_queries = sub_queries or ["Apple revenue 2024"]
    chunks = chunks or [_chunk("1"), _chunk("2")]
    citations = citations or []

    decomposer = MagicMock()
    decomposer.decompose = AsyncMock(return_value=DecompositionResult(sub_queries=sub_queries))

    loop = MagicMock()
    loop.run = AsyncMock(
        return_value=SelfCorrectionResult(
            final_chunks=chunks,
            final_query=sub_queries[0],
        )
    )

    synthesizer = MagicMock()
    synthesizer.synthesize = AsyncMock(
        return_value=SynthesisResult(answer=answer, citations=citations)
    )

    checker = MagicMock()
    checker.check = AsyncMock(
        return_value=HallucinationCheckResult(
            label=groundedness,
            reason="verified",
        )
    )

    return RAGAgentGraph(
        decomposer=decomposer,
        correction_loop=loop,
        synthesizer=synthesizer,
        checker=checker,
    )


# ---------------------------------------------------------------------------
# Graph execution — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_run_returns_agent_response() -> None:
    graph = _make_graph()
    result = await graph.run("What is Apple revenue?")

    assert isinstance(result, AgentResponse)
    assert result.answer == "Apple revenue was $391B."
    assert result.query == "What is Apple revenue?"


@pytest.mark.asyncio
async def test_graph_propagates_sub_queries() -> None:
    graph = _make_graph(sub_queries=["Apple revenue", "Apple margins"])
    result = await graph.run("Compare Apple revenue and margins")

    assert result.sub_queries == ["Apple revenue", "Apple margins"]


@pytest.mark.asyncio
async def test_graph_grounded_answer_not_flagged() -> None:
    graph = _make_graph(groundedness=GroundednessLabel.GROUNDED)
    result = await graph.run("query")

    assert result.is_hallucinated is False
    assert result.groundedness == GroundednessLabel.GROUNDED


@pytest.mark.asyncio
async def test_graph_hallucinated_answer_is_flagged() -> None:
    graph = _make_graph(groundedness=GroundednessLabel.HALLUCINATED)
    result = await graph.run("query")

    assert result.is_hallucinated is True
    assert result.groundedness == GroundednessLabel.HALLUCINATED


@pytest.mark.asyncio
async def test_graph_deduplicates_chunks_across_sub_queries() -> None:
    # Both sub-queries return the same chunk — should appear once
    shared_chunk = _chunk("shared")

    decomposer = MagicMock()
    decomposer.decompose = AsyncMock(return_value=DecompositionResult(sub_queries=["q1", "q2"]))
    loop = MagicMock()
    loop.run = AsyncMock(
        return_value=SelfCorrectionResult(final_chunks=[shared_chunk], final_query="q")
    )
    synthesizer = MagicMock()
    synthesizer.synthesize = AsyncMock(return_value=SynthesisResult(answer="ok", citations=[]))
    checker = MagicMock()
    checker.check = AsyncMock(
        return_value=HallucinationCheckResult(GroundednessLabel.GROUNDED, "ok")
    )

    graph = RAGAgentGraph(
        decomposer=decomposer,
        correction_loop=loop,
        synthesizer=synthesizer,
        checker=checker,
    )
    await graph.run("query")

    # synthesize called with 1 unique chunk, not 2
    call_args = synthesizer.synthesize.call_args
    chunks_passed = call_args.args[1]
    assert len(chunks_passed) == 1


@pytest.mark.asyncio
async def test_graph_passes_filters_to_retrieval_loop() -> None:
    graph = _make_graph()
    await graph.run("Apple revenue", filters={"ticker": "AAPL", "year": 2024})

    loop_call = graph._loop.run.call_args
    assert loop_call.kwargs.get("filters") == {"ticker": "AAPL", "year": 2024}
