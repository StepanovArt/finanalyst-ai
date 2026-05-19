"""
Tests for QueryDecomposer agent.

LLM is mocked — no API calls made.
"""

import pytest

from app.agents.decomposer import DecompositionResult, QueryDecomposer
from tests.conftest import MockLLMProvider


def _decomposer(response: str) -> QueryDecomposer:
    return QueryDecomposer(llm=MockLLMProvider(response=response))


# ---------------------------------------------------------------------------
# DecompositionResult
# ---------------------------------------------------------------------------


def test_decomposition_result_is_complex_when_multiple() -> None:
    result = DecompositionResult(sub_queries=["q1", "q2"])
    assert result.is_complex is True


def test_decomposition_result_not_complex_when_single() -> None:
    result = DecompositionResult(sub_queries=["q1"])
    assert result.is_complex is False


# ---------------------------------------------------------------------------
# QueryDecomposer.decompose — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decompose_simple_query_returns_single() -> None:
    llm_json = '{"sub_queries": ["What is Apple revenue Q1 2024?"]}'
    result = await _decomposer(llm_json).decompose("What is Apple revenue Q1 2024?")

    assert result.sub_queries == ["What is Apple revenue Q1 2024?"]
    assert result.is_complex is False


@pytest.mark.asyncio
async def test_decompose_complex_query_returns_multiple() -> None:
    llm_json = '{"sub_queries": ["Apple revenue 2024", "Microsoft revenue 2024"]}'
    result = await _decomposer(llm_json).decompose("Compare Apple and Microsoft revenue 2024")

    assert len(result.sub_queries) == 2
    assert "Apple revenue 2024" in result.sub_queries
    assert result.is_complex is True


@pytest.mark.asyncio
async def test_decompose_caps_at_five_sub_queries() -> None:
    import json

    six_queries = [f"query {i}" for i in range(6)]
    llm_json = json.dumps({"sub_queries": six_queries})
    result = await _decomposer(llm_json).decompose("very complex query")

    assert len(result.sub_queries) == 5


@pytest.mark.asyncio
async def test_decompose_strips_empty_strings() -> None:
    llm_json = '{"sub_queries": ["valid query", "", "  ", "another query"]}'
    result = await _decomposer(llm_json).decompose("some query")

    assert "" not in result.sub_queries
    assert len(result.sub_queries) == 2


# ---------------------------------------------------------------------------
# QueryDecomposer.decompose — fallback on bad LLM output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decompose_falls_back_on_invalid_json() -> None:
    result = await _decomposer("not json at all").decompose("What is Apple revenue?")

    assert result.sub_queries == ["What is Apple revenue?"]


@pytest.mark.asyncio
async def test_decompose_falls_back_on_missing_key() -> None:
    result = await _decomposer('{"wrong_key": []}').decompose("What is Apple revenue?")

    assert result.sub_queries == ["What is Apple revenue?"]


@pytest.mark.asyncio
async def test_decompose_falls_back_on_empty_list() -> None:
    result = await _decomposer('{"sub_queries": []}').decompose("What is Apple revenue?")

    assert result.sub_queries == ["What is Apple revenue?"]
