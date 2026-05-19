"""
Tests for QueryRewriter agent.
"""

import pytest

from app.agents.rewriter import QueryRewriter, RewriteResult
from tests.conftest import MockLLMProvider


def _rewriter(response: str) -> QueryRewriter:
    return QueryRewriter(llm=MockLLMProvider(response=response))


# ---------------------------------------------------------------------------
# RewriteResult
# ---------------------------------------------------------------------------


def test_rewrite_result_changed_when_different() -> None:
    r = RewriteResult(original_query="FCF", rewritten_query="free cash flow", synonyms_added=[])
    assert r.changed is True


def test_rewrite_result_not_changed_when_same() -> None:
    r = RewriteResult(original_query="q", rewritten_query="q", synonyms_added=[])
    assert r.changed is False


# ---------------------------------------------------------------------------
# QueryRewriter.rewrite — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rewrite_returns_llm_response() -> None:
    result = await _rewriter("free cash flow reported by Apple").rewrite("Apple FCF")

    assert result.rewritten_query == "free cash flow reported by Apple"
    assert result.original_query == "Apple FCF"
    assert result.changed is True


@pytest.mark.asyncio
async def test_rewrite_adds_glossary_synonyms() -> None:
    result = await _rewriter("rewritten query").rewrite("What is Apple FCF?")

    # FCF is in glossary — synonyms_added should be non-empty
    assert len(result.synonyms_added) > 0
    assert any("free cash flow" in s for s in result.synonyms_added)


@pytest.mark.asyncio
async def test_rewrite_no_synonyms_for_unknown_term() -> None:
    result = await _rewriter("rewritten").rewrite("some unknown term xyz")

    assert result.synonyms_added == []


@pytest.mark.asyncio
async def test_rewrite_strips_surrounding_quotes() -> None:
    result = await _rewriter('"rewritten query with quotes"').rewrite("original")

    assert result.rewritten_query == "rewritten query with quotes"


@pytest.mark.asyncio
async def test_rewrite_accepts_attempt_parameter() -> None:
    result = await _rewriter("aggressive rewrite").rewrite("original query", attempt=2)

    assert result.rewritten_query == "aggressive rewrite"


# ---------------------------------------------------------------------------
# QueryRewriter.rewrite — fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rewrite_falls_back_on_empty_response() -> None:
    result = await _rewriter("").rewrite("Apple revenue")

    assert result.rewritten_query == "Apple revenue"
    assert result.changed is False


@pytest.mark.asyncio
async def test_rewrite_falls_back_on_whitespace_response() -> None:
    result = await _rewriter("   ").rewrite("Apple revenue")

    assert result.rewritten_query == "Apple revenue"
