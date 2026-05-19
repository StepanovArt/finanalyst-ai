"""
Tests for HallucinationChecker agent.
"""

import json

import pytest

from app.agents.hallucination_checker import (
    GroundednessLabel,
    HallucinationChecker,
    HallucinationCheckResult,
)
from app.rag.retrieval import SearchResult
from tests.conftest import MockLLMProvider


def _checker(response: str) -> HallucinationChecker:
    return HallucinationChecker(llm=MockLLMProvider(response=response))


def _chunk(chunk_id: str = "1") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=0.9,
        ticker="AAPL",
        company="Apple Inc.",
        filing_type="10-K",
        year=2024,
        quarter="FY",
        section="MD&A",
        text="Net sales were $391 billion for fiscal year 2024.",
        context_prefix="prefix",
    )


def _response(label: str, reason: str = "ok") -> str:
    return json.dumps({"label": label, "reason": reason})


# ---------------------------------------------------------------------------
# HallucinationCheckResult properties
# ---------------------------------------------------------------------------


def test_is_acceptable_true_for_grounded() -> None:
    r = HallucinationCheckResult(GroundednessLabel.GROUNDED, "all facts supported")
    assert r.is_acceptable is True


def test_is_acceptable_true_for_partially_grounded() -> None:
    r = HallucinationCheckResult(GroundednessLabel.PARTIALLY_GROUNDED, "minor gaps")
    assert r.is_acceptable is True


def test_is_acceptable_false_for_hallucinated() -> None:
    r = HallucinationCheckResult(GroundednessLabel.HALLUCINATED, "invented numbers")
    assert r.is_acceptable is False


# ---------------------------------------------------------------------------
# HallucinationChecker.check — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_grounded_answer() -> None:
    result = await _checker(_response("grounded", "all facts present")).check(
        "Apple revenue was $391B in FY2024.", [_chunk()]
    )
    assert result.label == GroundednessLabel.GROUNDED
    assert result.is_acceptable is True
    assert result.reason == "all facts present"


@pytest.mark.asyncio
async def test_check_hallucinated_answer() -> None:
    result = await _checker(_response("hallucinated", "number not in context")).check(
        "Apple revenue was $500B.", [_chunk()]
    )
    assert result.label == GroundednessLabel.HALLUCINATED
    assert result.is_acceptable is False


@pytest.mark.asyncio
async def test_check_partially_grounded() -> None:
    result = await _checker(_response("partially_grounded", "minor inference")).check(
        "Apple had strong revenue.", [_chunk()]
    )
    assert result.label == GroundednessLabel.PARTIALLY_GROUNDED
    assert result.is_acceptable is True


# ---------------------------------------------------------------------------
# HallucinationChecker.check — edge cases and fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_empty_answer_returns_hallucinated() -> None:
    result = await _checker("{}").check("", [_chunk()])
    assert result.label == GroundednessLabel.HALLUCINATED


@pytest.mark.asyncio
async def test_check_falls_back_on_invalid_json() -> None:
    result = await _checker("not json").check("some answer", [_chunk()])
    assert result.label == GroundednessLabel.PARTIALLY_GROUNDED


@pytest.mark.asyncio
async def test_check_falls_back_on_unknown_label() -> None:
    result = await _checker('{"label": "unknown", "reason": "?"}').check("some answer", [_chunk()])
    assert result.label == GroundednessLabel.PARTIALLY_GROUNDED


@pytest.mark.asyncio
async def test_check_works_with_empty_chunks() -> None:
    result = await _checker(_response("grounded")).check("No data found.", [])
    assert result.label == GroundednessLabel.GROUNDED
