"""
Tests for RelevanceGrader agent.
"""

import json

import pytest

from app.agents.grader import ChunkGrade, GradeLabel, GradingResult, RelevanceGrader
from app.rag.retrieval import SearchResult
from tests.conftest import MockLLMProvider


def _grader(response: str) -> RelevanceGrader:
    return RelevanceGrader(llm=MockLLMProvider(response=response))


def _chunk(chunk_id: str, text: str = "some financial text") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=0.9,
        ticker="AAPL",
        company="Apple Inc.",
        filing_type="10-K",
        year=2024,
        quarter="FY",
        section="MD&A",
        text=text,
        context_prefix="prefix",
    )


def _grades_json(*labels: str) -> str:
    grades = [
        {"chunk_id": str(i), "label": label, "reason": f"reason {i}"}
        for i, label in enumerate(labels)
    ]
    return json.dumps({"grades": grades})


# ---------------------------------------------------------------------------
# GradingResult properties
# ---------------------------------------------------------------------------


def test_grading_result_relevant_count() -> None:
    grades = [
        ChunkGrade("1", GradeLabel.RELEVANT, ""),
        ChunkGrade("2", GradeLabel.IRRELEVANT, ""),
        ChunkGrade("3", GradeLabel.RELEVANT, ""),
    ]
    result = GradingResult(grades=grades)
    assert result.relevant_count == 2


def test_grading_result_has_sufficient_context_true() -> None:
    result = GradingResult(grades=[ChunkGrade("1", GradeLabel.RELEVANT, "")])
    assert result.has_sufficient_context is True


def test_grading_result_has_sufficient_context_false_when_empty() -> None:
    assert GradingResult(grades=[]).has_sufficient_context is False


def test_grading_result_has_sufficient_context_false_when_only_irrelevant() -> None:
    result = GradingResult(grades=[ChunkGrade("1", GradeLabel.IRRELEVANT, "")])
    assert result.has_sufficient_context is False


def test_grading_result_relevant_chunk_ids() -> None:
    grades = [
        ChunkGrade("1", GradeLabel.RELEVANT, ""),
        ChunkGrade("2", GradeLabel.IRRELEVANT, ""),
        ChunkGrade("3", GradeLabel.AMBIGUOUS, ""),
    ]
    result = GradingResult(grades=grades)
    assert result.relevant_chunk_ids() == ["1"]
    assert result.ambiguous_chunk_ids() == ["3"]


# ---------------------------------------------------------------------------
# RelevanceGrader.grade — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grade_empty_chunks_returns_empty() -> None:
    result = await _grader("{}").grade("query", [])
    assert result.grades == []


@pytest.mark.asyncio
async def test_grade_parses_labels_correctly() -> None:
    llm_response = _grades_json("relevant", "irrelevant", "ambiguous")
    chunks = [_chunk("0"), _chunk("1"), _chunk("2")]

    result = await _grader(llm_response).grade("Apple revenue 2024", chunks)

    assert len(result.grades) == 3
    assert result.grades[0].label == GradeLabel.RELEVANT
    assert result.grades[1].label == GradeLabel.IRRELEVANT
    assert result.grades[2].label == GradeLabel.AMBIGUOUS


@pytest.mark.asyncio
async def test_grade_has_sufficient_context_when_one_relevant() -> None:
    llm_response = _grades_json("relevant", "irrelevant")
    result = await _grader(llm_response).grade("query", [_chunk("0"), _chunk("1")])

    assert result.has_sufficient_context is True


@pytest.mark.asyncio
async def test_grade_unknown_label_becomes_ambiguous() -> None:
    raw = json.dumps({"grades": [{"chunk_id": "0", "label": "unknown_label", "reason": "?"}]})
    result = await _grader(raw).grade("query", [_chunk("0")])

    assert result.grades[0].label == GradeLabel.AMBIGUOUS


# ---------------------------------------------------------------------------
# RelevanceGrader.grade — fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grade_falls_back_on_invalid_json() -> None:
    chunks = [_chunk("0"), _chunk("1")]
    result = await _grader("not json").grade("query", chunks)

    assert len(result.grades) == 2
    assert all(g.label == GradeLabel.AMBIGUOUS for g in result.grades)


@pytest.mark.asyncio
async def test_grade_falls_back_on_missing_grades_key() -> None:
    chunks = [_chunk("0")]
    result = await _grader('{"wrong": []}').grade("query", chunks)

    # Empty grades list returned — no relevant chunks
    assert result.has_sufficient_context is False
