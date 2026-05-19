"""
Tests for AnswerSynthesizer agent.
"""

import json

import pytest

from app.agents.synthesizer import AnswerSynthesizer, SynthesisResult
from app.rag.retrieval import SearchResult
from tests.conftest import MockLLMProvider


def _synthesizer(response: str) -> AnswerSynthesizer:
    return AnswerSynthesizer(llm=MockLLMProvider(response=response))


def _chunk(chunk_id: str, text: str = "Revenue was $100B in FY2024.") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=0.95,
        ticker="AAPL",
        company="Apple Inc.",
        filing_type="10-K",
        year=2024,
        quarter="FY",
        section="MD&A",
        text=text,
        context_prefix="prefix",
    )


def _ok_response(answer: str, chunk_id: str = "1", quote: str = "Revenue was $100B") -> str:
    return json.dumps({
        "answer": answer,
        "citations": [{"chunk_id": chunk_id, "quote": quote}],
    })


# ---------------------------------------------------------------------------
# SynthesisResult properties
# ---------------------------------------------------------------------------


def test_synthesis_result_has_citations_true() -> None:
    from app.agents.synthesizer import Citation
    c = Citation("1", "quote", "Apple Inc.", "10-K", 2024, "FY", "MD&A")
    result = SynthesisResult(answer="answer", citations=[c])
    assert result.has_citations is True


def test_synthesis_result_has_citations_false_when_empty() -> None:
    result = SynthesisResult(answer="answer", citations=[])
    assert result.has_citations is False


# ---------------------------------------------------------------------------
# AnswerSynthesizer.synthesize — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_returns_answer_and_citation() -> None:
    chunks = [_chunk("1")]
    result = await _synthesizer(_ok_response("Apple revenue was $100B.", "1")).synthesize(
        "What is Apple revenue?", chunks
    )

    assert result.answer == "Apple revenue was $100B."
    assert result.has_citations is True
    assert result.citations[0].chunk_id == "1"
    assert result.citations[0].company == "Apple Inc."
    assert result.citations[0].year == 2024


@pytest.mark.asyncio
async def test_synthesize_skips_unknown_chunk_id_in_citations() -> None:
    chunks = [_chunk("1")]
    response = json.dumps({
        "answer": "Some answer.",
        "citations": [
            {"chunk_id": "1", "quote": "valid"},
            {"chunk_id": "999", "quote": "unknown chunk"},
        ],
    })
    result = await _synthesizer(response).synthesize("query", chunks)

    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "1"


@pytest.mark.asyncio
async def test_synthesize_no_citations_in_response() -> None:
    response = json.dumps({"answer": "No data found.", "citations": []})
    result = await _synthesizer(response).synthesize("query", [_chunk("1")])

    assert result.answer == "No data found."
    assert result.has_citations is False


@pytest.mark.asyncio
async def test_synthesize_truncates_long_quotes() -> None:
    long_quote = "x" * 400
    response = json.dumps({
        "answer": "answer",
        "citations": [{"chunk_id": "1", "quote": long_quote}],
    })
    result = await _synthesizer(response).synthesize("query", [_chunk("1")])

    assert len(result.citations[0].quote) <= 300


# ---------------------------------------------------------------------------
# AnswerSynthesizer.synthesize — fallback cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_empty_chunks_returns_no_info_message() -> None:
    result = await _synthesizer("{}").synthesize("query", [])

    assert "No relevant information" in result.answer
    assert result.has_citations is False


@pytest.mark.asyncio
async def test_synthesize_falls_back_on_invalid_json() -> None:
    result = await _synthesizer("not json").synthesize("query", [_chunk("1")])

    assert "Unable to synthesize" in result.answer
    assert result.has_citations is False


@pytest.mark.asyncio
async def test_synthesize_falls_back_on_empty_answer() -> None:
    response = json.dumps({"answer": "", "citations": []})
    result = await _synthesizer(response).synthesize("query", [_chunk("1")])

    assert "Unable to synthesize" in result.answer
