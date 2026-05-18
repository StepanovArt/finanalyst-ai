"""
Unit tests for HybridRetriever.

Qdrant and ML models are mocked — no server or GPU required.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.rag.retrieval import HybridRetriever, SearchResult, _build_filter


# ---------------------------------------------------------------------------
# _build_filter
# ---------------------------------------------------------------------------


def test_build_filter_empty_returns_none() -> None:
    assert _build_filter({}) is None


def test_build_filter_single_field() -> None:
    f = _build_filter({"ticker": "AAPL"})
    assert f is not None
    assert len(f.must) == 1
    assert f.must[0].key == "ticker"
    assert f.must[0].match.value == "AAPL"


def test_build_filter_multiple_fields() -> None:
    f = _build_filter({"ticker": "MSFT", "year": 2024})
    assert f is not None
    assert len(f.must) == 2


# ---------------------------------------------------------------------------
# HybridRetriever.search — no reranking
# ---------------------------------------------------------------------------


def _make_point(idx: int, score: float, ticker: str = "AAPL") -> MagicMock:
    point = MagicMock()
    point.id = idx
    point.score = score
    point.payload = {
        "ticker": ticker,
        "company": "Apple Inc.",
        "filing_type": "10-K",
        "year": 2024,
        "quarter": "FY",
        "section": "MD&A",
        "text": f"chunk text {idx}",
        "context_prefix": f"prefix {idx}",
    }
    return point


@pytest.fixture()
def mock_qdrant_results() -> MagicMock:
    result = MagicMock()
    result.points = [_make_point(i, 1.0 - i * 0.1) for i in range(5)]
    return result


def _make_retriever(mock_qdrant_results: MagicMock) -> HybridRetriever:
    """Return a HybridRetriever with Qdrant and embed mocked out."""
    from qdrant_client.models import SparseVector

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._client = MagicMock()
    retriever._client.query_points.return_value = mock_qdrant_results
    # Mock _embed_query so we never touch the real bge-m3 model
    retriever._embed_query = MagicMock(  # type: ignore[method-assign]
        return_value=([0.1] * 1024, SparseVector(indices=[42], values=[0.5]))
    )
    return retriever


def test_search_returns_search_results(mock_qdrant_results: MagicMock) -> None:
    retriever = _make_retriever(mock_qdrant_results)

    results = retriever.search("What is Apple's revenue?", limit=3)

    assert len(results) == 5  # no rerank → raw Qdrant results (limit=3 passed to Qdrant)
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].ticker == "AAPL"
    assert results[0].year == 2024


def test_search_passes_filter_to_qdrant(mock_qdrant_results: MagicMock) -> None:
    retriever = _make_retriever(mock_qdrant_results)

    retriever.search("revenue", filters={"ticker": "AAPL"})

    call_kwargs = retriever._client.query_points.call_args.kwargs
    assert call_kwargs["prefetch"][0].filter is not None


# ---------------------------------------------------------------------------
# HybridRetriever.search — with reranking
# ---------------------------------------------------------------------------


def test_search_rerank_reorders_results(mock_qdrant_results: MagicMock) -> None:
    retriever = _make_retriever(mock_qdrant_results)

    # Reranker gives highest score to the last chunk (index 4)
    reranker_scores = [0.1, 0.2, 0.3, 0.4, 0.9]
    mock_reranker = MagicMock()
    mock_reranker.compute_score.return_value = reranker_scores

    with patch("app.rag.retrieval._get_reranker", return_value=mock_reranker):
        results = retriever.search("revenue", limit=3, rerank=True)

    # Should return top-3 after sorting by reranker score
    assert len(results) == 3
    assert results[0].score == pytest.approx(0.9)
    assert results[0].chunk_id == "4"  # point index 4 had score 0.9


def test_search_rerank_fetches_pool_not_limit(mock_qdrant_results: MagicMock) -> None:
    from app.rag.retrieval import RERANK_POOL

    retriever = _make_retriever(mock_qdrant_results)

    mock_reranker = MagicMock()
    mock_reranker.compute_score.return_value = [0.5] * 5

    with patch("app.rag.retrieval._get_reranker", return_value=mock_reranker):
        retriever.search("revenue", limit=2, rerank=True)

    call_kwargs = retriever._client.query_points.call_args.kwargs
    assert call_kwargs["limit"] == RERANK_POOL
