"""
Tests for POST /agent/query endpoint.

RAGAgentGraph is mocked via dependency_overrides so no LLM or Qdrant calls
are made. Tests verify HTTP behaviour, schema mapping, and error paths.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.agents.dependencies import get_rag_graph
from app.agents.graph import AgentResponse
from app.agents.hallucination_checker import GroundednessLabel
from app.agents.synthesizer import Citation
from app.main import app

_TRACE_ID = "test-trace-id-1234"


def _citation() -> Citation:
    return Citation(
        chunk_id="c1",
        quote="Net sales were $391B.",
        company="Apple Inc.",
        filing_type="10-K",
        year=2024,
        quarter="FY",
        section="MD&A",
    )


def _make_graph(
    answer: str = "Apple revenue was $391B.",
    citations: list[Citation] | None = None,
    sub_queries: list[str] | None = None,
    is_hallucinated: bool = False,
    groundedness: GroundednessLabel = GroundednessLabel.GROUNDED,
) -> MagicMock:
    response = AgentResponse(
        query="What is Apple revenue?",
        answer=answer,
        citations=citations or [],
        sub_queries=sub_queries or ["Apple revenue 2024"],
        is_hallucinated=is_hallucinated,
        groundedness=groundedness,
        trace_id=_TRACE_ID,
    )
    graph = MagicMock()
    graph.run = AsyncMock(return_value=response)
    return graph


@pytest.fixture
async def client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture(autouse=True)
def override_graph():
    app.dependency_overrides[get_rag_graph] = lambda: _make_graph()
    yield
    app.dependency_overrides.pop(get_rag_graph, None)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_returns_200(client: AsyncClient) -> None:
    resp = await client.post("/agent/query", json={"query": "What is Apple revenue?"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_query_response_schema(client: AsyncClient) -> None:
    resp = await client.post("/agent/query", json={"query": "What is Apple revenue?"})
    data = resp.json()
    assert data["answer"] == "Apple revenue was $391B."
    assert data["trace_id"] == _TRACE_ID
    assert isinstance(data["sub_queries"], list)
    assert isinstance(data["citations"], list)
    assert "is_hallucinated" in data
    assert "groundedness" in data


@pytest.mark.asyncio
async def test_query_with_citations(client: AsyncClient) -> None:
    app.dependency_overrides[get_rag_graph] = lambda: _make_graph(citations=[_citation()])
    resp = await client.post("/agent/query", json={"query": "Apple revenue?"})
    data = resp.json()
    assert len(data["citations"]) == 1
    c = data["citations"][0]
    assert c["chunk_id"] == "c1"
    assert c["company"] == "Apple Inc."
    assert c["year"] == 2024


@pytest.mark.asyncio
async def test_query_passes_filters_to_graph(client: AsyncClient) -> None:
    mock_graph = _make_graph()
    app.dependency_overrides[get_rag_graph] = lambda: mock_graph

    await client.post(
        "/agent/query",
        json={"query": "Apple revenue?", "filters": {"ticker": "AAPL", "year": 2024}},
    )

    call_kwargs = mock_graph.run.call_args.kwargs
    assert call_kwargs["filters"] == {"ticker": "AAPL", "year": 2024}


@pytest.mark.asyncio
async def test_query_passes_session_and_user_id(client: AsyncClient) -> None:
    mock_graph = _make_graph()
    app.dependency_overrides[get_rag_graph] = lambda: mock_graph

    await client.post(
        "/agent/query",
        json={"query": "Apple?", "session_id": "sess-1", "user_id": "user-42"},
    )

    call_kwargs = mock_graph.run.call_args.kwargs
    assert call_kwargs["session_id"] == "sess-1"
    assert call_kwargs["user_id"] == "user-42"


@pytest.mark.asyncio
async def test_hallucinated_answer_flagged_in_response(client: AsyncClient) -> None:
    app.dependency_overrides[get_rag_graph] = lambda: _make_graph(
        is_hallucinated=True,
        groundedness=GroundednessLabel.HALLUCINATED,
    )
    resp = await client.post("/agent/query", json={"query": "Apple?"})
    data = resp.json()
    assert data["is_hallucinated"] is True
    assert data["groundedness"] == "hallucinated"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_empty_string_returns_422(client: AsyncClient) -> None:
    resp = await client.post("/agent/query", json={"query": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_query_missing_field_returns_422(client: AsyncClient) -> None:
    resp = await client.post("/agent/query", json={})
    assert resp.status_code == 422
