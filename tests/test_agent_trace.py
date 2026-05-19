"""
Tests for GET /agent/trace/{trace_id} endpoint.

Langfuse client is mocked at the tracing module level so no real Langfuse
server is required. Tests cover: not configured, happy path, not found.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

_TRACE_ID = "abc-123"


@pytest.fixture
async def client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def _make_observation(
    name: str = "decompose",
    obs_type: str = "SPAN",
    latency: float = 0.42,
    inp: str = "query",
    out: str = "sub-query",
) -> SimpleNamespace:
    return SimpleNamespace(name=name, type=obs_type, latency=latency, input=inp, output=out)


def _make_trace(
    observations: list | None = None,
    latency: float = 1.23,
    inp: str = "What is Apple revenue?",
    out: str = "Revenue was $391B.",
    html_path: str = f"/trace/{_TRACE_ID}",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=_TRACE_ID,
        input=inp,
        output=out,
        latency=latency,
        htmlPath=html_path,
        observations=observations or [],
    )


# ---------------------------------------------------------------------------
# Not configured
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_returns_503_when_langfuse_not_configured(client: AsyncClient) -> None:
    with patch("app.routers.agent.get_langfuse_client", return_value=None):
        resp = await client.get(f"/agent/trace/{_TRACE_ID}")
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_returns_200_with_trace_data(client: AsyncClient) -> None:
    mock_client = MagicMock()
    mock_client.get_trace.return_value = _make_trace()

    with patch("app.routers.agent.get_langfuse_client", return_value=mock_client):
        resp = await client.get(f"/agent/trace/{_TRACE_ID}")

    assert resp.status_code == 200
    data = resp.json()
    assert data["trace_id"] == _TRACE_ID
    assert data["input"] == "What is Apple revenue?"
    assert data["output"] == "Revenue was $391B."
    assert data["latency_ms"] == 1230.0


@pytest.mark.asyncio
async def test_trace_returns_langfuse_url(client: AsyncClient) -> None:
    mock_client = MagicMock()
    mock_client.get_trace.return_value = _make_trace(html_path=f"/trace/{_TRACE_ID}")

    with patch("app.routers.agent.get_langfuse_client", return_value=mock_client):
        resp = await client.get(f"/agent/trace/{_TRACE_ID}")

    url = resp.json()["langfuse_url"]
    assert url.startswith("http")
    assert _TRACE_ID in url


@pytest.mark.asyncio
async def test_trace_observations_mapped_correctly(client: AsyncClient) -> None:
    obs = [
        _make_observation("decompose", "SPAN", 0.1, "q", "sub-q"),
        _make_observation("retrieve", "SPAN", 0.5, "sub-q", "3 chunks"),
        _make_observation("synthesize", "GENERATION", 0.8, "sub-q + chunks", "answer"),
        _make_observation("check", "SPAN", 0.05, "answer", "grounded"),
    ]
    mock_client = MagicMock()
    mock_client.get_trace.return_value = _make_trace(observations=obs)

    with patch("app.routers.agent.get_langfuse_client", return_value=mock_client):
        resp = await client.get(f"/agent/trace/{_TRACE_ID}")

    observations = resp.json()["observations"]
    assert len(observations) == 4
    assert observations[0]["name"] == "decompose"
    assert observations[2]["type"] == "GENERATION"
    assert observations[1]["latency_ms"] == 500.0


@pytest.mark.asyncio
async def test_trace_with_no_observations(client: AsyncClient) -> None:
    mock_client = MagicMock()
    mock_client.get_trace.return_value = _make_trace(observations=[])

    with patch("app.routers.agent.get_langfuse_client", return_value=mock_client):
        resp = await client.get(f"/agent/trace/{_TRACE_ID}")

    assert resp.status_code == 200
    assert resp.json()["observations"] == []


# ---------------------------------------------------------------------------
# Not found
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_returns_404_when_not_found(client: AsyncClient) -> None:
    mock_client = MagicMock()
    mock_client.get_trace.side_effect = Exception("404 Not Found")

    with patch("app.routers.agent.get_langfuse_client", return_value=mock_client):
        resp = await client.get("/agent/trace/nonexistent-id")

    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()
