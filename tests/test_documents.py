"""
Tests for /documents/upload and /documents/status endpoints.

Indexing pipeline (parse/embed/upsert) is mocked — no Qdrant or ML models needed.
"""

import io
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.indexing import JobStatus, job_store


@pytest.fixture(autouse=True)
def reset_job_store() -> None:
    """Wipe job store between tests."""
    job_store._jobs.clear()


# ---------------------------------------------------------------------------
# POST /documents/upload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_returns_202_with_job_id() -> None:
    fake_file = io.BytesIO(b"<SGML>fake content</SGML>")

    with patch("app.routers.documents.run_indexing_job", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/documents/upload",
                data={"ticker": "AAPL", "filing_type": "10-K"},
                files={"file": ("full-submission.txt", fake_file, "text/plain")},
            )

    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert "AAPL" in body["message"]


@pytest.mark.asyncio
async def test_upload_invalid_filing_type_returns_422() -> None:
    fake_file = io.BytesIO(b"content")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post(
            "/documents/upload",
            data={"ticker": "AAPL", "filing_type": "8-K"},
            files={"file": ("f.txt", fake_file, "text/plain")},
        )

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_upload_file_too_large_returns_413() -> None:
    big_file = io.BytesIO(b"x" * (51 * 1024 * 1024))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post(
            "/documents/upload",
            data={"ticker": "AAPL", "filing_type": "10-K"},
            files={"file": ("big.txt", big_file, "text/plain")},
        )

    assert resp.status_code == 413


# ---------------------------------------------------------------------------
# GET /documents/status/{job_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_not_found_returns_404() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/documents/status/nonexistent-id")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_status_returns_job_state() -> None:
    job_id = await job_store.create(ticker="MSFT", filing_type="10-Q")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get(f"/documents/status/{job_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job_id
    assert body["ticker"] == "MSFT"
    assert body["status"] == JobStatus.PENDING


# ---------------------------------------------------------------------------
# JobStore unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_job_store_create_and_get() -> None:
    job_id = await job_store.create(ticker="NVDA", filing_type="10-K")
    job = await job_store.get(job_id)

    assert job is not None
    assert job.ticker == "NVDA"
    assert job.status == JobStatus.PENDING


@pytest.mark.asyncio
async def test_job_store_update() -> None:
    job_id = await job_store.create(ticker="META", filing_type="10-Q")
    job = await job_store.get(job_id)
    assert job is not None

    job.status = JobStatus.DONE
    job.chunks_total = 42
    job.chunks_done = 42
    await job_store.update(job)

    updated = await job_store.get(job_id)
    assert updated is not None
    assert updated.status == JobStatus.DONE
    assert updated.chunks_done == 42


@pytest.mark.asyncio
async def test_job_store_returns_none_for_missing() -> None:
    result = await job_store.get("does-not-exist")
    assert result is None
