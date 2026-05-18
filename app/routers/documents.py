"""
Document upload and indexing endpoints.

POST /documents/upload  — upload a full-submission.txt file, start background indexing
GET  /documents/status/{job_id} — poll indexing progress
"""

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.indexing import JobStatus, IndexingJob, job_store, run_indexing_job

router = APIRouter(prefix="/documents", tags=["documents"])

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB — EDGAR filings are typically 5-20 MB


class UploadResponse(BaseModel):
    job_id: str
    message: str


class StatusResponse(BaseModel):
    job_id: str
    ticker: str
    filing_type: str
    status: JobStatus
    chunks_total: int
    chunks_done: int
    error: str


def _job_to_response(job: IndexingJob) -> StatusResponse:
    return StatusResponse(
        job_id=job.job_id,
        ticker=job.ticker,
        filing_type=job.filing_type,
        status=job.status,
        chunks_total=job.chunks_total,
        chunks_done=job.chunks_done,
        error=job.error,
    )


@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    ticker: str = Form(..., description="Company ticker symbol, e.g. AAPL"),
    filing_type: str = Form(..., description="Filing type: 10-K or 10-Q"),
) -> UploadResponse:
    """Upload a SEC EDGAR full-submission.txt file and index it in the background.

    Returns a job_id immediately. Poll /documents/status/{job_id} for progress.
    """
    if filing_type not in ("10-K", "10-Q"):
        raise HTTPException(status_code=422, detail="filing_type must be '10-K' or '10-Q'")

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")

    raw_text = content.decode("utf-8", errors="replace")

    job_id = await job_store.create(ticker=ticker.upper(), filing_type=filing_type)
    background_tasks.add_task(run_indexing_job, job_id, raw_text, ticker, filing_type)

    return UploadResponse(
        job_id=job_id,
        message=f"Indexing started for {ticker.upper()} {filing_type}. Poll /documents/status/{job_id}",
    )


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str) -> StatusResponse:
    """Return the current status of an indexing job."""
    job = await job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _job_to_response(job)
