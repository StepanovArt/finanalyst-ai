"""
Background indexing service: parse → chunk → embed → upsert to Qdrant.

Job lifecycle: pending → processing → done | failed

Job state is stored in-memory. In a multi-process deployment this would
move to Redis, but a single-process server is sufficient here.
"""

import asyncio
import uuid
from dataclasses import dataclass
from enum import StrEnum

from loguru import logger

from app.rag.chunking import chunk_document
from app.rag.contextual import add_deterministic_context
from app.rag.ingestion import parse_filing_content
from app.rag.vector_store import COLLECTION_NAME, QDRANT_URL, get_client

MAX_CHUNK_ID_OFFSET = 10_000_000  # avoid colliding with batch-indexed points


class JobStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class IndexingJob:
    job_id: str
    ticker: str
    filing_type: str
    status: JobStatus = JobStatus.PENDING
    chunks_total: int = 0
    chunks_done: int = 0
    error: str = ""


class JobStore:
    """Thread-safe in-memory store for indexing jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, IndexingJob] = {}
        self._lock = asyncio.Lock()

    async def create(self, ticker: str, filing_type: str) -> str:
        job_id = str(uuid.uuid4())
        async with self._lock:
            self._jobs[job_id] = IndexingJob(job_id=job_id, ticker=ticker, filing_type=filing_type)
        return job_id

    async def get(self, job_id: str) -> IndexingJob | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update(self, job: IndexingJob) -> None:
        async with self._lock:
            self._jobs[job.job_id] = job


# Module-level singleton used by the router
job_store = JobStore()


async def run_indexing_job(
    job_id: str,
    raw_content: str,
    ticker: str,
    filing_type: str,
) -> None:
    """Parse, chunk, embed and upsert a filing. Updates job_store throughout."""
    job = await job_store.get(job_id)
    if job is None:
        return

    job.status = JobStatus.PROCESSING
    await job_store.update(job)

    try:
        # --- parse ---
        logger.info(f"[{job_id}] Parsing {ticker} {filing_type}")
        doc = parse_filing_content(raw_content, ticker=ticker, filing_type=filing_type)

        # --- chunk + context prefix ---
        chunks = chunk_document(doc)
        contextual = add_deterministic_context(chunks)

        job.chunks_total = len(contextual)
        await job_store.update(job)
        logger.info(f"[{job_id}] {len(contextual)} chunks ready, starting embedding")

        # --- embed + upsert in a thread (CPU-bound) ---
        await asyncio.get_event_loop().run_in_executor(None, _embed_and_upsert, job_id, contextual)

        job.chunks_done = job.chunks_total
        job.status = JobStatus.DONE
        await job_store.update(job)
        logger.info(f"[{job_id}] Done — {job.chunks_total} chunks indexed")

    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error = str(exc)
        await job_store.update(job)
        logger.error(f"[{job_id}] Failed: {exc}")


def _embed_and_upsert(job_id: str, contextual: list) -> None:
    """CPU-bound: embed all chunks and upsert to Qdrant. Runs in thread pool."""
    from FlagEmbedding import BGEM3FlagModel
    from qdrant_client.models import PointStruct, SparseVector

    BATCH_SIZE = 16
    MAX_LENGTH = 512

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    client = get_client(QDRANT_URL)

    # Use a UUID-based offset so uploaded points don't collide with batch-indexed ones
    import hashlib

    id_offset = int(hashlib.md5(job_id.encode()).hexdigest(), 16) % MAX_CHUNK_ID_OFFSET

    for start in range(0, len(contextual), BATCH_SIZE):
        batch = contextual[start : start + BATCH_SIZE]
        texts = [c.contextualized_text for c in batch]

        output = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        points = []
        for i, ctx_chunk in enumerate(batch):
            chunk = ctx_chunk.chunk
            dense = output["dense_vecs"][i].tolist()
            sparse_raw = output["lexical_weights"][i]

            payload = {
                "ticker": chunk.ticker,
                "company": chunk.company,
                "filing_type": chunk.filing_type,
                "period": chunk.period,
                "year": chunk.year,
                "quarter": chunk.quarter,
                "accession": chunk.accession,
                "section": chunk.section,
                "currency": chunk.currency,
                "chunk_index": chunk.chunk_index,
                "context_prefix": ctx_chunk.context_prefix,
                "text": chunk.text,
            }

            points.append(
                PointStruct(
                    id=id_offset + start + i,
                    vector={
                        "dense": dense,
                        "sparse": SparseVector(
                            indices=[int(k) for k in sparse_raw],
                            values=[float(sparse_raw[k]) for k in sparse_raw],
                        ),
                    },
                    payload=payload,
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.debug(f"[{job_id}] Upserted {start + len(batch)}/{len(contextual)}")
