"""
Agent endpoints.

POST /agent/query     — runs the full agentic RAG pipeline
GET  /agent/trace/{id} — fetches Langfuse trace details by trace_id
"""

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.agents.dependencies import RAGAgentGraph, get_rag_graph
from app.core.config import settings
from app.core.tracing import get_langfuse_client

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    filters: dict | None = None
    session_id: str | None = None
    user_id: str | None = None


class CitationOut(BaseModel):
    chunk_id: str
    quote: str
    company: str
    filing_type: str
    year: int
    quarter: str
    section: str


class AgentQueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationOut]
    sub_queries: list[str]
    is_hallucinated: bool
    groundedness: str
    trace_id: str


class ObservationOut(BaseModel):
    name: str
    type: str
    latency_ms: float | None
    input: str | None
    output: str | None


class TraceResponse(BaseModel):
    trace_id: str
    input: str | None
    output: str | None
    latency_ms: float | None
    observations: list[ObservationOut]
    langfuse_url: str


@router.get("/trace/{trace_id}", response_model=TraceResponse)
async def get_trace(trace_id: str) -> TraceResponse:
    """Fetch a Langfuse trace by its ID.

    Returns the pipeline execution timeline: per-node spans with latency and
    I/O, total latency, and a direct link to the Langfuse UI.

    Raises 503 when Langfuse is not configured; 404 when trace not found.
    """
    client = get_langfuse_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Langfuse is not configured.")

    try:
        trace = client.get_trace(trace_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id!r} not found.")

    def _obs(o: object) -> ObservationOut:
        latency = getattr(o, "latency", None)
        return ObservationOut(
            name=str(getattr(o, "name", "") or ""),
            type=str(getattr(o, "type", "") or ""),
            latency_ms=round(latency * 1000, 1) if latency is not None else None,
            input=str(getattr(o, "input", "") or "") or None,
            output=str(getattr(o, "output", "") or "") or None,
        )

    total_latency = getattr(trace, "latency", None)
    html_path = getattr(trace, "htmlPath", f"/trace/{trace_id}")
    observations = [_obs(o) for o in (getattr(trace, "observations", None) or [])]

    return TraceResponse(
        trace_id=trace_id,
        input=str(getattr(trace, "input", "") or "") or None,
        output=str(getattr(trace, "output", "") or "") or None,
        latency_ms=round(total_latency * 1000, 1) if total_latency is not None else None,
        observations=observations,
        langfuse_url=f"{settings.langfuse_host.rstrip('/')}{html_path}",
    )


@router.post("/query", response_model=AgentQueryResponse)
async def agent_query(
    body: AgentQueryRequest,
    graph: RAGAgentGraph = Depends(get_rag_graph),
) -> AgentQueryResponse:
    """Run the agentic RAG pipeline for a financial question.

    Returns the synthesized answer, source citations enriched from filing
    metadata, hallucination check result, and a Langfuse trace_id for
    observability.
    """
    result = await graph.run(
        query=body.query,
        filters=body.filters,
        session_id=body.session_id,
        user_id=body.user_id,
    )

    citations = [CitationOut(**asdict(c)) for c in result.citations]

    return AgentQueryResponse(
        query=result.query,
        answer=result.answer,
        citations=citations,
        sub_queries=result.sub_queries,
        is_hallucinated=result.is_hallucinated,
        groundedness=result.groundedness,
        trace_id=result.trace_id,
    )
