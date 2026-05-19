"""
Agent query endpoint.

POST /agent/query — runs the full agentic RAG pipeline and returns
an answer with citations, quality signals, and a Langfuse trace_id.
"""

from dataclasses import asdict

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.agents.dependencies import RAGAgentGraph, get_rag_graph

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
