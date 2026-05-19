"""
LangGraph orchestration for the Agentic RAG pipeline.

Graph flow:
  decompose → retrieve → synthesize → check → END

State machine nodes:
  decompose  — QueryDecomposer splits query into sub-queries
  retrieve   — SelfCorrectionLoop runs per sub-query; chunks are accumulated
  synthesize — AnswerSynthesizer generates answer with citations
  check      — HallucinationChecker validates answer faithfulness

Why LangGraph over a plain async function?
- Explicit state machine makes the flow auditable and visualisable
- Each node is independently testable and replaceable
- LangGraph handles async node execution natively
- Cycles are trivial to add later (e.g. re-synthesize on hallucination)
- Integrates with LangSmith / Langfuse for tracing via CallbackHandler
"""

import uuid
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from app.agents.decomposer import QueryDecomposer
from app.agents.hallucination_checker import GroundednessLabel, HallucinationChecker
from app.agents.self_correction import SelfCorrectionLoop
from app.agents.synthesizer import AnswerSynthesizer, Citation
from app.core.tracing import get_langfuse_handler
from app.rag.retrieval import SearchResult


class AgentState(TypedDict):
    """Mutable state passed between LangGraph nodes."""

    query: str
    filters: dict | None
    sub_queries: list[str]
    chunks: list[SearchResult]
    answer: str
    citations: list[Citation]
    groundedness: str  # GroundednessLabel value
    is_hallucinated: bool


@dataclass
class AgentResponse:
    """Final output returned to the API layer."""

    query: str
    answer: str
    citations: list[Citation]
    sub_queries: list[str]
    is_hallucinated: bool
    groundedness: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class RAGAgentGraph:
    """Compiled LangGraph that orchestrates the full RAG pipeline."""

    def __init__(
        self,
        decomposer: QueryDecomposer,
        correction_loop: SelfCorrectionLoop,
        synthesizer: AnswerSynthesizer,
        checker: HallucinationChecker,
    ) -> None:
        self._decomposer = decomposer
        self._loop = correction_loop
        self._synthesizer = synthesizer
        self._checker = checker
        self._graph = self._build()

    def _build(self):  # type: ignore[return]
        graph: StateGraph = StateGraph(AgentState)

        graph.add_node("decompose", self._decompose_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("synthesize", self._synthesize_node)
        graph.add_node("check", self._check_node)

        graph.set_entry_point("decompose")
        graph.add_edge("decompose", "retrieve")
        graph.add_edge("retrieve", "synthesize")
        graph.add_edge("synthesize", "check")
        graph.add_edge("check", END)

        return graph.compile()

    async def _decompose_node(self, state: AgentState) -> dict:
        result = await self._decomposer.decompose(state["query"])
        logger.info(f"Decomposed into {len(result.sub_queries)} sub-queries")
        return {"sub_queries": result.sub_queries}

    async def _retrieve_node(self, state: AgentState) -> dict:
        all_chunks: list[SearchResult] = []

        for sub_query in state["sub_queries"]:
            result = await self._loop.run(
                sub_query,
                filters=state.get("filters"),
            )
            all_chunks.extend(result.final_chunks)

        # Deduplicate by chunk_id (same chunk may appear in multiple sub-query results)
        seen: set[str] = set()
        unique = [c for c in all_chunks if not (c.chunk_id in seen or seen.add(c.chunk_id))]  # type: ignore[func-returns-value]

        n_sub = len(state["sub_queries"])
        logger.info(f"Retrieved {len(unique)} unique chunks across {n_sub} sub-queries")
        return {"chunks": unique}

    async def _synthesize_node(self, state: AgentState) -> dict:
        result = await self._synthesizer.synthesize(state["query"], state["chunks"])
        return {"answer": result.answer, "citations": result.citations}

    async def _check_node(self, state: AgentState) -> dict:
        result = await self._checker.check(state["answer"], state["chunks"])
        is_hallucinated = not result.is_acceptable
        if is_hallucinated:
            logger.warning(f"Hallucination detected: {result.reason}")
        return {
            "groundedness": result.label,
            "is_hallucinated": is_hallucinated,
        }

    async def run(
        self,
        query: str,
        filters: dict | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AgentResponse:
        """Execute the full RAG pipeline for a user query.

        Args:
            query: the user question
            filters: optional Qdrant metadata filters (e.g. {"ticker": "AAPL"})
            session_id: forwarded to Langfuse for session grouping
            user_id: forwarded to Langfuse for per-user analytics

        Returns:
            AgentResponse with answer, citations, quality metadata, and trace_id
        """
        trace_id = str(uuid.uuid4())

        initial: AgentState = {
            "query": query,
            "filters": filters,
            "sub_queries": [],
            "chunks": [],
            "answer": "",
            "citations": [],
            "groundedness": GroundednessLabel.PARTIALLY_GROUNDED,
            "is_hallucinated": False,
        }

        config: dict = {}
        handler = get_langfuse_handler(session_id=session_id, user_id=user_id)
        if handler is not None:
            config["callbacks"] = [handler]

        final: AgentState = await self._graph.ainvoke(initial, config=config or None)

        return AgentResponse(
            query=query,
            answer=final["answer"],
            citations=final["citations"],
            sub_queries=final["sub_queries"],
            is_hallucinated=final["is_hallucinated"],
            groundedness=final["groundedness"],
            trace_id=trace_id,
        )
