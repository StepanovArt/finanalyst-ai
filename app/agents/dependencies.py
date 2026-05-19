"""
FastAPI dependency that builds and caches the RAGAgentGraph singleton.

All heavy initialisation (LLM client, QdrantClient) happens once at first
request rather than at import time, so the app starts fast and tests can
override get_rag_graph() via dependency_overrides without touching real infra.
"""

from functools import lru_cache

from app.agents.decomposer import QueryDecomposer
from app.agents.grader import RelevanceGrader
from app.agents.graph import RAGAgentGraph
from app.agents.hallucination_checker import HallucinationChecker
from app.agents.rewriter import QueryRewriter
from app.agents.self_correction import SelfCorrectionLoop
from app.agents.synthesizer import AnswerSynthesizer
from app.rag.retrieval import HybridRetriever
from app.services.llm.dependencies import get_llm_provider


@lru_cache(maxsize=1)
def _build_rag_graph() -> RAGAgentGraph:
    llm = get_llm_provider()
    retriever = HybridRetriever()
    decomposer = QueryDecomposer(llm=llm)
    rewriter = QueryRewriter(llm=llm)
    grader = RelevanceGrader(llm=llm)
    correction_loop = SelfCorrectionLoop(
        retriever=retriever,
        grader=grader,
        rewriter=rewriter,
    )
    synthesizer = AnswerSynthesizer(llm=llm)
    checker = HallucinationChecker(llm=llm)
    return RAGAgentGraph(
        decomposer=decomposer,
        correction_loop=correction_loop,
        synthesizer=synthesizer,
        checker=checker,
    )


def get_rag_graph() -> RAGAgentGraph:
    return _build_rag_graph()
