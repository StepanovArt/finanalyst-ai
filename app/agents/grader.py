"""
Relevance Grader Agent.

Evaluates each retrieved chunk against the query and assigns a label:
  relevant   — chunk clearly contains information to answer the query
  ambiguous  — chunk is partially related; may help synthesis
  irrelevant — chunk does not address the query

All chunks are graded in a single LLM call to minimise latency and cost.

The GradingResult.has_sufficient_context property decides whether to proceed
to answer synthesis or trigger the self-correction loop (retry with rewritten query).
"""

import json
from dataclasses import dataclass
from enum import StrEnum

from loguru import logger

from app.rag.retrieval import SearchResult
from app.services.llm.base import LLMProvider, Message

_SYSTEM_PROMPT = """\
You are a relevance grader for a financial RAG system.

Given a user query and a list of document chunks from SEC filings (10-K, 10-Q),
grade each chunk as:
  "relevant"   — directly addresses the query (contains the asked metric, company, period)
  "ambiguous"  — partially related; mentions the topic but lacks specifics
  "irrelevant" — does not address the query at all

Respond ONLY with a JSON object. No explanation, no markdown.

Response format:
{
  "grades": [
    {"chunk_id": "<id>", "label": "relevant|ambiguous|irrelevant", "reason": "<one sentence>"},
    ...
  ]
}
"""

_USER_TEMPLATE = """\
Query: {query}

Chunks to grade:
{chunks_text}

Grade each chunk:"""


class GradeLabel(StrEnum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    AMBIGUOUS = "ambiguous"


@dataclass
class ChunkGrade:
    chunk_id: str
    label: GradeLabel
    reason: str


@dataclass
class GradingResult:
    grades: list[ChunkGrade]

    @property
    def relevant_count(self) -> int:
        return sum(1 for g in self.grades if g.label == GradeLabel.RELEVANT)

    @property
    def has_sufficient_context(self) -> bool:
        """True when at least one chunk is relevant (enough to attempt synthesis)."""
        return self.relevant_count >= 1

    def relevant_chunk_ids(self) -> list[str]:
        return [g.chunk_id for g in self.grades if g.label == GradeLabel.RELEVANT]

    def ambiguous_chunk_ids(self) -> list[str]:
        return [g.chunk_id for g in self.grades if g.label == GradeLabel.AMBIGUOUS]


class RelevanceGrader:
    """Grades retrieved chunks for relevance to the query via a single LLM call."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def grade(self, query: str, chunks: list[SearchResult]) -> GradingResult:
        """Grade all chunks in one LLM call.

        Args:
            query: the user question
            chunks: retrieved SearchResult objects to evaluate

        Returns:
            GradingResult with per-chunk labels; falls back to all-ambiguous on error
        """
        if not chunks:
            return GradingResult(grades=[])

        chunks_text = "\n\n".join(
            f"[chunk_id: {c.chunk_id}]\n"
            f"Source: {c.company} {c.filing_type} {c.year} {c.quarter} — {c.section}\n"
            f"Text: {c.text[:400]}"
            for c in chunks
        )

        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(
                role="user",
                content=_USER_TEMPLATE.format(query=query, chunks_text=chunks_text),
            ),
        ]

        try:
            raw = await self._llm.generate(messages)
            data = json.loads(raw)
            raw_grades = data.get("grades", [])

            grades = []
            for g in raw_grades:
                try:
                    label = GradeLabel(g["label"])
                except (KeyError, ValueError):
                    label = GradeLabel.AMBIGUOUS
                grades.append(
                    ChunkGrade(
                        chunk_id=str(g.get("chunk_id", "")),
                        label=label,
                        reason=str(g.get("reason", "")),
                    )
                )

            relevant = sum(1 for g in grades if g.label == GradeLabel.RELEVANT)
            logger.debug(
                f"Graded {len(grades)} chunks: {relevant} relevant, "
                f"{len(grades) - relevant} not relevant"
            )
            return GradingResult(grades=grades)

        except Exception as exc:
            logger.warning(f"RelevanceGrader fallback (error: {exc}): marking all as ambiguous")
            return GradingResult(
                grades=[
                    ChunkGrade(
                        chunk_id=c.chunk_id,
                        label=GradeLabel.AMBIGUOUS,
                        reason="grader error",
                    )
                    for c in chunks
                ]
            )
