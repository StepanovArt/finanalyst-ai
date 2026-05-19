"""
Answer Synthesis Agent.

Generates a grounded answer from retrieved chunks with source citations.
The LLM is instructed to answer ONLY from the provided context — no hallucination.

Citations are returned as structured objects so the API layer can render
them as clickable references (filing → section → excerpt).
"""

import json
from dataclasses import dataclass

from loguru import logger

from app.rag.retrieval import SearchResult
from app.services.llm.base import LLMProvider, Message

_SYSTEM_PROMPT = """\
You are a financial analyst assistant answering questions from SEC filings.

Rules:
- Answer ONLY using the provided document chunks. Do not use outside knowledge.
- Be precise: include exact figures, dates, and percentages from the text.
- Cite the chunks you used by their chunk_id.
- If the context does not contain enough information, say so explicitly.
- Return ONLY a JSON object. No markdown, no explanation outside the JSON.

Response format:
{
  "answer": "<your answer in 1-4 sentences>",
  "citations": [
    {"chunk_id": "<id>", "quote": "<exact short excerpt that supports your answer>"},
    ...
  ]
}
"""

_USER_TEMPLATE = """\
Question: {query}

Context chunks:
{context}

Answer the question using only the chunks above:"""


@dataclass
class Citation:
    chunk_id: str
    quote: str
    company: str
    filing_type: str
    year: int
    quarter: str
    section: str


@dataclass
class SynthesisResult:
    answer: str
    citations: list[Citation]

    @property
    def has_citations(self) -> bool:
        return len(self.citations) > 0


class AnswerSynthesizer:
    """Generates grounded answers with source citations from retrieved chunks."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def synthesize(
        self,
        query: str,
        chunks: list[SearchResult],
    ) -> SynthesisResult:
        """Generate an answer with citations.

        Args:
            query: the user question
            chunks: retrieved chunks to use as context (pre-filtered for relevance)

        Returns:
            SynthesisResult with answer text and structured citations
        """
        if not chunks:
            return SynthesisResult(
                answer="No relevant information found in the available filings.",
                citations=[],
            )

        chunk_index = {c.chunk_id: c for c in chunks}
        context = "\n\n".join(
            f"[chunk_id: {c.chunk_id}]\n"
            f"Source: {c.company} ({c.ticker}) {c.filing_type} {c.year} {c.quarter}"
            f" — {c.section}\n"
            f"Text: {c.text[:600]}"
            for c in chunks
        )

        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(
                role="user",
                content=_USER_TEMPLATE.format(query=query, context=context),
            ),
        ]

        try:
            raw = await self._llm.generate(messages)
            data = json.loads(raw)

            answer = str(data.get("answer", "")).strip()
            if not answer:
                raise ValueError("empty answer in LLM response")

            citations = []
            for c in data.get("citations", []):
                chunk_id = str(c.get("chunk_id", ""))
                source = chunk_index.get(chunk_id)
                if source is None:
                    continue
                citations.append(
                    Citation(
                        chunk_id=chunk_id,
                        quote=str(c.get("quote", ""))[:300],
                        company=source.company,
                        filing_type=source.filing_type,
                        year=source.year,
                        quarter=source.quarter,
                        section=source.section,
                    )
                )

            logger.info(f"Synthesized answer with {len(citations)} citation(s)")
            return SynthesisResult(answer=answer, citations=citations)

        except Exception as exc:
            logger.warning(f"AnswerSynthesizer fallback (error: {exc})")
            return SynthesisResult(
                answer="Unable to synthesize an answer from the retrieved context.",
                citations=[],
            )
