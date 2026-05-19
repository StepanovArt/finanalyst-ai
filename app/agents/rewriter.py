"""
Query Rewriting Agent.

Two-stage rewriting pipeline:
1. Deterministic: expand financial abbreviations via glossary (FCF → free cash flow)
2. LLM-based: rephrase for better retrieval using financial document language

Used by the self-correction loop when retrieval returns low-relevance results.
The `attempt` parameter signals to the LLM how aggressive to be — later attempts
use more varied phrasing to escape local retrieval minima.
"""

from dataclasses import dataclass

from loguru import logger

from app.data.glossary import get_synonyms
from app.services.llm.base import LLMProvider, Message

_SYSTEM_PROMPT = """\
You are a financial document retrieval expert.

Your task: rewrite a user query to improve retrieval from SEC filings (10-K, 10-Q).

Rules:
- Use precise financial terminology as it appears in SEC filings.
- Expand abbreviations (FCF → free cash flow, COGS → cost of revenue).
- If the query mentions a comparison, focus on one side at a time in the rewrite.
- Make the query more specific: add "in the filing", "reported", "for the period" where natural.
- On attempt > 1, rephrase significantly — use different synonyms and sentence structure.
- Return ONLY the rewritten query as plain text. No explanation, no quotes.
"""

_USER_TEMPLATE = """\
Original query: {query}
Attempt number: {attempt}
Known synonyms for terms in this query: {synonyms}

Rewrite the query for better SEC filing retrieval:"""


@dataclass
class RewriteResult:
    original_query: str
    rewritten_query: str
    synonyms_added: list[str]

    @property
    def changed(self) -> bool:
        return self.rewritten_query.strip() != self.original_query.strip()


class QueryRewriter:
    """Rewrites financial queries for better retrieval using glossary + LLM."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def rewrite(self, query: str, attempt: int = 1) -> RewriteResult:
        """Rewrite a query for improved retrieval.

        Args:
            query: the original or previously rewritten query
            attempt: retry count (1 = first rewrite, 2+ = more aggressive)

        Returns:
            RewriteResult with the rewritten query and metadata
        """
        synonyms = get_synonyms(query)
        synonyms_str = "; ".join(synonyms) if synonyms else "none found"

        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(
                role="user",
                content=_USER_TEMPLATE.format(
                    query=query,
                    attempt=attempt,
                    synonyms=synonyms_str,
                ),
            ),
        ]

        try:
            rewritten = await self._llm.generate(messages)
            rewritten = rewritten.strip().strip('"').strip("'")

            if not rewritten:
                raise ValueError("LLM returned empty rewrite")

            logger.debug(f"Rewrite attempt {attempt}: '{query}' → '{rewritten}'")
            return RewriteResult(
                original_query=query,
                rewritten_query=rewritten,
                synonyms_added=synonyms,
            )

        except Exception as exc:
            logger.warning(f"QueryRewriter fallback (error: {exc}): returning original query")
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                synonyms_added=synonyms,
            )
