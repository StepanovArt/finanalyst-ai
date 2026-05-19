"""
Query Decomposition Agent.

Splits a complex financial query into independent sub-queries that can each
be retrieved separately, then merged for answer synthesis.

Examples:
  "Compare Apple and Microsoft revenue and margins in 2024"
  → ["Apple revenue 2024", "Microsoft revenue 2024",
     "Apple profit margins 2024", "Microsoft profit margins 2024"]

  "What is Apple's net income?" → ["What is Apple's net income?"]

Why decompose?
A single embedding for "Compare A and B on X and Y" pulls toward the average
of all four concepts — retrieval misses chunks that are specific to one pair.
Separate queries each get focused embeddings → better recall.
"""

import json

from loguru import logger
from pydantic import BaseModel

from app.services.llm.base import LLMProvider, Message

_SYSTEM_PROMPT = """\
You are a financial query decomposition assistant.

Your job: decide if a user question is complex (asks about multiple companies,
multiple metrics, or multiple time periods), and if so, split it into simple
independent sub-queries — one concept per query.

Rules:
- Each sub-query must be self-contained and answerable from a single document chunk.
- Keep company names, tickers, years, and quarters in every sub-query that needs them.
- If the query is already simple (one company, one metric), return it unchanged.
- Return 1 to 5 sub-queries. Never more than 5.
- Respond ONLY with a JSON object. No explanation, no markdown.

Response format:
{"sub_queries": ["sub-query 1", "sub-query 2"]}
"""

_USER_TEMPLATE = 'Decompose this financial query into sub-queries:\n\n"{query}"'


class DecompositionResult(BaseModel):
    sub_queries: list[str]

    @property
    def is_complex(self) -> bool:
        return len(self.sub_queries) > 1


class QueryDecomposer:
    """Splits complex financial queries into independent sub-queries via LLM."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def decompose(self, query: str) -> DecompositionResult:
        """Decompose a query. Falls back to [query] on any LLM/parse error.

        Args:
            query: the original user question

        Returns:
            DecompositionResult with sub_queries list (len >= 1)
        """
        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=_USER_TEMPLATE.format(query=query)),
        ]

        try:
            raw = await self._llm.generate(messages)
            data = json.loads(raw)
            sub_queries = data.get("sub_queries", [])

            if not isinstance(sub_queries, list) or not sub_queries:
                raise ValueError("empty or invalid sub_queries")

            # Sanitise: ensure all entries are non-empty strings, cap at 5
            sub_queries = [str(q).strip() for q in sub_queries if str(q).strip()][:5]
            if not sub_queries:
                raise ValueError("all sub_queries were empty after sanitisation")

            result = DecompositionResult(sub_queries=sub_queries)
            if result.is_complex:
                logger.debug(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
            return result

        except Exception as exc:
            logger.warning(f"QueryDecomposer fallback (error: {exc}): returning original query")
            return DecompositionResult(sub_queries=[query])
