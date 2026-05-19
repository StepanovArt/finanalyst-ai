"""
Hallucination Checker (Self-RAG faithfulness check).

Verifies that every claim in the synthesized answer is traceable to the
retrieved context. Catches cases where the LLM "fills in" numbers or facts
not present in the chunks.

Labels:
  grounded            — all claims supported by context
  partially_grounded  — most claims supported; minor gaps or vague statements
  hallucinated        — answer contains specific claims not found in context

Why a separate check instead of just prompting the synthesizer better?
Chain-of-thought faithfulness checking works better as a second pass — the
checker sees the already-written answer and can spot specific claim-by-claim
mismatches that a single-pass synthesizer misses.
"""

import json
from dataclasses import dataclass
from enum import StrEnum

from loguru import logger

from app.rag.retrieval import SearchResult
from app.services.llm.base import LLMProvider, Message

_SYSTEM_PROMPT = """\
You are a faithfulness checker for a financial RAG system.

Given a generated answer and the source document chunks it was based on,
determine whether the answer is grounded in the context.

Label the answer as:
  "grounded"            — every specific claim (number, date, metric, percentage)
                          is directly supported by the context chunks
  "partially_grounded"  — most claims are supported; minor vagueness or
                          small inferential leaps present, but no invented facts
  "hallucinated"        — answer contains specific figures, facts, or claims
                          that do NOT appear in the provided chunks

Respond ONLY with a JSON object. No markdown.

Response format:
{"label": "grounded|partially_grounded|hallucinated", "reason": "<one sentence>"}
"""

_USER_TEMPLATE = """\
Generated answer:
{answer}

Source context:
{context}

Is the answer grounded in the context?"""


class GroundednessLabel(StrEnum):
    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    HALLUCINATED = "hallucinated"


@dataclass
class HallucinationCheckResult:
    label: GroundednessLabel
    reason: str

    @property
    def is_acceptable(self) -> bool:
        """True when the answer can be shown to the user as-is."""
        return self.label in (GroundednessLabel.GROUNDED, GroundednessLabel.PARTIALLY_GROUNDED)


class HallucinationChecker:
    """Checks synthesized answers for faithfulness to retrieved context."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def check(
        self,
        answer: str,
        chunks: list[SearchResult],
    ) -> HallucinationCheckResult:
        """Verify the answer is grounded in the retrieved chunks.

        Args:
            answer: the synthesized answer text
            chunks: the chunks used as context for synthesis

        Returns:
            HallucinationCheckResult with label and reason;
            falls back to partially_grounded on any error
        """
        if not answer.strip():
            return HallucinationCheckResult(
                label=GroundednessLabel.HALLUCINATED,
                reason="Empty answer cannot be verified.",
            )

        context = "\n\n".join(
            f"[chunk_id: {c.chunk_id}] {c.company} {c.filing_type} {c.year} {c.quarter}"
            f" — {c.section}:\n{c.text[:500]}"
            for c in chunks
        )

        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(
                role="user",
                content=_USER_TEMPLATE.format(answer=answer, context=context),
            ),
        ]

        try:
            raw = await self._llm.generate(messages)
            data = json.loads(raw)
            label = GroundednessLabel(data["label"])
            reason = str(data.get("reason", "")).strip()

            logger.info(f"Hallucination check: {label} — {reason}")
            return HallucinationCheckResult(label=label, reason=reason)

        except Exception as exc:
            logger.warning(f"HallucinationChecker fallback (error: {exc}): partially_grounded")
            return HallucinationCheckResult(
                label=GroundednessLabel.PARTIALLY_GROUNDED,
                reason="Check could not be completed.",
            )
