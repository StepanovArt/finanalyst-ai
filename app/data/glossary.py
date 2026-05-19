"""
Financial glossary loader and synonym lookup.

Used by the Query Rewriting Agent to expand user queries:
  "what's Apple's FCF?" → also searches for "free cash flow", "cash flow after capex"
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_GLOSSARY_PATH = Path(__file__).parent / "financial_glossary.json"


@dataclass(frozen=True)
class GlossaryTerm:
    term: str
    full_name: str
    category: str
    synonyms: tuple[str, ...]


@lru_cache(maxsize=1)
def load_glossary() -> list[GlossaryTerm]:
    """Load and cache the financial glossary from JSON."""
    data = json.loads(_GLOSSARY_PATH.read_text(encoding="utf-8"))
    return [
        GlossaryTerm(
            term=entry["term"],
            full_name=entry["full_name"],
            category=entry["category"],
            synonyms=tuple(entry["synonyms"]),
        )
        for entry in data["terms"]
    ]


def get_synonyms(query: str) -> list[str]:
    """Return all synonyms for any financial term found in the query.

    Performs case-insensitive substring match against term names and synonyms.
    Returns deduplicated list of synonyms to append to the original query.

    Example:
        get_synonyms("what is Apple's FCF?")
        → ["free cash flow", "levered free cash flow", "cash flow after capex"]
    """
    query_lower = query.lower()
    result: list[str] = []
    seen: set[str] = set()

    for entry in load_glossary():
        # Match against the short term, full name, or any existing synonym
        candidates = [entry.term.lower(), entry.full_name.lower()] + [
            s.lower() for s in entry.synonyms
        ]
        if any(c in query_lower for c in candidates):
            for synonym in entry.synonyms:
                if synonym.lower() not in seen and synonym.lower() not in query_lower:
                    result.append(synonym)
                    seen.add(synonym.lower())

    return result


def expand_query(query: str) -> str:
    """Append relevant financial synonyms to a query string.

    Example:
        expand_query("Apple EBITDA 2024")
        → "Apple EBITDA 2024 [also: earnings before interest taxes depreciation
           and amortization, operating earnings, adjusted operating profit]"
    """
    synonyms = get_synonyms(query)
    if not synonyms:
        return query
    expansion = "; ".join(synonyms)
    return f"{query} [also: {expansion}]"
