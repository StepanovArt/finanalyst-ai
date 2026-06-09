"""
Structured chunking of SEC filing text by report sections.

Strategy:
1. Split text into sections using ITEM headers (SEC standard structure)
2. Detect financial statement subsections (Income Statement, Balance Sheet, etc.)
3. For sections larger than MAX_WORDS, slide a window with overlap
4. Each chunk carries full metadata for downstream filtering
"""

import hashlib
import re
from dataclasses import dataclass

from app.rag.ingestion import FilingDocument

MAX_WORDS = 800
OVERLAP_WORDS = 100

# Map of regex pattern → canonical section name
# Order matters: more specific patterns first
_SECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Financial statement headers (appear inside Item 8 / Item 1 Part II)
    (re.compile(r"consolidated\s+statements?\s+of\s+operations", re.I), "Income Statement"),
    (
        re.compile(r"consolidated\s+statements?\s+of\s+(comprehensive\s+)?income", re.I),
        "Income Statement",
    ),
    (re.compile(r"consolidated\s+balance\s+sheets?", re.I), "Balance Sheet"),
    (
        re.compile(r"consolidated\s+statements?\s+of\s+cash\s+flows?", re.I),
        "Cash Flow Statement",
    ),
    (
        re.compile(r"consolidated\s+statements?\s+of\s+(stockholders|shareholders)", re.I),
        "Equity Statement",
    ),
    (
        re.compile(r"notes?\s+to\s+(the\s+)?consolidated\s+financial\s+statements?", re.I),
        "Notes to FS",
    ),
    # 10-K ITEM headers
    (re.compile(r"item\s+1a[\.\s]+risk\s+factors?", re.I), "Risk Factors"),
    (re.compile(r"item\s+1[\.\s]+business", re.I), "Business Overview"),
    (re.compile(r"item\s+7a[\.\s]+quantitative", re.I), "Market Risk"),
    (re.compile(r"item\s+7[\.\s]+management.s\s+discussion", re.I), "MD&A"),
    (re.compile(r"item\s+8[\.\s]+financial\s+statements?", re.I), "Financial Statements"),
    (re.compile(r"item\s+9a[\.\s]+controls", re.I), "Controls & Procedures"),
    # 10-Q ITEM headers
    (re.compile(r"item\s+2[\.\s]+management.s\s+discussion", re.I), "MD&A"),
    (re.compile(r"item\s+1[\.\s]+financial\s+statements?", re.I), "Financial Statements"),
    (re.compile(r"item\s+3[\.\s]+quantitative", re.I), "Market Risk"),
    (re.compile(r"item\s+4[\.\s]+controls", re.I), "Controls & Procedures"),
]

# Sections worth indexing — skip boilerplate
_RELEVANT_SECTIONS = {
    "Business Overview",
    "Risk Factors",
    "MD&A",
    "Financial Statements",
    "Income Statement",
    "Balance Sheet",
    "Cash Flow Statement",
    "Equity Statement",
    "Notes to FS",
    "Market Risk",
}


def _derive_year(period: str) -> int:
    """Extract year from YYYYMMDD period string."""
    return int(period[:4])


def _derive_quarter(period: str, filing_type: str) -> str:
    """Derive fiscal quarter label from period and filing type.

    10-K is always FY. For 10-Q we map period end month to quarter:
    Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec.
    Note: this is calendar-quarter approximation; actual fiscal quarter
    depends on the company's fiscal year end.
    """
    if filing_type == "10-K":
        return "FY"
    month = int(period[4:6])
    return f"Q{(month - 1) // 3 + 1}"


@dataclass
class Chunk:
    """A single retrieval unit from a filing."""

    id: str
    ticker: str
    company: str
    filing_type: str
    period: str
    year: int
    quarter: str
    accession: str
    section: str
    currency: str
    text: str
    chunk_index: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "id": self.id,
            "ticker": self.ticker,
            "company": self.company,
            "filing_type": self.filing_type,
            "period": self.period,
            "year": self.year,
            "quarter": self.quarter,
            "accession": self.accession,
            "section": self.section,
            "currency": self.currency,
            "text": self.text,
            "chunk_index": self.chunk_index,
        }


def _detect_section(line: str) -> str | None:
    """Return canonical section name if line matches a section header."""
    for pattern, name in _SECTION_PATTERNS:
        if pattern.search(line):
            return name
    return None


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split filing text into (section_name, section_text) pairs."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_section = "Preamble"
    current_lines: list[str] = []

    for line in lines:
        detected = _detect_section(line)
        if detected:
            if current_lines:
                sections.append((current_section, current_lines))
            current_section = detected
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_section, current_lines))

    return [(name, "\n".join(lines).strip()) for name, lines in sections]


def _make_id(
    ticker: str,
    period: str,
    filing_type: str,
    accession: str,
    section: str,
    seq: int,
) -> str:
    """Build a globally-unique chunk id.

    The hash includes the accession (distinguishes filings that share a
    ticker+type+period, e.g. an amended re-filing) and a document-global running
    sequence number `seq` (distinguishes chunks across section blocks that share
    a canonical name — without it, every "Income Statement" or "Notes to FS"
    block restarts at index 0 and collides). 12 hex chars make birthday
    collisions across the corpus negligible.
    """
    raw = f"{ticker}_{filing_type}_{period}_{accession}_{section}_{seq}"
    short = hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"{ticker}_{period}_{short}"


def _sliding_window(
    section_text: str,
    max_words: int = MAX_WORDS,
    overlap: int = OVERLAP_WORDS,
) -> list[str]:
    """Split long text into overlapping word-window chunks."""
    words = section_text.split()
    if len(words) <= max_words:
        return [section_text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        start += max_words - overlap

    return chunks


def chunk_document(doc: FilingDocument) -> list[Chunk]:
    """Chunk a FilingDocument into retrieval-ready Chunks."""
    sections = _split_into_sections(doc.text)
    chunks: list[Chunk] = []
    seq = 0  # document-global counter — guarantees unique ids across all sections

    for section_name, section_text in sections:
        if section_name not in _RELEVANT_SECTIONS:
            continue
        if len(section_text.split()) < 30:
            continue

        windows = _sliding_window(section_text)
        for idx, window_text in enumerate(windows):
            chunk_id = _make_id(
                doc.ticker, doc.period, doc.filing_type, doc.accession, section_name, seq
            )
            seq += 1
            chunks.append(
                Chunk(
                    id=chunk_id,
                    ticker=doc.ticker,
                    company=doc.company,
                    filing_type=doc.filing_type,
                    period=doc.period,
                    year=_derive_year(doc.period),
                    quarter=_derive_quarter(doc.period, doc.filing_type),
                    accession=doc.accession,
                    section=section_name,
                    currency=doc.currency,
                    text=window_text,
                    chunk_index=idx,
                )
            )

    return chunks
