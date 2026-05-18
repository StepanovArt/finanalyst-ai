"""
Parse SEC EDGAR full-submission.txt files.

EDGAR stores filings as SGML envelopes containing multiple documents.
We extract only the primary filing (10-K or 10-Q) and parse its HTML.

Output: FilingDocument with plain text + extracted tables.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag


@dataclass
class FilingTable:
    """A financial table extracted from a filing."""

    headers: list[str]
    rows: list[list[str]]
    caption: str = ""


@dataclass
class FilingDocument:
    """Parsed content of a single SEC filing."""

    ticker: str
    filing_type: str  # "10-K" or "10-Q"
    accession: str
    period: str  # e.g. "20240928"
    text: str
    tables: list[FilingTable] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _extract_main_document(raw: str, filing_type: str) -> str:
    """Extract the primary HTML document from the SGML envelope."""
    # EDGAR wraps each document in <DOCUMENT>...</DOCUMENT> blocks
    # We want the one whose <TYPE> matches our filing type (10-K or 10-Q)
    pattern = re.compile(
        r"<DOCUMENT>\s*<TYPE>" + re.escape(filing_type) + r"\b.*?</DOCUMENT>",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(raw)
    if not match:
        raise ValueError(f"No {filing_type} document found in SGML envelope")

    block = match.group(0)

    # The actual content is between <TEXT> and </TEXT> tags
    text_match = re.search(r"<TEXT>(.*?)</TEXT>", block, re.DOTALL | re.IGNORECASE)
    if not text_match:
        raise ValueError("No <TEXT> block found inside document")

    return text_match.group(1).strip()


def _parse_tables(soup: BeautifulSoup) -> list[FilingTable]:
    """Extract all HTML tables as structured data."""
    tables: list[FilingTable] = []

    for table_tag in soup.find_all("table"):
        if not isinstance(table_tag, Tag):
            continue

        # Try to find a caption (often in a <p> or <div> just before the table)
        caption = ""
        prev = table_tag.find_previous_sibling()
        if prev and isinstance(prev, Tag):
            caption_text = prev.get_text(strip=True)
            if len(caption_text) < 200:
                caption = caption_text

        rows_data: list[list[str]] = []
        for tr in table_tag.find_all("tr"):
            if not isinstance(tr, Tag):
                continue
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            # Skip empty rows and single-cell decorative rows
            if len(cells) > 1 and any(c.strip() for c in cells):
                rows_data.append(cells)

        if not rows_data:
            continue

        # First row is headers if it contains <th> tags
        first_tr = table_tag.find("tr")
        has_th = first_tr and isinstance(first_tr, Tag) and first_tr.find("th")
        if has_th and len(rows_data) > 1:
            tables.append(FilingTable(headers=rows_data[0], rows=rows_data[1:], caption=caption))
        else:
            tables.append(FilingTable(headers=[], rows=rows_data, caption=caption))

    return tables


def _extract_text(soup: BeautifulSoup) -> str:
    """Extract clean plain text, removing scripts and styles."""
    for tag in soup(["script", "style", "ix:header", "ix:hidden"]):
        tag.decompose()

    # get_text with separator preserves paragraph boundaries
    text = soup.get_text(separator="\n", strip=True)

    # Collapse runs of blank lines into a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _parse_period(raw: str) -> str:
    """Extract CONFORMED PERIOD OF REPORT from SGML header."""
    match = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", raw)
    return match.group(1) if match else "unknown"


def parse_filing(path: Path) -> FilingDocument:
    """Parse a full-submission.txt file into a FilingDocument.

    Args:
        path: Path to full-submission.txt

    Returns:
        FilingDocument with extracted text and tables
    """
    # Path structure: .../sec-edgar-filings/{TICKER}/{TYPE}/{ACCESSION}/full-submission.txt
    parts = path.parts
    accession = parts[-2]
    filing_type = parts[-3]
    ticker = parts[-4]

    raw = path.read_text(encoding="utf-8", errors="replace")
    period = _parse_period(raw)

    html_content = _extract_main_document(raw, filing_type)
    soup = BeautifulSoup(html_content, "lxml")

    return FilingDocument(
        ticker=ticker,
        filing_type=filing_type,
        accession=accession,
        period=period,
        text=_extract_text(soup),
        tables=_parse_tables(soup),
    )


def parse_all_filings(filings_dir: Path) -> list[FilingDocument]:
    """Parse all downloaded filings from the EDGAR directory structure."""
    docs: list[FilingDocument] = []
    for submission_file in sorted(filings_dir.rglob("full-submission.txt")):
        try:
            doc = parse_filing(submission_file)
            docs.append(doc)
        except (ValueError, OSError) as exc:
            print(f"  SKIP {submission_file.parent.name}: {exc}")
    return docs
