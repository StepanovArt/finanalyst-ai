"""
Unit tests for RAG pipeline: ingestion, chunking, contextual prefix, indexing job.
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal SGML fixture (mimics a full-submission.txt structure)
# ---------------------------------------------------------------------------

_SGML = """\
COMPANY CONFORMED NAME: Acme Corp
CONFORMED PERIOD OF REPORT: 20240930

<DOCUMENT>
<TYPE>10-K
<TEXT>
<html>
<body>
<p>ITEM 1. BUSINESS</p>
<p>Acme Corp is a technology company that sells widgets globally.
The company reports in United States dollars.</p>
<p>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</p>
<p>Revenue increased 10 percent year over year driven by strong demand.</p>
</body>
</html>
</TEXT>
</DOCUMENT>
"""

_SGML_EUR = """\
COMPANY CONFORMED NAME: EuroCo
CONFORMED PERIOD OF REPORT: 20240630
<DOCUMENT>
<TYPE>10-Q
<TEXT>
<html><body><p>The company reports in Euro currency EUR.</p></body></html>
</TEXT>
</DOCUMENT>
"""


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------


def test_parse_filing_content_returns_filing_document() -> None:
    from app.rag.ingestion import parse_filing_content

    doc = parse_filing_content(_SGML, ticker="acme", filing_type="10-K")

    assert doc.ticker == "ACME"
    assert doc.filing_type == "10-K"
    assert doc.company == "Acme Corp"
    assert doc.period == "20240930"
    assert doc.currency == "USD"
    assert "widget" in doc.text.lower()


def test_parse_filing_content_detects_eur_currency() -> None:
    from app.rag.ingestion import parse_filing_content

    doc = parse_filing_content(_SGML_EUR, ticker="EURO", filing_type="10-Q")

    assert doc.currency == "EUR"


def test_parse_filing_content_unknown_period() -> None:
    from app.rag.ingestion import parse_filing_content

    sgml = "<DOCUMENT>\n<TYPE>10-K\n<TEXT>\n<html><body>text</body></html>\n</TEXT>\n</DOCUMENT>"
    doc = parse_filing_content(sgml, ticker="X", filing_type="10-K")

    assert doc.period == "unknown"
    assert doc.company == ""


def test_parse_filing_content_raises_on_missing_document() -> None:
    from app.rag.ingestion import parse_filing_content

    with pytest.raises(ValueError, match="No 10-K"):
        parse_filing_content("no document here", ticker="X", filing_type="10-K")


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------


def _make_filing_doc(ticker: str = "ACME", filing_type: str = "10-K") -> object:
    from app.rag.ingestion import FilingDocument

    text = (
        "ITEM 1. BUSINESS\n\n"
        + "Acme sells widgets. " * 50
        + "\n\nITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS\n\n"
        + "Revenue grew rapidly. " * 50
    )
    return FilingDocument(
        ticker=ticker,
        filing_type=filing_type,
        accession="0001234-24-001",
        period="20240930",
        company="Acme Corp",
        text=text,
    )


def test_chunk_document_returns_chunks() -> None:
    from app.rag.chunking import chunk_document

    doc = _make_filing_doc()
    chunks = chunk_document(doc)

    assert len(chunks) > 0
    assert all(c.ticker == "ACME" for c in chunks)
    assert all(c.year == 2024 for c in chunks)


def test_chunk_document_10k_quarter_is_fy() -> None:
    from app.rag.chunking import chunk_document

    chunks = chunk_document(_make_filing_doc(filing_type="10-K"))

    assert all(c.quarter == "FY" for c in chunks)


def test_chunk_document_10q_has_quarter() -> None:
    from app.rag.chunking import chunk_document

    chunks = chunk_document(_make_filing_doc(filing_type="10-Q"))

    assert all(c.quarter.startswith("Q") for c in chunks)


def test_chunk_to_dict_has_required_keys() -> None:
    from app.rag.chunking import chunk_document

    chunks = chunk_document(_make_filing_doc())
    d = chunks[0].to_dict()

    for key in ("id", "ticker", "company", "filing_type", "year", "quarter", "section", "text"):
        assert key in d


# ---------------------------------------------------------------------------
# contextual
# ---------------------------------------------------------------------------


def test_build_deterministic_prefix_contains_metadata() -> None:
    from app.rag.chunking import chunk_document
    from app.rag.contextual import build_deterministic_prefix

    chunks = chunk_document(_make_filing_doc())
    prefix = build_deterministic_prefix(chunks[0])

    assert "ACME" in prefix
    assert "2024" in prefix
    assert "10-K" in prefix


def test_add_deterministic_context_wraps_all_chunks() -> None:
    from app.rag.chunking import chunk_document
    from app.rag.contextual import add_deterministic_context

    chunks = chunk_document(_make_filing_doc())
    contextual = add_deterministic_context(chunks)

    assert len(contextual) == len(chunks)
    assert all(c.context_prefix for c in contextual)
    assert all(c.chunk.text in c.contextualized_text for c in contextual)


# ---------------------------------------------------------------------------
# indexing — job lifecycle with mocked pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_indexing_job_sets_done_status() -> None:
    from app.services.indexing import JobStatus, job_store, run_indexing_job

    job_id = await job_store.create(ticker="ACME", filing_type="10-K")

    mock_doc = MagicMock()
    mock_chunks = [MagicMock()] * 3
    mock_contextual = [MagicMock(contextualized_text="text")] * 3

    with (
        patch("app.services.indexing.parse_filing_content", return_value=mock_doc),
        patch("app.services.indexing.chunk_document", return_value=mock_chunks),
        patch("app.services.indexing.add_deterministic_context", return_value=mock_contextual),
        patch("app.services.indexing._embed_and_upsert"),
    ):
        await run_indexing_job(job_id, "raw content", "ACME", "10-K")

    job = await job_store.get(job_id)
    assert job is not None
    assert job.status == JobStatus.DONE
    assert job.chunks_total == 3
    assert job.chunks_done == 3


@pytest.mark.asyncio
async def test_run_indexing_job_sets_failed_on_error() -> None:
    from app.services.indexing import JobStatus, job_store, run_indexing_job

    job_id = await job_store.create(ticker="FAIL", filing_type="10-Q")

    with patch(
        "app.services.indexing.parse_filing_content",
        side_effect=ValueError("bad SGML"),
    ):
        await run_indexing_job(job_id, "bad", "FAIL", "10-Q")

    job = await job_store.get(job_id)
    assert job is not None
    assert job.status == JobStatus.FAILED
    assert "bad SGML" in job.error


@pytest.mark.asyncio
async def test_run_indexing_job_noop_for_unknown_id() -> None:
    from app.services.indexing import run_indexing_job

    # Should not raise — job simply doesn't exist
    await run_indexing_job("nonexistent", "content", "X", "10-K")
