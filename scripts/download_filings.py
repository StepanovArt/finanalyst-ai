"""
Download SEC EDGAR filings (10-K, 10-Q) for selected companies.

Usage:
    uv run --extra data python scripts/download_filings.py

Downloads to: data/filings/{ticker}/{filing_type}/
SEC requires a User-Agent header with name + email (fair use policy).
"""

from pathlib import Path

from sec_edgar_downloader import Downloader

DOWNLOAD_DIR = Path(__file__).parent.parent / "data" / "filings"

# 5 companies × (1×10-K + 4×10-Q) = 25 filings
COMPANIES = ["AAPL", "MSFT", "NVDA", "META", "AMZN"]

FILINGS = [
    ("10-K", 2),  # last 2 annual reports (~2 years)
    ("10-Q", 6),  # last 6 quarterly reports (~1.5 years)
]


def main() -> None:
    dl = Downloader(
        company_name="FinAnalyst-AI",
        email_address="artem852460@gmail.com",
        download_folder=DOWNLOAD_DIR,
    )

    total = 0
    for ticker in COMPANIES:
        for filing_type, limit in FILINGS:
            print(f"[{ticker}] Downloading {limit}x {filing_type}...")
            dl.get(filing_type, ticker, limit=limit)
            total += limit

    print(f"\nDone. Downloaded up to {total} filings to {DOWNLOAD_DIR}")
    print("Actual count (some filings may be unavailable):")

    for path in sorted(DOWNLOAD_DIR.rglob("full-submission.txt")):
        print(f"  {path.relative_to(DOWNLOAD_DIR)}")


if __name__ == "__main__":
    main()
