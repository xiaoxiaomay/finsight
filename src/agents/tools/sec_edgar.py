"""SEC EDGAR filing download and parsing tool.

Downloads 10-K and 10-Q filings from SEC EDGAR (free, no API key needed).
Parses HTML filings into structured text with section detection.

SEC EDGAR API Requirements:
- User-Agent header with company name and email
- Max 10 requests per second
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup

from src.config.logging_config import get_logger

logger = get_logger("tools.sec_edgar")

EDGAR_BASE = "https://www.sec.gov"
EDGAR_DATA = "https://data.sec.gov"
TICKERS_URL = f"{EDGAR_BASE}/files/company_tickers.json"

HEADERS = {
    "User-Agent": "FinSight/1.0 (research@finsight.dev)",
    "Accept-Encoding": "gzip, deflate",
}

# Standard 10-K section patterns (Item numbers)
SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"(?:item\s*1a[\.\:\s])", "Risk Factors"),
    (r"(?:item\s*1b[\.\:\s])", "Unresolved Staff Comments"),
    (r"(?:item\s*1[\.\:\s])", "Business"),
    (r"(?:item\s*2[\.\:\s])", "Properties"),
    (r"(?:item\s*3[\.\:\s])", "Legal Proceedings"),
    (r"(?:item\s*5[\.\:\s])", "Market for Common Equity"),
    (r"(?:item\s*7a[\.\:\s])", "Market Risk Disclosures"),
    (r"(?:item\s*7[\.\:\s])", "MD&A"),
    (r"(?:item\s*8[\.\:\s])", "Financial Statements"),
    (r"(?:item\s*9a[\.\:\s])", "Controls and Procedures"),
    (r"(?:item\s*9[\.\:\s])", "Accountant Disagreements"),
    (r"(?:item\s*10[\.\:\s])", "Directors and Officers"),
    (r"(?:item\s*11[\.\:\s])", "Executive Compensation"),
    (r"(?:item\s*12[\.\:\s])", "Security Ownership"),
    (r"(?:item\s*13[\.\:\s])", "Related Transactions"),
    (r"(?:item\s*14[\.\:\s])", "Accountant Fees"),
    (r"(?:item\s*15[\.\:\s])", "Exhibits"),
]

_last_request_time = 0.0


@dataclass
class EdgarFiling:
    """Parsed SEC filing."""

    accession_number: str
    form_type: str
    filing_date: str
    primary_doc_url: str
    company_name: str = ""
    cik: str = ""


@dataclass
class FilingDocument:
    """Downloaded and parsed filing document."""

    filing: EdgarFiling
    raw_html: str = ""
    full_text: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    tables: list[str] = field(default_factory=list)


def _rate_limit() -> None:
    """Enforce SEC's 10 req/sec limit."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < 0.15:
        time.sleep(0.15 - elapsed)
    _last_request_time = time.time()


def get_cik(symbol: str) -> str | None:
    """Look up CIK number for a ticker symbol.

    Uses SEC's company_tickers.json mapping file.
    """
    _rate_limit()
    try:
        resp = httpx.get(TICKERS_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        symbol_upper = symbol.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == symbol_upper:
                return str(entry["cik_str"])
    except Exception:
        logger.warning("cik_lookup_failed", symbol=symbol)
    return None


def get_filings(
    symbol: str,
    form_type: str = "10-K",
    count: int = 5,
) -> list[EdgarFiling]:
    """Get recent filings for a company from SEC EDGAR.

    Args:
        symbol: Ticker symbol (e.g., 'AAPL').
        form_type: Filing type ('10-K', '10-Q').
        count: Maximum number of filings to return.

    Returns:
        List of EdgarFiling metadata objects.
    """
    cik = get_cik(symbol)
    if not cik:
        logger.warning("no_cik_found", symbol=symbol)
        return []

    # Get submissions JSON
    cik_padded = cik.zfill(10)
    url = f"{EDGAR_DATA}/submissions/CIK{cik_padded}.json"

    _rate_limit()
    try:
        resp = httpx.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.warning("submissions_fetch_failed", symbol=symbol, cik=cik)
        return []

    company_name = data.get("name", symbol)
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    filings = []
    for i, form in enumerate(forms):
        if form == form_type and len(filings) < count:
            acc = accessions[i].replace("-", "")
            doc_url = (
                f"{EDGAR_BASE}/Archives/edgar/data/{cik}/{acc}/{primary_docs[i]}"
            )
            filings.append(EdgarFiling(
                accession_number=accessions[i],
                form_type=form,
                filing_date=dates[i],
                primary_doc_url=doc_url,
                company_name=company_name,
                cik=cik,
            ))

    logger.info("filings_found", symbol=symbol, form_type=form_type, count=len(filings))
    return filings


def download_filing(filing: EdgarFiling) -> FilingDocument:
    """Download and parse a single SEC filing.

    Args:
        filing: Filing metadata from get_filings().

    Returns:
        FilingDocument with parsed text and sections.
    """
    _rate_limit()
    try:
        resp = httpx.get(filing.primary_doc_url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        raw_html = resp.text
    except Exception:
        logger.warning("filing_download_failed", url=filing.primary_doc_url)
        return FilingDocument(filing=filing)

    # Parse HTML
    full_text, tables = parse_filing_html(raw_html)
    sections = detect_sections(full_text)

    doc = FilingDocument(
        filing=filing,
        raw_html=raw_html,
        full_text=full_text,
        sections=sections,
        tables=tables,
    )

    logger.info(
        "filing_parsed",
        accession=filing.accession_number,
        text_length=len(full_text),
        sections_found=len(sections),
        tables_found=len(tables),
    )
    return doc


def parse_filing_html(html: str) -> tuple[str, list[str]]:
    """Parse SEC filing HTML into clean text.

    Returns:
        Tuple of (full_text, list_of_table_texts).
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove script, style, and hidden elements
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()

    # Extract tables separately (important for financial data)
    tables = []
    for table in soup.find_all("table"):
        table_text = _parse_table(table)
        if table_text and len(table_text) > 10:
            tables.append(table_text)

    # Get full text
    text = soup.get_text(separator="\n", strip=True)

    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text, tables


def _parse_table(table_tag) -> str:
    """Convert HTML table to readable text format."""
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            cell_text = td.get_text(strip=True)
            if cell_text:
                cells.append(cell_text)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def detect_sections(text: str) -> dict[str, str]:
    """Detect standard 10-K sections in filing text.

    Looks for 'Item 1', 'Item 1A', etc. patterns and extracts
    the text between them as named sections.

    Returns:
        Dict mapping section_name → section_text.
    """
    # Find all section boundaries
    boundaries: list[tuple[int, str]] = []
    text_lower = text.lower()

    for pattern, name in SECTION_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            boundaries.append((match.start(), name))

    if not boundaries:
        return {"full_document": text}

    # Sort by position
    boundaries.sort(key=lambda x: x[0])

    # Extract text between boundaries
    sections = {}
    for i, (start, name) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        section_text = text[start:end].strip()

        # Only keep substantial sections (>100 chars)
        if len(section_text) > 100:
            sections[name] = section_text

    return sections
