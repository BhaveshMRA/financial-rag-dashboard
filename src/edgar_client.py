"""
edgar_client.py
---------------
Handles all SEC EDGAR data fetching:
  1. XBRL structured financial data (numbers)  → EDGAR Financial Data API
  2. Narrative text (MD&A, Risk Factors)        → EDGAR filing HTML download

Companies: NVIDIA (nvda), AMD (amd), Intel (intc)
"""

import json
import re
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

# ── Company registry ──────────────────────────────────────────────────────────
COMPANIES = {
    "nvda": {"name": "NVIDIA",  "cik": "1045810",  "ticker": "NVDA"},
    "amd":  {"name": "AMD",     "cik": "2488",      "ticker": "AMD"},
    "intc": {"name": "Intel",   "cik": "50863",     "ticker": "INTC"},
}

# ── EDGAR API constants ───────────────────────────────────────────────────────
HEADERS = {"User-Agent": "financial-rag-dashboard research@example.com"}
BASE_DATA_URL = "https://data.sec.gov"
BASE_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data"

# XBRL concept keys we care about (in priority order — first match wins)
XBRL_CONCEPTS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "stockholders_equity": ["StockholdersEquity", "StockholdersEquityAttributableToParent"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _padded_cik(cik: str) -> str:
    """Zero-pad CIK to 10 digits as EDGAR expects."""
    return f"CIK{str(cik).zfill(10)}"


def _get(url: str, retries: int = 3) -> dict | str | None:
    """GET with retry + rate-limit courtesy pause."""
    for attempt in range(retries):
        try:
            time.sleep(0.15)  # EDGAR rate limit: max ~10 req/s
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            # Return JSON if possible, else raw text
            try:
                return resp.json()
            except Exception:
                return resp.text
        except requests.HTTPError as e:
            print(f"  HTTP {e.response.status_code} on attempt {attempt+1}: {url}")
            if e.response.status_code == 429:
                time.sleep(5)
        except Exception as e:
            print(f"  Error on attempt {attempt+1}: {e}")
            time.sleep(2)
    return None


# ── Part 1: XBRL financial data ───────────────────────────────────────────────

def fetch_xbrl_facts(cik: str) -> dict | None:
    """
    Fetch all XBRL company facts for a given CIK.
    Returns the raw JSON dict or None on failure.
    """
    url = f"{BASE_DATA_URL}/api/xbrl/companyfacts/{_padded_cik(cik)}.json"
    print(f"  Fetching XBRL facts from: {url}")
    return _get(url)


def _extract_annual_series(facts_json: dict, concept_keys: list[str]) -> list[dict]:
    """
    Given the XBRL facts JSON, extract annual (10-K) USD values.
    We check all concept keys in the priority list, and select the one
    that has the most recent 'end' date (or the most entries if dates match).
    This handles cases where a company switches concepts over time (e.g. NVIDIA revenue).

    Returns a list of dicts: [{end, val, fy, filed, accn}, ...]
    sorted by fiscal year end date.
    """
    us_gaap = facts_json.get("facts", {}).get("us-gaap", {})

    best_series = []
    best_max_date = ""

    for key in concept_keys:
        if key not in us_gaap:
            continue
        usd_entries = us_gaap[key].get("units", {}).get("USD", [])
        # Keep only annual 10-K entries with full-year period flag
        annual = [
            x for x in usd_entries
            if x.get("form") == "10-K" and x.get("fp") == "FY"
        ]
        if not annual:
            continue

        # Detect restatements: same 'end' date with multiple 'filed' dates
        by_end: dict[str, list] = {}
        for entry in annual:
            by_end.setdefault(entry["end"], []).append(entry)

        result = []
        for end_date, entries in by_end.items():
            entries_sorted = sorted(entries, key=lambda x: x["filed"])
            chosen = entries_sorted[-1]  # most recently filed = authoritative
            row = {
                "end": end_date,
                "val": chosen["val"],
                "fy": chosen.get("fy"),
                "filed": chosen["filed"],
                "accn": chosen["accn"],
                "concept": key,
                "restated": len(entries) > 1,           # flag restatement
                "prior_val": entries_sorted[0]["val"] if len(entries) > 1 else None,
            }
            result.append(row)

        if result:
            max_date = max(x["end"] for x in result)
            if max_date > best_max_date or (max_date == best_max_date and len(result) > len(best_series)):
                best_max_date = max_date
                best_series = result

    return sorted(best_series, key=lambda x: x["end"])


def build_financials(cik: str, company_key: str) -> dict:
    """
    Fetch and structure all financial metrics for one company.
    Returns dict keyed by metric name, each containing a time series.
    Also records the XBRL concept used (for traceability).
    """
    print(f"\nFetching XBRL data for {COMPANIES[company_key]['name']}...")
    facts = fetch_xbrl_facts(cik)
    if not facts:
        print(f"  FAILED to fetch XBRL data for CIK {cik}")
        return {}

    financials = {
        "company": COMPANIES[company_key]["name"],
        "ticker": COMPANIES[company_key]["ticker"],
        "cik": cik,
        "metrics": {},
    }

    for metric_name, concept_keys in XBRL_CONCEPTS.items():
        series = _extract_annual_series(facts, concept_keys)
        if series:
            # Record which concept was actually found (traceability)
            found_concept = series[0]["concept"]
            financials["metrics"][metric_name] = {
                "series": series,
                "xbrl_concept": found_concept,
                "source": f"SEC EDGAR XBRL — us-gaap/{found_concept}",
            }
            print(f"  ✓ {metric_name} ({found_concept}): {len(series)} annual data points")
        else:
            print(f"  ✗ {metric_name}: not found in XBRL (will be absent from analysis)")

    return financials


# ── Part 2: Filing text (MD&A + Risk Factors) ─────────────────────────────────

def get_10k_filing_list(cik: str, max_filings: int = 3) -> list[dict]:
    """
    Get the most recent N 10-K filings for a company via EDGAR submissions API.
    Returns list of dicts: [{accession, filed, report_date, primary_document}, ...]
    """
    url = f"{BASE_DATA_URL}/submissions/{_padded_cik(cik)}.json"
    data = _get(url)
    if not data:
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filed_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    primary_docs = recent.get("primaryDocument", [])

    results = []
    for i, form in enumerate(forms):
        if form == "10-K":
            results.append({
                "accession": accessions[i],
                "filed": filed_dates[i],
                "report_date": report_dates[i],
                "primary_document": primary_docs[i],
            })
        if len(results) >= max_filings:
            break

    print(f"  Found {len(results)} 10-K filings for CIK {cik}")
    return results


def _fetch_filing_html(cik: str, accession: str, primary_doc: str) -> str | None:
    """Download the primary 10-K HTML document."""
    acc_no_dashes = accession.replace("-", "")
    url = f"{BASE_ARCHIVE_URL}/{cik}/{acc_no_dashes}/{primary_doc}"
    print(f"  Downloading filing: {url}")
    html = _get(url)
    return html if isinstance(html, str) else None


# Minimum character length for a section to be considered real content (not TOC)
_MIN_SECTION_CHARS = 500


def _extract_section(html: str, item_start: str, item_end: str,
                     fallback_starts: list[str] | None = None) -> str:
    """
    Extract a named section from 10-K HTML.
    Uses html.parser (more forgiving on iXBRL than lxml).
    Picks the LONGEST match — skips Table of Contents entries which
    are tiny and appear before the actual section body.

    Some filers (e.g., Intel) omit the "Item 1A" label from the section
    body and only include it in the TOC. For those cases, pass
    `fallback_starts` — additional patterns tried if the primary pattern
    yields only short (TOC-like) matches.
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    end_pattern = re.compile(item_end, re.IGNORECASE)

    def _best_from_pattern(pat_str: str) -> str:
        pattern = re.compile(pat_str, re.IGNORECASE)
        best = ""
        for match in pattern.finditer(text):
            end_match = end_pattern.search(text, match.end())
            section = (
                text[match.start(): end_match.start()]
                if end_match
                else text[match.start(): match.start() + 20000]
            )
            if len(section) > len(best):
                best = section
        return best.strip()

    # Try primary pattern first
    best = _best_from_pattern(item_start)

    # If the best result looks like a TOC entry (very short), try fallbacks
    if len(best) < _MIN_SECTION_CHARS and fallback_starts:
        for fallback in fallback_starts:
            candidate = _best_from_pattern(fallback)
            if len(candidate) > len(best):
                best = candidate
            if len(best) >= _MIN_SECTION_CHARS:
                break

    return best

def chunk_text(text: str, company: str, section: str,
               filing_meta: dict, chunk_size: int = 1000,
               overlap: int = 150) -> list[dict]:
    """
    Split text into overlapping chunks and attach metadata for citation.
    Each chunk = {text, metadata: {company, section, filed, accn, chunk_id}}
    """
    words = text.split()
    chunks = []
    i = 0
    chunk_id = 0

    while i < len(words):
        chunk_words = words[i: i + chunk_size]
        chunk_text_str = " ".join(chunk_words)
        if len(chunk_text_str.strip()) > 100:  # skip tiny fragments
            chunks.append({
                "text": chunk_text_str,
                "metadata": {
                    "company": company,
                    "section": section,
                    "filed": filing_meta["filed"],
                    "report_date": filing_meta["report_date"],
                    "accession": filing_meta["accession"],
                    "chunk_id": chunk_id,
                    "source_label": (
                        f"{company} 10-K (filed {filing_meta['filed']}) — {section}"
                    ),
                },
            })
            chunk_id += 1
        i += chunk_size - overlap

    return chunks


def fetch_narrative_chunks(company_key: str,
                           max_filings: int = 2) -> list[dict]:
    """
    For a company, fetch the last N 10-Ks, extract MD&A and Risk Factors,
    chunk them, and return all chunks with metadata.
    """
    info = COMPANIES[company_key]
    cik = info["cik"]
    company_name = info["name"]
    print(f"\nFetching narrative text for {company_name}...")

    filings = get_10k_filing_list(cik, max_filings=max_filings)
    all_chunks = []

    for filing in filings:
        html = _fetch_filing_html(cik, filing["accession"], filing["primary_document"])
        if not html:
            print(f"  Skipping filing {filing['accession']} — download failed")
            continue

        # Extract MD&A (Item 7)
        mda_text = _extract_section(
            html,
            item_start=r"Item\s+7[.\s]+Management.{0,40}Discussion",
            item_end=r"Item\s+7A|Item\s+8",
        )
        if mda_text:
            chunks = chunk_text(mda_text, company_name, "MD&A", filing)
            all_chunks.extend(chunks)
            print(f"  ✓ MD&A: {len(chunks)} chunks from {filing['filed']}")
        else:
            print(f"  ✗ MD&A not extracted from {filing['filed']}")

        # Extract Risk Factors (Item 1A)
        # Fallback handles Intel-style filings where "Item 1A" only appears in
        # the TOC and the body section opens with a bare "Risk Factors" heading.
        risk_text = _extract_section(
            html,
            item_start=r"Item\s+1A[.\s]+Risk\s+Factor",
            item_end=r"Item\s+1[BC]|Item\s+2",
            fallback_starts=[
                # Intel / some filers: narrative opens with standalone heading
                r"Risk\s+Factors\s+The\s+following\s+(?:summarizes|describes|sets\s+forth)",
                r"Risk\s+Factors\s+(?:The\s+)?(?:material|following|below)",
                # Generic fallback: "RISK FACTORS" as an all-caps standalone heading
                r"RISK\s+FACTORS\s+[A-Z]",
            ],
        )
        if risk_text:
            chunks = chunk_text(risk_text, company_name, "Risk Factors", filing)
            all_chunks.extend(chunks)
            print(f"  ✓ Risk Factors: {len(chunks)} chunks from {filing['filed']}")
        else:
            print(f"  ✗ Risk Factors not extracted from {filing['filed']}")

    return all_chunks
