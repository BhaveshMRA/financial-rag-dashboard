"""
data_pipeline.py
----------------
Run this ONCE before launching the Streamlit app.

What it does:
  1. Fetches XBRL financial data for NVIDIA, AMD, Intel from EDGAR
  2. Saves structured metrics to data/financials/{company}.json
  3. Downloads 10-K HTML filings, extracts MD&A + Risk Factors
  4. Chunks narrative text and embeds into ChromaDB vectorstore

Usage:
  python data_pipeline.py

Takes ~5-10 minutes depending on EDGAR response times.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.edgar_client import (
    COMPANIES,
    build_financials,
    fetch_narrative_chunks,
)
from src.rag_chain import add_chunks_to_vectorstore

FINANCIALS_DIR = Path("data/financials")
FINANCIALS_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline():
    all_financials = {}

    # ── Step 1: XBRL financial data ───────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Fetching XBRL financial data from SEC EDGAR")
    print("=" * 60)

    for company_key, info in COMPANIES.items():
        out_path = FINANCIALS_DIR / f"{company_key}.json"

        if out_path.exists():
            print(f"\n[{info['name']}] Found cached financials at {out_path}, skipping fetch.")
            with open(out_path) as f:
                all_financials[company_key] = json.load(f)
            continue

        financials = build_financials(info["cik"], company_key)
        if financials:
            all_financials[company_key] = financials
            with open(out_path, "w") as f:
                json.dump(financials, f, indent=2, default=str)
            print(f"  Saved → {out_path}")
        else:
            print(f"  FAILED for {info['name']}")

    # ── Step 2: Narrative text (MD&A + Risk Factors) ──────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Fetching 10-K narrative text and embedding")
    print("=" * 60)
    print("Note: downloading full 10-K HTML files — this takes a few minutes.")

    all_chunks = []
    for company_key in COMPANIES:
        chunks = fetch_narrative_chunks(company_key, max_filings=3)
        all_chunks.extend(chunks)
        print(f"  {COMPANIES[company_key]['name']}: {len(chunks)} total chunks")

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    if all_chunks:
        print("Embedding and storing in ChromaDB...")
        add_chunks_to_vectorstore(all_chunks)
        print("Vectorstore ready.")
    else:
        print("WARNING: No chunks generated. Check EDGAR connectivity.")

    # ── Step 3: Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    for company_key, fin in all_financials.items():
        metrics_found = list(fin.get("metrics", {}).keys())
        print(f"  {fin['company']}: {len(metrics_found)} metrics — {metrics_found}")

    print("\nRun 'streamlit run app.py' to launch the dashboard.")


if __name__ == "__main__":
    run_pipeline()
