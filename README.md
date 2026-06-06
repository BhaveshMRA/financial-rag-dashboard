```markdown
# Financial RAG Dashboard — NVIDIA · AMD · Intel

Comparative financial analysis dashboard over SEC 10-K filings with
a cited RAG chatbot built on LangGraph.

**Live Application:** [financial-rag-dashboard.streamlit.app](https://financial-rag-dashboard.streamlit.app/)

**Built for:** CustomerInsights.AI AI Engineer Intern Take-Home Exercise  
**Companies:** NVIDIA (NVDA), AMD (AMD), Intel (INTC)  
**Data source:** SEC EDGAR (public filings, no authentication required)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in your credentials
cp .env.example .env
# Edit .env: add your OLLAMA_API_KEY and/or GROQ_API_KEY

# 3. Run the data pipeline (one-time, ~5-10 min)
python data_pipeline.py

# 4. Launch the dashboard
streamlit run app.py

# 5. (Optional) Run evaluation
python eval/run_eval.py
```

---

## Architecture

```
data_pipeline.py
    ├── src/edgar_client.py   → SEC EDGAR XBRL API (structured numbers)
    │                            + 10-K HTML download (MD&A + Risk Factors)
    └── src/rag_chain.py      → Embed chunks → store in ChromaDB

app.py (Streamlit — 4 tabs)
    ├── Dashboard
    │     ├── src/metrics.py  → Derived metrics (margins, YoY, D/E, FCF)
    │     └── src/charts.py   → Plotly comparative visualizations
    ├── RAG Chat
    │     └── src/rag_chain.py → LangGraph pipeline (see below)
    ├── Data Conflicts
    │     └── src/metrics.py  → Restatement detection
    └── Evaluation
          └── eval/run_eval.py → Labeled question results
```

### LangGraph RAG Pipeline

```
START
  └─→ rewrite_query    LLM rewrites the question into a retrieval-optimized query
        └─→ retrieve         ChromaDB semantic search (top-5 chunks)
              └─→ gate             Two-stage relevance check:
                                    1. Cosine distance threshold (no LLM cost)
                                    2. LLM yes/no on best 2 chunks
                    ├─→ generate   Cited answer with source labels   → END
                    └─→ refuse     Explicit refusal, no fabrication  → END
```

State is a `TypedDict` — every field is inspectable at each node.
The conditional edge at `gate` is what makes this agentic rather than a
simple single-shot retrieval chain.

### Two data paths (intentionally separate)

- **Structured numbers** → EDGAR XBRL API → JSON → `metrics.py` → charts  
- **Narrative text** → 10-K HTML → BeautifulSoup → chunked → ChromaDB → RAG

This separation matters: XBRL is authoritative for numbers (zero LLM
hallucination risk in the numbers layer). The LLM only touches narrative
text and derived metric explanations.

---

## Design Decisions

**Why NVIDIA, AMD, Intel?**  
Direct competitors in the AI chip / semiconductor market. NVIDIA's explosive
revenue growth vs. AMD catching up vs. Intel's structural challenges creates
a rich, verifiable financial story with clear quant-to-narrative linkage.
All three have clean multi-year XBRL data on EDGAR.

**Why EDGAR XBRL for numbers, not PDF parsing?**  
EDGAR's financial data API returns clean JSON with accession numbers baked in
— every value is traceable to its source filing. No OCR, no table extraction,
no risk of misreading a number. PDF parsing was considered and rejected
because accuracy of the numbers layer is the highest-priority requirement.

**Why LangGraph instead of a flat LangChain chain?**  
LangGraph's `StateGraph` makes the agentic flow explicit — each node has a
clear responsibility, the conditional edge at `gate` is visible in the code,
and the state is fully inspectable at every step. A flat LCEL chain hides
this structure. LangGraph also maps directly to prior project experience
(CS 584 agentic confidence calibration work).

**Why raw httpx for the LLM call instead of ChatOllama?**  
LangChain's `ChatOllama` does not cleanly expose the `Authorization` header
or `follow_redirects=True`, both of which are required for Ollama Cloud.
Dropping to raw `httpx` in `src/llm_client.py` gives full control with an
automatic fallback to Groq if Ollama fails. Wrapped as a plain function
rather than a LangChain runnable to keep it simple and debuggable.

**Temperature: 0.1 throughout**  
Financial figures must be precise. Low temperature reduces paraphrasing
of numbers and keeps the model closer to what the context actually says.

---

## Document Sources

All documents are public SEC filings. No login required.

| Company | CIK    | Filings used        | EDGAR link |
|---------|--------|---------------------|------------|
| NVIDIA  | 1045810 | 10-K FY2023, FY2024 | [EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=1045810&type=10-K) |
| AMD     | 2488    | 10-K FY2022, FY2023 | [EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=2488&type=10-K) |
| Intel   | 50863   | 10-K FY2022, FY2023 | [EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=50863&type=10-K) |

XBRL financial data API (no login):  
`https://data.sec.gov/api/xbrl/companyfacts/CIK{padded_cik}.json`

---

## File Structure

```
financial-rag-dashboard/
├── app.py                  # Streamlit app (4 tabs)
├── data_pipeline.py        # One-time data fetch + embed script
├── src/
│   ├── edgar_client.py     # EDGAR XBRL API + 10-K HTML download
│   ├── llm_client.py       # Ollama Cloud + Groq (raw httpx, auto-fallback)
│   ├── rag_chain.py        # LangGraph RAG pipeline
│   ├── metrics.py          # Derived metrics with source attribution
│   └── charts.py           # Plotly visualizations
├── data/
│   └── financials/         # Cached XBRL JSON per company
├── vectorstore/            # ChromaDB persistent storage
├── eval/
│   ├── labeled_questions.json   # 15 labeled questions (5 unanswerable)
│   └── run_eval.py              # Evaluation runner
├── .env.example
├── requirements.txt
├── README.md
└── WRITEUP.md
```

---

## Evaluation Results

Run `python eval/run_eval.py` to generate `eval/eval_results.json`.  
Results are displayed in the Evaluation tab of the Streamlit app.

See `WRITEUP.md` for interpretation and honest assessment of hallucination rate.
```