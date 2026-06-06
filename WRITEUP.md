# WRITEUP.md

## Architecture

The system uses two separate data paths, intentionally:

**Structured numbers (EDGAR XBRL API → metrics.py):**  
All financial figures are fetched directly from SEC EDGAR's
`/api/xbrl/companyfacts/{CIK}.json` endpoint, which returns clean JSON for
every us-gaap XBRL concept. This means derived metrics (margins, YoY growth,
D/E ratio, FCF) are computed from authoritative structured data — no PDF
parsing, no LLM inference involved in the numbers themselves. Every value
carries its source accession number.

**Narrative text (10-K HTML → ChromaDB → RAG):**  
The MD&A and Risk Factors sections are downloaded from EDGAR, chunked at
~1000 words with 150-word overlap, embedded with `all-MiniLM-L6-v2`, and
stored in ChromaDB. The RAG chain runs query rewriting before retrieval
and a two-stage relevance gate before answering.

**Key files:**
- `src/edgar_client.py` — all EDGAR API and HTML download logic
- `src/metrics.py` — derived metric computations with source attribution
- `src/rag_chain.py` — query rewrite → retrieve → gate → answer pipeline
- `src/llm_client.py` — Ollama Cloud + Groq wrapper (raw httpx, not ChatOllama)
- `data_pipeline.py` — one-time data fetch and embedding script
- `app.py` — Streamlit dashboard (Dashboard + Chat + Conflicts + Eval tabs)

---

## Hallucination Rate and Executive Trust

**Measured hallucination rate:** 25% (1 out of 4 unanswerable questions received a fabricated answer).

**Would I trust this in front of an executive?**  
No, I would not put this dashboard in front of an executive yet.

**What I fix first:**  
The first fix is improving narrative chunk extraction — NVIDIA and AMD returned 0 chunks from their 10-K HTML filings due to iXBRL section header matching issues, leaving the RAG with almost no context to work with. Without fixing that, any narrative question about those two companies either hallucinates or false-refuses.

---

## Most Interesting Insight

NVIDIA's operating margin expanded from 37.31% (FY2022) to 54.12% (FY2024), and reached 60.38% in FY2026, driven by the massive demand for AI data center compute. In contrast, Intel's operating margin compressed from 3.70% (FY2022) to -21.99% (FY2024) due to heavy investments in its foundry strategy and loss of market share, before improving to -4.19% in FY2025. This creates a massive 76.11 percentage point spread between NVIDIA and Intel in FY2024, highlighting the divergent paths of the two chipmakers during the AI infrastructure boom.

Confidence: high — both margin figures are computed directly from XBRL data and cross-checked against the raw income statement values in the filings.

---

## One Failure I Found and Diagnosed

Q5 — "Compare the free cash flow of NVIDIA, AMD, and Intel for fiscal year 2023" — was marked answerable but the system refused it (🚫).

**Root cause:** FCF is computed from XBRL data (Operating Cash Flow − CapEx) and stored in structured JSON, but the RAG chatbot only searches the text chunk vectorstore. The two data paths — XBRL numbers and narrative text — are correctly separated by design, but the chatbot has no bridge to the computed metrics layer.

**Fix:** Add a tool-use node to the LangGraph that can query the XBRL JSON directly when the question is about a financial figure.

---

## How I Used AI Tools

Claude (claude.ai) was used throughout this exercise to:
- Scaffold the initial project structure and file organization
- Write boilerplate for the EDGAR API client and LangChain chain
- Suggest the two-data-path architecture (XBRL + narrative separate)

**One place where I overrode the AI's suggestion:**  
Claude initially suggested using LangChain's `ChatOllama` class with a
`base_url` override for Ollama Cloud. I replaced this with a raw `httpx`
call because `ChatOllama` doesn't expose the `Authorization` header or
`follow_redirects=True` cleanly — both of which are required for
Ollama Cloud to work. The working `src/llm_client.py` is the result of
that override.

---

## Where the Framework Helped vs. Where I Fought It

**LangChain helped:**  
The `RunnableLambda` + LCEL pattern made the query-rewrite → retrieve →
gate → answer pipeline composable and readable. The `HuggingFaceEmbeddings`
class and ChromaDB integration worked cleanly out of the box.

**Where I dropped down to raw API:**  
The LLM call itself (see above). `ChatOllama`'s auth support is incomplete
for cloud-hosted Ollama. Dropping to raw `httpx` gave me full control over
redirect handling and auth headers at the cost of losing some LangChain
callback/tracing support — an acceptable tradeoff for reliability.
