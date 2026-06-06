"""
rag_chain.py
------------
LangGraph-based RAG pipeline with explicit state and conditional edges.

Graph structure:
  START
    └─→ rewrite_query   (query optimization before retrieval)
          └─→ retrieve        (ChromaDB semantic search)
                └─→ gate            (relevance check — is context sufficient?)
                      ├─→ generate  (if YES → cited answer)
                      └─→ refuse    (if NO  → explicit refusal, no hallucination)
                            └─→ END

This satisfies the "agentic behavior beyond single-shot retrieval" requirement.
State is explicit and inspectable at every node.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import chromadb
from pathlib import Path
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from src.llm_client import call_llm

# ── Paths ─────────────────────────────────────────────────────────────────────
VECTORSTORE_DIR = str(Path(__file__).parent.parent / "vectorstore")
COLLECTION_NAME = "financial_filings"
RELEVANCE_THRESHOLD = 0.65


# ── Singletons ────────────────────────────────────────────────────────────────
_embedder = None
_collection = None

def get_embedder() -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedder

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── State definition ──────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question:        str
    company_filter:  Optional[str]
    rewritten_query: str
    chunks:          list
    sufficient:      bool
    answer:          str
    refused:         bool
    sources:         list[str]
    conflicts:       list[str]


# ── Vectorstore ingest (called from data_pipeline.py) ─────────────────────────

def add_chunks_to_vectorstore(chunks: list[dict]) -> None:
    """Embed and store text chunks in ChromaDB."""
    if not chunks:
        return

    collection = get_collection()
    embedder   = get_embedder()
    texts      = [c["text"] for c in chunks]
    metadatas  = [c["metadata"] for c in chunks]
    ids = [
        f"{m['company'].lower()}_{m['section'].lower().replace(' ','_')}"
        f"_{m['filed']}_{m['chunk_id']}"
        for m in metadatas
    ]

    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch_embeddings = embedder.embed_documents(texts[i:i+batch_size])
        collection.upsert(
            ids=ids[i:i+batch_size],
            embeddings=batch_embeddings,
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
        )
    print(f"  Stored {len(texts)} chunks in ChromaDB")


# ── Node 1: Query rewriting ───────────────────────────────────────────────────

REWRITE_PROMPT = SystemMessage(content="""You are a financial analyst query optimizer.
Rewrite the user's question into a precise retrieval query for SEC 10-K filings.
Focus on financial concepts, company names, and time periods.
Return ONLY the rewritten query. No explanation, no preamble.""")

def rewrite_query(state: RAGState) -> RAGState:
    messages = [
        REWRITE_PROMPT,
        HumanMessage(content=f"Original question: {state['question']}")
    ]
    rewritten = call_llm(messages, temperature=0.0).strip()
    return {**state, "rewritten_query": rewritten}


# ── Node 2: Retrieval ─────────────────────────────────────────────────────────

def detect_company_from_text(text: str) -> str | None:
    """Helper to auto-detect company targets from query text."""
    text_lower = text.lower()
    if "intel corporation" in text_lower or "intel's" in text_lower or "intel 10-k" in text_lower:
        return "Intel"
    if "nvidia corporation" in text_lower or "nvidia's" in text_lower or "nvidia 10-k" in text_lower:
        return "NVIDIA"
    if "advanced micro devices" in text_lower or "amd's" in text_lower or "amd 10-k" in text_lower:
        return "AMD"

    has_nvda = "nvidia" in text_lower or "nvda" in text_lower
    has_amd = "amd" in text_lower
    has_intel = "intel" in text_lower or "intc" in text_lower

    if has_nvda and not has_amd and not has_intel:
        return "NVIDIA"
    if has_amd and not has_nvda and not has_intel:
        return "AMD"
    if has_intel and not has_nvda and not has_amd:
        return "Intel"
    return None


import json

def load_financials_for_rag() -> dict:
    """Load cached financials for structured context injection."""
    financials_dir = Path(__file__).parent.parent / "data/financials"
    all_fin = {}
    for key in ["nvda", "amd", "intc"]:
        path = financials_dir / f"{key}.json"
        if path.exists():
            with open(path) as f:
                all_fin[key] = json.load(f)
    return all_fin


def generate_financials_context_chunk(all_fin: dict, company_filter: str | None = None) -> str:
    """Format structured financial data facts into a markdown context chunk for RAG."""
    lines = ["STRUCTURED FINANCIAL DATA FROM SEC EDGAR XBRL (us-gaap):"]

    companies_to_show = ["NVIDIA", "AMD", "Intel"]
    if company_filter:
        companies_to_show = [company_filter]

    for comp in companies_to_show:
        comp_key = None
        for k, v in {"nvda": "NVIDIA", "amd": "AMD", "intc": "Intel"}.items():
            if v.lower() == comp.lower():
                comp_key = k
                break
        if not comp_key or comp_key not in all_fin:
            continue

        fin = all_fin[comp_key]
        lines.append(f"\nCompany: {fin['company']} (Ticker: {fin['ticker']})")
        lines.append("| Metric | Period End | FY | Value |")
        lines.append("|---|---|---|---|")

        # Raw metrics
        for metric_name in ["revenue", "gross_profit", "operating_income", "net_income", "rd_expense", "long_term_debt", "stockholders_equity", "operating_cash_flow", "capex"]:
            metric_data = fin.get("metrics", {}).get(metric_name, {})
            series = metric_data.get("series", [])
            for entry in series:
                val = entry["val"]
                fy_val = entry["end"][:4]
                if abs(val) >= 1e9:
                    val_str = f"${val/1e9:.3f} Billion"
                else:
                    val_str = f"${val/1e6:.3f} Million"
                lines.append(f"| {metric_name} | {entry['end']} | {fy_val} | {val_str} |")

        # Computed metrics
        from src.metrics import compute_margins, compute_free_cash_flow, compute_leverage

        margins_df = compute_margins(fin)
        if not margins_df.empty:
            for idx, row in margins_df.iterrows():
                y_str = idx.strftime("%Y-%m-%d")
                fy_val = y_str[:4]
                lines.append(f"| gross_margin_pct | {y_str} | {fy_val} | {row.get('gross_margin_pct', 'N/A')}% |")
                lines.append(f"| operating_margin_pct | {y_str} | {fy_val} | {row.get('operating_margin_pct', 'N/A')}% |")
                lines.append(f"| net_margin_pct | {y_str} | {fy_val} | {row.get('net_margin_pct', 'N/A')}% |")

        fcf_df = compute_free_cash_flow(fin)
        if not fcf_df.empty:
            for idx, row in fcf_df.iterrows():
                y_str = idx.strftime("%Y-%m-%d")
                fy_val = y_str[:4]
                fcf_val = row.get("free_cash_flow")
                if fcf_val is not None:
                    if abs(fcf_val) >= 1e9:
                        fcf_str = f"${fcf_val/1e9:.3f} Billion"
                    else:
                        fcf_str = f"${fcf_val/1e6:.3f} Million"
                    lines.append(f"| free_cash_flow | {y_str} | {fy_val} | {fcf_str} |")

        leverage_df = compute_leverage(fin)
        if not leverage_df.empty:
            for idx, row in leverage_df.iterrows():
                y_str = idx.strftime("%Y-%m-%d")
                fy_val = y_str[:4]
                de_val = row.get("debt_to_equity")
                de_str = f"{de_val:.3f}x" if de_val is not None else "N/A"
                lines.append(f"| debt_to_equity | {y_str} | {fy_val} | {de_str} |")

    return "\n".join(lines)


def retrieve(state: RAGState) -> RAGState:
    """Retrieve top-5 relevant chunks from ChromaDB and inject structured financial metrics context."""
    collection = get_collection()
    embedder   = get_embedder()

    query_embedding = embedder.embed_query(state["rewritten_query"])

    # Auto-detect company filter if not explicitly provided
    comp_filter = state.get("company_filter")
    if not comp_filter:
        comp_filter = detect_company_from_text(state["rewritten_query"] + " " + state["question"])

    where_filter = None
    if comp_filter:
        where_filter = {"company": {"$eq": comp_filter}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks = [
        {
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]

    # Generate and inject structured financials chunk
    try:
        all_fin = load_financials_for_rag()
        structured_context = generate_financials_context_chunk(all_fin, comp_filter)
        if structured_context:
            special_chunk = {
                "text": structured_context,
                "metadata": {
                    "company": comp_filter or "All",
                    "section": "XBRL Financials",
                    "filed": "Calculated",
                    "report_date": "Calculated",
                    "accession": "XBRL",
                    "chunk_id": 9999,
                    "source_label": "SEC EDGAR XBRL Data",
                },
                "distance": 0.0 # High relevance
            }
            chunks.insert(0, special_chunk)
    except Exception as e:
        print(f"Error injecting structured context: {e}")

    return {**state, "chunks": chunks}


# ── Node 3: Relevance gate ────────────────────────────────────────────────────

def gate(state: RAGState) -> RAGState:
    """
    Two-stage relevance check before answering.
    Stage 1: distance threshold (no LLM cost).
    Stage 2: LLM yes/no on best 3 chunks.
    """
    chunks = state["chunks"]

    if not chunks:
        return {**state, "sufficient": False}

    best_distance = min(c["distance"] for c in chunks)
    if best_distance > RELEVANCE_THRESHOLD:
        return {**state, "sufficient": False}

    # Checked best 3 chunks to prevent false negatives on split/longer documents
    context_sample = "\n".join([c["text"] for c in chunks[:3]])
    messages = [
        SystemMessage(content=(
            "You are a relevance checker. Answer ONLY 'YES' or 'NO'.\n"
            "Does the provided context contain information to answer the question?"
        )),
        HumanMessage(content=f"Context:\n{context_sample}\n\nQuestion: {state['question']}"),
    ]
    verdict = call_llm(messages, temperature=0.0).strip().upper()
    return {**state, "sufficient": verdict.startswith("YES")}


# ── Conditional edge ──────────────────────────────────────────────────────────

def route_after_gate(state: RAGState) -> str:
    return "generate" if state["sufficient"] else "refuse"


# ── Node 4a: Answer generation ────────────────────────────────────────────────

RAG_SYSTEM = SystemMessage(content="""You are a precise financial analyst assistant.
Answer questions using ONLY the provided context from SEC filings.

Rules:
1. If context lacks information, say exactly:
   "I cannot answer this from the available filings."
   Never estimate or use prior knowledge.
2. For every number cited, include its source: (Source: [source_label])
3. For computed metrics, show the formula and inputs.
4. If you see conflicting values for the same period, surface BOTH and note the discrepancy.
5. Be concise and grounded — no narrative beyond what the filings support.""")

def _format_context(chunks: list) -> str:
    return "\n---\n".join([
        f"[{i+1}] {c['metadata'].get('source_label', 'Unknown')}\n{c['text']}"
        for i, c in enumerate(chunks)
    ])

def _detect_conflicts(chunks: list) -> list[str]:
    conflicts = []
    for c in chunks:
        if c["metadata"].get("restated"):
            conflicts.append(
                f"⚠️ Restatement: {c['metadata']['company']} "
                f"period {c['metadata'].get('report_date', '?')} — "
                f"values differ across filings."
            )
    return list(set(conflicts))

def generate(state: RAGState) -> RAGState:
    chunks   = state["chunks"]
    context  = _format_context(chunks)
    messages = [
        RAG_SYSTEM,
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}"),
    ]
    answer    = call_llm(messages, temperature=0.1)
    sources   = list({c["metadata"].get("source_label", "Unknown") for c in chunks})
    conflicts = _detect_conflicts(chunks)

    if conflicts:
        answer += "\n\n" + "\n".join(conflicts)

    return {**state, "answer": answer, "refused": False,
            "sources": sources, "conflicts": conflicts}


# ── Node 4b: Refusal ──────────────────────────────────────────────────────────

def refuse(state: RAGState) -> RAGState:
    return {
        **state,
        "answer": (
            "I cannot answer this from the available filings. "
            "The retrieved context does not contain sufficient information "
            f"to address: '{state['question']}'. "
            "Please check SEC EDGAR directly for this information."
        ),
        "refused":   True,
        "sources":   [],
        "conflicts": [],
    }


# ── Build the LangGraph ───────────────────────────────────────────────────────

def build_rag_graph() -> StateGraph:
    workflow = StateGraph(RAGState)

    workflow.add_node("rewrite",  rewrite_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("gate",     gate)
    workflow.add_node("generate", generate)
    workflow.add_node("refuse",   refuse)

    workflow.set_entry_point("rewrite")
    workflow.add_edge("rewrite",  "retrieve")
    workflow.add_edge("retrieve", "gate")
    workflow.add_conditional_edges(
        "gate",
        route_after_gate,
        {"generate": "generate", "refuse": "refuse"}
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("refuse",   END)

    return workflow.compile()


_rag_graph = None

def get_rag_graph():
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = build_rag_graph()
    return _rag_graph


# ── Public interface ──────────────────────────────────────────────────────────

def run_rag(question: str, company_filter: str | None = None) -> dict:
    graph = get_rag_graph()
    initial_state: RAGState = {
        "question":        question,
        "company_filter":  company_filter,
        "rewritten_query": "",
        "chunks":          [],
        "sufficient":      False,
        "answer":          "",
        "refused":         False,
        "sources":         [],
        "conflicts":       [],
    }
    final_state = graph.invoke(initial_state)
    return {
        "answer":          final_state["answer"],
        "sources":         final_state["sources"],
        "rewritten_query": final_state["rewritten_query"],
        "refused":         final_state["refused"],
        "conflicts":       final_state["conflicts"],
    }