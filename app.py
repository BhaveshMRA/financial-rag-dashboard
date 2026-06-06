"""
app.py
------
Main Streamlit application.
Two tabs:
  1. Financial Dashboard — comparative charts (NVIDIA vs AMD vs Intel)
  2. RAG Chatbot         — cited, grounded Q&A over 10-K filings

Run after data_pipeline.py has completed.
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path

from src.metrics import (
    compute_margins, compute_yoy_growth,
    compute_leverage, compute_free_cash_flow,
    combine_metric, restatement_conflicts,
)
from src.charts import (
    revenue_comparison_chart, margin_trend_chart,
    yoy_growth_chart, leverage_chart, fcf_chart,
    parse_query_for_chart, generate_rag_comparison_chart,
)
from src.rag_chain import run_rag

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial RAG Dashboard — NVDA / AMD / INTC",
    page_icon="📊",
    layout="wide",
)

FINANCIALS_DIR = Path("data/financials")
COMPANY_DISPLAY = {"nvda": "NVIDIA", "amd": "AMD", "intc": "Intel"}


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading financial data...")
def load_all_financials() -> dict:
    """Load pre-fetched financials from JSON files."""
    all_fin = {}
    for key in ["nvda", "amd", "intc"]:
        path = FINANCIALS_DIR / f"{key}.json"
        if path.exists():
            with open(path) as f:
                all_fin[key] = json.load(f)
        else:
            st.warning(
                f"No data found for {COMPANY_DISPLAY[key]}. "
                "Run `python data_pipeline.py` first.",
                icon="⚠️"
            )
    return all_fin


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("📊 Financial RAG Dashboard")
st.sidebar.markdown("**Companies:** NVIDIA · AMD · Intel")
st.sidebar.markdown("**Source:** SEC EDGAR 10-K filings")
st.sidebar.markdown("**LLM:** gemma4:31b via Ollama Cloud")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data provenance**\n"
    "Financial figures are extracted from SEC EDGAR XBRL "
    "(us-gaap concepts). Every number is traceable to its "
    "source accession number."
)

all_fin = load_all_financials()
if not all_fin:
    st.error("No financial data loaded. Run `python data_pipeline.py` first.")
    st.stop()

tab_dashboard, tab_chat, tab_conflicts, tab_eval = st.tabs([
    "📈 Dashboard", "💬 RAG Chat", "⚠️ Data Conflicts", "🧪 Evaluation"
])


# ── Tab 1: Dashboard ──────────────────────────────────────────────────────────

with tab_dashboard:

    st.header("Comparative Financial Dashboard")
    st.caption(
        "All figures sourced from SEC EDGAR XBRL (us-gaap). "
        "Hover over charts for exact values. "
        "See Data Conflicts tab for restatement flags."
    )

    # ── Revenue ──────────────────────────────────────────────────────────────
    st.subheader("Revenue")
    rev_df = combine_metric(all_fin, compute_yoy_growth)
    margins_df = combine_metric(all_fin, compute_margins)

    if not margins_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                revenue_comparison_chart(margins_df),
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                yoy_growth_chart(rev_df),
                use_container_width=True
            )

        # Source attribution
        sources = []
        for key, fin in all_fin.items():
            src = fin.get("metrics", {}).get("revenue", {}).get("source", "")
            if src:
                sources.append(f"**{fin['company']}**: {src}")
        if sources:
            with st.expander("📎 Data sources"):
                for s in sources:
                    st.markdown(s)

    # ── Margins ───────────────────────────────────────────────────────────────
    st.subheader("Profit Margins")
    if not margins_df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(
                margin_trend_chart(margins_df, "gross_margin_pct", "Gross Margin (%)"),
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                margin_trend_chart(margins_df, "operating_margin_pct", "Operating Margin (%)"),
                use_container_width=True
            )
        with col3:
            st.plotly_chart(
                margin_trend_chart(margins_df, "net_margin_pct", "Net Margin (%)"),
                use_container_width=True
            )
        with st.expander("📐 Formula"):
            st.code(
                "Gross Margin    = Gross Profit / Revenue × 100\n"
                "Operating Margin = Operating Income / Revenue × 100\n"
                "Net Margin       = Net Income / Revenue × 100\n\n"
                "Inputs: us-gaap/GrossProfit, OperatingIncomeLoss, "
                "NetIncomeLoss, Revenues (or equivalent) — SEC EDGAR XBRL"
            )

    # ── Leverage & FCF ────────────────────────────────────────────────────────
    st.subheader("Capital Structure & Cash Flow")
    lev_df = combine_metric(all_fin, compute_leverage)
    fcf_df = combine_metric(all_fin, compute_free_cash_flow)

    col1, col2 = st.columns(2)
    with col1:
        if not lev_df.empty:
            st.plotly_chart(leverage_chart(lev_df), use_container_width=True)
            with st.expander("📐 Formula"):
                st.code(
                    "Debt-to-Equity = Long-Term Debt / Stockholders' Equity\n"
                    "Inputs: LongTermDebt, StockholdersEquity — SEC EDGAR XBRL"
                )
    with col2:
        if not fcf_df.empty:
            st.plotly_chart(fcf_chart(fcf_df), use_container_width=True)
            with st.expander("📐 Formula"):
                st.code(
                    "FCF = Operating Cash Flow − Capital Expenditures\n"
                    "Inputs: NetCashProvidedByUsedInOperatingActivities,\n"
                    "        PaymentsToAcquirePropertyPlantAndEquipment — SEC EDGAR XBRL"
                )

    # ── Snapshot table ────────────────────────────────────────────────────────
    st.subheader("Latest Year Snapshot")
    if not margins_df.empty:
        latest_rows = []
        for key, fin in all_fin.items():
            company = fin["company"]
            sub = margins_df[margins_df["company"] == company].sort_values("date")
            if sub.empty:
                continue
            row = sub.iloc[-1]
            latest_rows.append({
                "Company": company,
                "FY End": row["date"].strftime("%Y-%m-%d"),
                "Revenue ($B)": f"{row['revenue']/1e9:.1f}",
                "Gross Margin": f"{row.get('gross_margin_pct', 'N/A'):.1f}%" if pd.notna(row.get("gross_margin_pct")) else "N/A",
                "Op. Margin": f"{row.get('operating_margin_pct', 'N/A'):.1f}%" if pd.notna(row.get("operating_margin_pct")) else "N/A",
                "Net Margin": f"{row.get('net_margin_pct', 'N/A'):.1f}%" if pd.notna(row.get("net_margin_pct")) else "N/A",
            })
        if latest_rows:
            st.dataframe(pd.DataFrame(latest_rows), use_container_width=True, hide_index=True)


# ── Tab 2: RAG Chat ───────────────────────────────────────────────────────────

with tab_chat:
    st.header("RAG Chat — Ask About the Filings")
    st.caption(
        "Answers are grounded in SEC 10-K filings (MD&A + Risk Factors). "
        "Every answer includes source citations. "
        "If the answer is not in the filings, the system will refuse rather than guess."
    )

    # Company filter
    company_filter = st.selectbox(
        "Filter by company (optional)",
        options=["All companies", "NVIDIA", "AMD", "Intel"],
    )
    filter_val = None if company_filter == "All companies" else company_filter

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display prior messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- {s}")
            # RAG chart rendering right below sources
            if msg["role"] == "assistant" and msg.get("original_query"):
                companies, metric = parse_query_for_chart(
                    msg["original_query"] + " " + msg.get("rewritten_query", ""),
                    company_filter=msg.get("company_filter")
                )
                if metric:
                    fig = generate_rag_comparison_chart(all_fin, companies, metric)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            if msg.get("rewritten_query") and msg["rewritten_query"] != msg.get("original_query"):
                with st.expander("🔄 Query rewritten to"):
                    st.code(msg["rewritten_query"])

    # Input
    if question := st.chat_input("Ask a question about NVIDIA, AMD, or Intel..."):
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "original_query": question,
        })
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving from filings..."):
                result = run_rag(question, company_filter=filter_val)

            answer = result["answer"]
            refused = result["refused"]

            if refused:
                st.warning(answer, icon="🚫")
            else:
                st.markdown(answer)

            if result.get("sources"):
                with st.expander("📎 Sources"):
                    for s in result["sources"]:
                        st.markdown(f"- {s}")

            # RAG chart rendering right below sources
            if not refused:
                companies, metric = parse_query_for_chart(
                    question + " " + result["rewritten_query"],
                    company_filter=filter_val
                )
                if metric:
                    fig = generate_rag_comparison_chart(all_fin, companies, metric)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            if result["rewritten_query"] != question:
                with st.expander("🔄 Query rewritten to"):
                    st.code(result["rewritten_query"])

            if result.get("conflicts"):
                st.warning(
                    "Data conflicts detected:\n" + "\n".join(result["conflicts"]),
                    icon="⚠️"
                )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": result.get("sources", []),
            "rewritten_query": result.get("rewritten_query", ""),
            "original_query": question,
            "company_filter": filter_val,
        })


# ── Tab 3: Data Conflicts ─────────────────────────────────────────────────────

with tab_conflicts:
    st.header("⚠️ Restatements & Data Conflicts")
    st.markdown(
        "When the same metric appears with different values across filings "
        "(e.g., a restated figure), the system flags it here rather than "
        "silently choosing one value. This is the 'messy reality of financial data' handler."
    )

    conflicts = restatement_conflicts(all_fin)

    if conflicts:
        st.error(f"Found {len(conflicts)} restatement(s) in the data.")
        for c in conflicts:
            with st.expander(f"{c['company']} — {c['metric']} — {c['period_end']}"):
                st.markdown(f"**Note:** {c['note']}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Original Value", f"${c['original_value']:,}")
                col2.metric("Restated Value", f"${c['current_value']:,}")
                col3.metric("Change", f"${c['change']:+,}")
    else:
        st.success(
            "No restatements detected in the loaded data. "
            "All values are consistent across filings.",
            icon="✅"
        )


# ── Tab 4: Evaluation ─────────────────────────────────────────────────────────

with tab_eval:
    st.header("🧪 Evaluation Framework")
    eval_path = Path("eval/eval_results.json")

    if eval_path.exists():
        with open(eval_path) as f:
            results = json.load(f)

        summary = results.get("summary", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Questions", summary.get("total", "—"))
        col2.metric("Correct Answers", f"{summary.get('correct_rate', 0):.0%}")
        col3.metric("Citation Accuracy", f"{summary.get('citation_accuracy', 0):.0%}")
        col4.metric("Hallucination Rate", f"{summary.get('hallucination_rate', 0):.0%}")

        st.markdown("---")
        st.subheader("Per-Question Results")
        for item in results.get("results", []):
            icon = "✅" if item.get("correct") else ("🚫" if item.get("refused") else "❌")
            with st.expander(f"{icon} {item['question']}"):
                st.markdown(f"**Expected:** {item.get('expected', 'N/A')}")
                st.markdown(f"**Got:** {item.get('answer', 'N/A')}")
                st.markdown(f"**Correct:** {item.get('correct')}")
                st.markdown(f"**Refused:** {item.get('refused', False)}")
                if item.get("sources"):
                    st.markdown("**Sources cited:** " + ", ".join(item["sources"]))
    else:
        st.info(
            "Evaluation results not yet available. "
            "Run `python eval/run_eval.py` to generate them.",
            icon="ℹ️"
        )
        st.markdown("**Evaluation question set preview:**")
        eval_q_path = Path("eval/labeled_questions.json")
        if eval_q_path.exists():
            with open(eval_q_path) as f:
                questions = json.load(f)
            for q in questions:
                st.markdown(
                    f"- **[{q['category']}]** {q['question']} "
                    f"{'*(unanswerable)*' if q.get('unanswerable') else ''}"
                )
