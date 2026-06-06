"""
charts.py
---------
Plotly visualizations for the Streamlit dashboard.
Each chart function:
  - Takes a combined DataFrame (output of metrics.combine_metric)
  - Returns a plotly Figure
  - Includes a data note with the XBRL source
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

COMPANY_COLORS = {
    "NVIDIA": "#76b900",   # NVIDIA green
    "AMD":    "#ed1c24",   # AMD red
    "Intel":  "#0071c5",   # Intel blue
}


def _billions(val: float) -> str:
    """Format a dollar value in billions for display."""
    if pd.isna(val):
        return "N/A"
    return f"${val / 1e9:.1f}B"


def revenue_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: Annual revenue for NVIDIA, AMD, Intel side-by-side.
    Each bar labeled with the $ value.
    """
    fig = go.Figure()

    for company in df["company"].unique():
        comp_df = df[df["company"] == company].sort_values("date")
        fig.add_trace(go.Bar(
            name=company,
            x=comp_df["date"].dt.year,
            y=comp_df["revenue"] / 1e9,
            marker_color=COMPANY_COLORS.get(company, "#888"),
            text=[_billions(v * 1e9) for v in comp_df["revenue"] / 1e9],
            textposition="outside",
        ))

    fig.update_layout(
        title="Annual Revenue (USD Billions)",
        xaxis_title="Fiscal Year",
        yaxis_title="Revenue ($B)",
        barmode="group",
        legend_title="Company",
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def margin_trend_chart(df: pd.DataFrame, margin_col: str,
                       title: str) -> go.Figure:
    """
    Line chart showing a margin trend for all companies over time.
    margin_col: 'gross_margin_pct', 'operating_margin_pct', or 'net_margin_pct'
    """
    fig = go.Figure()

    for company in df["company"].unique():
        comp_df = df[df["company"] == company].dropna(
            subset=[margin_col]
        ).sort_values("date")
        if comp_df.empty:
            continue
        fig.add_trace(go.Scatter(
            name=company,
            x=comp_df["date"].dt.year,
            y=comp_df[margin_col],
            mode="lines+markers",
            line=dict(color=COMPANY_COLORS.get(company, "#888"), width=2),
            marker=dict(size=7),
            hovertemplate=f"{company}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Fiscal Year",
        yaxis_title="Margin (%)",
        legend_title="Company",
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def yoy_growth_chart(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart: Year-over-year revenue growth (%) per company per year.
    Positive = green, negative = red per bar.
    """
    fig = go.Figure()

    for company in df["company"].unique():
        comp_df = df[df["company"] == company].dropna(
            subset=["yoy_growth_pct"]
        ).sort_values("date")
        colors = [
            COMPANY_COLORS.get(company, "#888") if v >= 0 else "#cc3333"
            for v in comp_df["yoy_growth_pct"]
        ]
        fig.add_trace(go.Bar(
            name=company,
            x=comp_df["date"].dt.year,
            y=comp_df["yoy_growth_pct"],
            marker_color=colors,
            text=[f"{v:.1f}%" for v in comp_df["yoy_growth_pct"]],
            textposition="outside",
        ))

    fig.update_layout(
        title="Year-over-Year Revenue Growth (%)",
        xaxis_title="Fiscal Year",
        yaxis_title="YoY Growth (%)",
        barmode="group",
        legend_title="Company",
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def leverage_chart(df: pd.DataFrame) -> go.Figure:
    """
    Line chart: Debt-to-Equity ratio over time.
    """
    fig = go.Figure()

    for company in df["company"].unique():
        comp_df = df[df["company"] == company].dropna(
            subset=["debt_to_equity"]
        ).sort_values("date")
        if comp_df.empty:
            continue
        fig.add_trace(go.Scatter(
            name=company,
            x=comp_df["date"].dt.year,
            y=comp_df["debt_to_equity"].astype(float),
            mode="lines+markers",
            line=dict(color=COMPANY_COLORS.get(company, "#888"), width=2),
            hovertemplate=f"{company}: %{{y:.2f}}x<extra></extra>",
        ))

    fig.update_layout(
        title="Debt-to-Equity Ratio",
        xaxis_title="Fiscal Year",
        yaxis_title="D/E Ratio",
        legend_title="Company",
        height=360,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def fcf_chart(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: Free Cash Flow ($B) per company.
    """
    fig = go.Figure()

    for company in df["company"].unique():
        comp_df = df[df["company"] == company].dropna(
            subset=["free_cash_flow"]
        ).sort_values("date")
        if comp_df.empty:
            continue
        fig.add_trace(go.Bar(
            name=company,
            x=comp_df["date"].dt.year,
            y=comp_df["free_cash_flow"] / 1e9,
            marker_color=COMPANY_COLORS.get(company, "#888"),
            text=[_billions(v * 1e9) for v in comp_df["free_cash_flow"] / 1e9],
            textposition="outside",
        ))

    fig.update_layout(
        title="Free Cash Flow (Operating CF − CapEx, $B)",
        xaxis_title="Fiscal Year",
        yaxis_title="FCF ($B)",
        barmode="group",
        legend_title="Company",
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Dynamic RAG Chart Generator ────────────────────────────────────────────────

from src.metrics import (
    compute_margins, compute_yoy_growth,
    compute_leverage, compute_free_cash_flow,
    combine_metric,
)

RAG_METRIC_CONFIGS = {
    "revenue": {
        "fn": compute_yoy_growth,
        "col": "revenue",
        "title": "Revenue Comparison",
        "ylabel": "Revenue ($B)",
        "scale": 1e9,
        "format": "${val:.2f}B"
    },
    "gross_margin_pct": {
        "fn": compute_margins,
        "col": "gross_margin_pct",
        "title": "Gross Margin Comparison (%)",
        "ylabel": "Gross Margin (%)",
        "scale": 1.0,
        "format": "{val:.1f}%"
    },
    "gross_profit": {
        "fn": compute_margins,
        "col": "gross_profit",
        "title": "Gross Profit Comparison",
        "ylabel": "Gross Profit ($B)",
        "scale": 1e9,
        "format": "${val:.2f}B"
    },
    "operating_margin_pct": {
        "fn": compute_margins,
        "col": "operating_margin_pct",
        "title": "Operating Margin Comparison (%)",
        "ylabel": "Operating Margin (%)",
        "scale": 1.0,
        "format": "{val:.1f}%"
    },
    "operating_income": {
        "fn": compute_margins,
        "col": "operating_income",
        "title": "Operating Income Comparison",
        "ylabel": "Operating Income ($B)",
        "scale": 1e9,
        "format": "${val:.2f}B"
    },
    "net_margin_pct": {
        "fn": compute_margins,
        "col": "net_margin_pct",
        "title": "Net Margin Comparison (%)",
        "ylabel": "Net Margin (%)",
        "scale": 1.0,
        "format": "{val:.1f}%"
    },
    "net_income": {
        "fn": compute_margins,
        "col": "net_income",
        "title": "Net Income Comparison",
        "ylabel": "Net Income ($B)",
        "scale": 1e9,
        "format": "${val:.2f}B"
    },
    "rd_expense": {
        "fn": None,
        "col": "rd_expense",
        "title": "R&D Expense Comparison",
        "ylabel": "R&D Expense ($B)",
        "scale": 1e9,
        "format": "${val:.2f}B"
    },
    "free_cash_flow": {
        "fn": compute_free_cash_flow,
        "col": "free_cash_flow",
        "title": "Free Cash Flow Comparison",
        "ylabel": "Free Cash Flow ($B)",
        "scale": 1e9,
        "format": "${val:.2f}B"
    },
    "operating_cash_flow": {
        "fn": compute_free_cash_flow,
        "col": "operating_cash_flow",
        "title": "Operating Cash Flow Comparison",
        "ylabel": "Operating Cash Flow ($B)",
        "scale": 1e9,
        "format": "${val:.2f}B"
    },
    "capex": {
        "fn": compute_free_cash_flow,
        "col": "capex",
        "title": "Capital Expenditures Comparison",
        "ylabel": "CapEx ($B)",
        "scale": 1e9,
        "format": "${val:.2f}B"
    },
    "debt_to_equity": {
        "fn": compute_leverage,
        "col": "debt_to_equity",
        "title": "Debt-to-Equity Comparison",
        "ylabel": "D/E Ratio",
        "scale": 1.0,
        "format": "{val:.3f}x"
    },
    "long_term_debt": {
        "fn": compute_leverage,
        "col": "long_term_debt",
        "title": "Long-Term Debt Comparison",
        "ylabel": "Debt ($B)",
        "scale": 1e9,
        "format": "${val:.2f}B"
    },
    "stockholders_equity": {
        "fn": compute_leverage,
        "col": "stockholders_equity",
        "title": "Stockholders' Equity Comparison",
        "ylabel": "Equity ($B)",
        "scale": 1e9,
        "format": "${val:.2f}B"
    },
    "total_assets": {
        "fn": None,
        "col": "total_assets",
        "title": "Total Assets Comparison",
        "ylabel": "Assets ($B)",
        "scale": 1e9,
        "format": "${val:.2f}B"
    },
    "total_liabilities": {
        "fn": None,
        "col": "total_liabilities",
        "title": "Total Liabilities Comparison",
        "ylabel": "Liabilities ($B)",
        "scale": 1e9,
        "format": "${val:.2f}B"
    }
}

COMPANY_DISPLAY = {"nvda": "NVIDIA", "amd": "AMD", "intc": "Intel"}


def extract_raw_metric_df(all_financials: dict, metric_name: str) -> pd.DataFrame:
    """Helper to structure raw non-calculated metrics from files."""
    frames = []
    for key, fin in all_financials.items():
        metric_data = fin.get("metrics", {}).get(metric_name)
        if not metric_data:
            continue
        series = metric_data.get("series", [])
        if not series:
            continue
        df = pd.DataFrame(series)
        df["end"] = pd.to_datetime(df["end"])
        df = df.sort_values("end").drop_duplicates(subset=["end"])
        df = df.rename(columns={"end": "date"})
        df["company"] = fin["company"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_query_for_chart(query: str, company_filter: str | None = None) -> tuple[list[str], str | None]:
    """Parse original query and rewritten query to identify company and metric targets."""
    query_lower = query.lower()

    # 1. Detect companies
    companies = []
    if "nvidia" in query_lower or "nvda" in query_lower:
        companies.append("nvda")
    if "amd" in query_lower:
        companies.append("amd")
    if "intel" in query_lower or "intc" in query_lower:
        companies.append("intc")

    # Override/restrict with dropdown filter if applicable
    if company_filter:
        filter_lower = company_filter.lower()
        comp_key = None
        if "nvidia" in filter_lower or "nvda" in filter_lower:
            comp_key = "nvda"
        elif "amd" in filter_lower:
            comp_key = "amd"
        elif "intel" in filter_lower or "intc" in filter_lower:
            comp_key = "intc"
        if comp_key:
            companies = [comp_key]

    # If nothing is detected in query, and filter is empty, show all
    if not companies:
        companies = ["nvda", "amd", "intc"]

    # 2. Detect metric based on keywords
    metric = None

    # Margins and Profits
    if "gross margin" in query_lower:
        metric = "gross_margin_pct"
    elif "gross profit" in query_lower or "gp" in query_lower:
        metric = "gross_profit"
    elif "operating margin" in query_lower:
        metric = "operating_margin_pct"
    elif "operating income" in query_lower or "operating profit" in query_lower or "ebit" in query_lower:
        metric = "operating_income"
    elif "net margin" in query_lower:
        metric = "net_margin_pct"
    elif "net income" in query_lower or "net profit" in query_lower or "net earnings" in query_lower:
        metric = "net_income"
    elif "r&d" in query_lower or "research and development" in query_lower or "rd expense" in query_lower:
        metric = "rd_expense"
    elif "free cash flow" in query_lower or "fcf" in query_lower:
        metric = "free_cash_flow"
    elif "capex" in query_lower or "capital expenditure" in query_lower:
        metric = "capex"
    elif "operating cash flow" in query_lower or "cash from operations" in query_lower:
        metric = "operating_cash_flow"
    elif "debt to equity" in query_lower or "debt-to-equity" in query_lower or "d/e" in query_lower:
        metric = "debt_to_equity"
    elif "long term debt" in query_lower or "long-term debt" in query_lower:
        metric = "long_term_debt"
    elif "liabilities" in query_lower or "total liabilities" in query_lower:
        metric = "total_liabilities"
    elif "equity" in query_lower or "stockholders equity" in query_lower or "shareholders equity" in query_lower:
        metric = "stockholders_equity"
    elif "assets" in query_lower or "total assets" in query_lower:
        metric = "total_assets"
    elif "revenue" in query_lower or "sales" in query_lower or "topline" in query_lower or "top line" in query_lower:
        metric = "revenue"

    return companies, metric


def generate_rag_comparison_chart(all_financials: dict, companies: list[str], metric_key: str) -> go.Figure | None:
    """Generate a dynamic trend comparison line chart based on parsed query options."""
    config = RAG_METRIC_CONFIGS.get(metric_key)
    if not config:
        return None

    # Map keys to standard company names
    comp_names = [COMPANY_DISPLAY.get(c, c) for c in companies]

    # Pull standard or calculated metrics
    if config["fn"]:
        df = combine_metric(all_financials, config["fn"])
    else:
        df = extract_raw_metric_df(all_financials, metric_key)
        if not df.empty:
            df = df.rename(columns={"val": metric_key})

    if df.empty:
        return None

    # Filter to selected companies
    df = df[df["company"].isin(comp_names)]
    if df.empty:
        return None

    df = df.sort_values("date")
    df["year"] = df["date"].dt.year

    fig = go.Figure()
    has_data = False

    for company in df["company"].unique():
        # Clean data for plotting
        comp_df = df[df["company"] == company].dropna(subset=[config["col"]]).drop_duplicates(subset=["year"])
        if comp_df.empty:
            continue
        has_data = True

        y_vals = comp_df[config["col"]]
        if config["scale"] != 1.0:
            y_vals = y_vals / config["scale"]

        fig.add_trace(go.Scatter(
            name=company,
            x=comp_df["year"],
            y=y_vals,
            mode="lines+markers",
            line=dict(color=COMPANY_COLORS.get(company, "#888"), width=3),
            marker=dict(size=8),
            text=[config["format"].format(val=float(v) * config["scale"]) for v in y_vals],
            hovertemplate=f"<b>{company}</b><br>Year: %{{x}}<br>Value: %{{text}}<extra></extra>",
        ))

    if not has_data:
        return None

    fig.update_layout(
        title=dict(
            text=f"📊 {config['title']} (Historical Trend)",
            font=dict(size=15, color="#1e293b", family="Outfit, Inter, sans-serif")
        ),
        xaxis=dict(
            title="Fiscal Year",
            dtick=1,
            gridcolor="rgba(241, 245, 249, 0.5)",
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title=config["ylabel"],
            gridcolor="rgba(241, 245, 249, 0.5)",
            tickfont=dict(size=11)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=280,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

