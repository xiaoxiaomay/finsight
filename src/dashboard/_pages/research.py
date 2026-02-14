"""AI Research Assistant page."""

from __future__ import annotations

import json
import time

import streamlit as st


def _mock_report(symbol: str) -> dict | None:
    """Try to load a pre-generated report from disk."""
    from pathlib import Path

    report_path = Path("reports") / f"{symbol.upper()}_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    return None


def _render_score_bar(label: str, score: float, max_score: float = 100) -> None:
    """Render a labeled score bar."""
    pct = min(score / max_score, 1.0)
    color = "#00d4aa" if score >= 65 else "#ffa726" if score >= 45 else "#ff4b4b"
    st.markdown(
        f"""<div style="margin-bottom:8px;">
        <span style="color:#aaa;font-size:12px;">{label}</span>
        <div style="background:#1e1e2f;border-radius:4px;height:20px;margin-top:2px;">
            <div style="background:{color};width:{pct*100:.0f}%;height:100%;
                        border-radius:4px;text-align:right;padding-right:6px;
                        font-size:11px;line-height:20px;color:#fff;">
                {score:.1f}
            </div>
        </div></div>""",
        unsafe_allow_html=True,
    )


def _render_report_card(report: dict) -> None:
    """Render a research report as a dashboard card."""
    symbol = report.get("symbol", "?")
    rec = report.get("recommendation", "N/A")
    composite = report.get("composite_score", 0)
    company = report.get("company_name", symbol)

    rec_colors = {
        "Strong Buy": "#00d4aa",
        "Buy": "#00d4aa",
        "Hold": "#ffa726",
        "Sell": "#ff4b4b",
        "Strong Sell": "#ff4b4b",
    }
    rec_color = rec_colors.get(rec, "#888")

    st.markdown(
        f"""<div style="background:#1e1e2f;border:1px solid #2d2d44;border-radius:8px;
                padding:16px;margin-bottom:16px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <h3 style="margin:0;color:#e0e0e0;">{company} ({symbol})</h3>
                <span style="background:{rec_color};color:#000;padding:4px 12px;
                       border-radius:4px;font-weight:bold;font-size:14px;">{rec}</span>
            </div>
            <p style="color:#888;font-size:13px;margin-top:4px;">
                Composite Score: <strong style="color:#e0e0e0;">{composite:.1f}/100</strong>
                &nbsp;|&nbsp; Conviction: {report.get('conviction', 'N/A')}
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Score bars
    scores = [
        ("Fundamental", report.get("fundamental_score", 0)),
        ("Quant Signal", report.get("quant_score", 0)),
        ("Sentiment", report.get("sentiment_score", 0)),
        ("Macro", report.get("macro_score", 0)),
        ("Peer", report.get("peer_score", 0)),
    ]
    sc1, sc2 = st.columns(2)
    for i, (label, score) in enumerate(scores):
        with sc1 if i % 2 == 0 else sc2:
            _render_score_bar(label, score)

    st.divider()

    # Bull / Bear case
    col_bull, col_bear = st.columns(2)
    with col_bull:
        st.markdown("##### Bull Case")
        for point in report.get("bull_case", []):
            st.markdown(f"- {point}")
    with col_bear:
        st.markdown("##### Bear Case")
        for point in report.get("bear_case", []):
            st.markdown(f"- {point}")

    # Key risks & catalysts
    st.divider()
    col_risk, col_cat = st.columns(2)
    with col_risk:
        st.markdown("##### Key Risks")
        for r in report.get("key_risks", []):
            st.markdown(f"- {r}")
    with col_cat:
        st.markdown("##### Catalysts")
        for c in report.get("catalysts", []):
            st.markdown(f"- {c}")

    # Earnings details
    ea = report.get("earnings_analysis")
    if ea:
        st.divider()
        st.markdown("##### Financial Snapshot")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Gross Margin", f"{ea.get('gross_margin', 0):.1%}")
        m2.metric("Operating Margin", f"{ea.get('operating_margin', 0):.1%}")
        m3.metric("Net Margin", f"{ea.get('net_margin', 0):.1%}")
        m4.metric("FCF Yield", f"{ea.get('fcf_yield', 0):.1%}")

    # Quant signals
    qs = report.get("quant_signals")
    if qs and qs.get("factor_scores"):
        st.divider()
        st.markdown("##### Factor Exposures")
        import pandas as pd

        factors_df = pd.DataFrame(qs["factor_scores"])
        if not factors_df.empty:
            import plotly.express as pfx

            fig = pfx.bar(
                factors_df,
                x="factor_name",
                y="z_score",
                color="z_score",
                color_continuous_scale=["#ff4b4b", "#888", "#00d4aa"],
                color_continuous_midpoint=0,
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="",
                yaxis_title="Z-Score",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Data sources
    sources = report.get("data_sources", [])
    if sources:
        st.divider()
        st.markdown(
            "##### Data Sources: " + " | ".join(f"`{s}`" for s in sources)
        )


def render() -> None:
    """Render AI Research Assistant page."""

    st.markdown("## AI Research Assistant")
    st.caption("Enter a stock symbol to generate a multi-agent research report.")

    # Chat history
    if "research_messages" not in st.session_state:
        st.session_state.research_messages = []

    # Display past messages
    for msg in st.session_state.research_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and isinstance(msg["content"], dict):
                _render_report_card(msg["content"])
            else:
                st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("e.g. Analyze AAPL, or: What is MSFT's quant signal?")

    if user_input:
        st.session_state.research_messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Extract symbol (simple heuristic)
        symbol = None
        for word in user_input.upper().split():
            cleaned = word.strip(".,!?()\"'")
            if 1 <= len(cleaned) <= 5 and cleaned.isalpha() and cleaned not in {
                "A", "I", "THE", "IS", "OF", "FOR", "AND", "OR", "TO", "IN",
                "ON", "AT", "BY", "AN", "IT", "AS", "IF", "SO", "DO", "NO",
                "UP", "MY", "ME", "WE", "HE", "BE", "WHAT", "FULL",
            }:
                symbol = cleaned
                break

        with st.chat_message("assistant"):
            if symbol:
                with st.spinner(f"Running multi-agent analysis for {symbol}..."):
                    # Try loading pre-generated report
                    report = _mock_report(symbol)

                    if report:
                        _render_report_card(report)
                        st.session_state.research_messages.append(
                            {"role": "assistant", "content": report}
                        )
                    else:
                        # Show placeholder for symbols without pre-generated reports
                        time.sleep(0.5)
                        placeholder = {
                            "symbol": symbol,
                            "company_name": symbol,
                            "recommendation": "Hold",
                            "conviction": "low",
                            "composite_score": 50.0,
                            "fundamental_score": 50.0,
                            "quant_score": 50.0,
                            "sentiment_score": 50.0,
                            "macro_score": 50.0,
                            "peer_score": 50.0,
                            "bull_case": [
                                "Demo mode — connect live agents for real analysis."
                            ],
                            "bear_case": [
                                "No live data available in demo mode."
                            ],
                            "key_risks": ["Demo placeholder"],
                            "catalysts": ["Demo placeholder"],
                            "data_sources": ["Demo Mode"],
                        }
                        st.info(
                            f"No pre-generated report for **{symbol}**. "
                            "Showing placeholder. Run `python scripts/generate_report.py "
                            f"{symbol}` to generate a full report."
                        )
                        _render_report_card(placeholder)
                        st.session_state.research_messages.append(
                            {"role": "assistant", "content": placeholder}
                        )
            else:
                msg = (
                    "I couldn't identify a stock symbol in your query. "
                    "Try something like: **Analyze AAPL** or **MSFT quant signal**"
                )
                st.markdown(msg)
                st.session_state.research_messages.append(
                    {"role": "assistant", "content": msg}
                )
