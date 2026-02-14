"""Data Explorer page — browse market data, factors, macro, and quality logs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.sample_data import get_sample_prices

# --------------------------------------------------------------------------- #
# Cached helpers
# --------------------------------------------------------------------------- #

@st.cache_data(ttl=3600)
def _get_factor_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute sample factor z-scores for the explorer."""
    rng = np.random.default_rng(99)
    symbols = prices.columns.tolist()
    factors = [
        "momentum_12_1", "earnings_yield", "roe", "gross_profitability",
        "book_to_market", "ev_ebitda", "accruals", "asset_growth",
        "volatility_60d", "short_term_reversal",
    ]
    rows = []
    for sym in symbols:
        for f in factors:
            rows.append({
                "symbol": sym,
                "factor": f,
                "z_score": rng.normal(0, 1),
                "percentile": rng.uniform(0, 100),
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def _get_macro_indicators() -> pd.DataFrame:
    """Generate sample macro indicator time series."""
    rng = np.random.default_rng(88)
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    indicators = {
        "GDP Growth (%)": 2.5 + np.cumsum(rng.normal(0, 0.15, 60)),
        "CPI YoY (%)": 2.0 + np.cumsum(rng.normal(0, 0.1, 60)),
        "Unemployment (%)": 4.0 + np.cumsum(rng.normal(0, 0.05, 60)),
        "Fed Funds Rate (%)": np.clip(
            2.0 + np.cumsum(rng.normal(0.02, 0.1, 60)), 0, 6
        ),
        "10Y Treasury (%)": np.clip(
            3.5 + np.cumsum(rng.normal(0, 0.08, 60)), 1, 6
        ),
        "VIX": np.clip(18 + np.cumsum(rng.normal(0, 0.5, 60)), 10, 45),
    }
    df = pd.DataFrame(indicators, index=dates)
    df.index.name = "date"
    return df


@st.cache_data(ttl=3600)
def _get_quality_log() -> pd.DataFrame:
    """Generate sample data quality log entries."""
    rng = np.random.default_rng(55)
    checks = [
        ("market_data_coverage", "pass", "500 symbols, 1.2M rows"),
        ("market_data_freshness", "pass", "Latest: 2026-02-12"),
        ("market_data_gaps", "warning", "3 symbols with gaps > 5 days"),
        ("fundamentals_pit_integrity", "pass", "All filing_date >= report_date"),
        ("fundamentals_coverage", "pass", "480 symbols, 7 years"),
        ("macro_coverage", "pass", "15 indicators, 3600 observations"),
        ("macro_freshness", "warning", "2 indicators > 30 days stale"),
    ]
    rows = []
    for name, status, detail in checks:
        rows.append({
            "check_name": name,
            "status": status,
            "details": detail,
            "checked_at": pd.Timestamp("2026-02-12 08:00:00"),
            "duration_ms": rng.integers(100, 5000),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Page render
# --------------------------------------------------------------------------- #

def render() -> None:
    """Render Data Explorer page."""

    st.markdown("## Data Explorer")

    tab_market, tab_factor, tab_macro, tab_quality = st.tabs([
        "Market Data", "Factor Signals", "Macro Indicators", "Data Quality",
    ])

    prices = get_sample_prices()
    symbols = prices.columns.tolist()

    # ================================================================== #
    # Tab 1: Market Data
    # ================================================================== #
    with tab_market:
        st.markdown("#### Market Data Browser")

        mc1, mc2 = st.columns([1, 2])
        with mc1:
            selected_syms = st.multiselect(
                "Select Symbols",
                symbols,
                default=symbols[:5],
                max_selections=10,
            )
            date_range = st.date_input(
                "Date Range",
                value=(prices.index[0].date(), prices.index[-1].date()),
                min_value=prices.index[0].date(),
                max_value=prices.index[-1].date(),
            )

        if selected_syms and len(date_range) == 2:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            filtered = prices.loc[start:end, selected_syms]

            with mc2:
                fig = px.line(
                    filtered,
                    labels={"value": "Price ($)", "variable": "Symbol"},
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation="h", y=1.05, x=0),
                )
                st.plotly_chart(fig, use_container_width=True)

            # OHLCV-style summary (using close prices)
            st.markdown("##### Price Statistics")
            stats = filtered.describe().T[["mean", "std", "min", "max"]]
            stats.columns = ["Mean", "Std Dev", "Min", "Max"]
            stats["Return"] = (filtered.iloc[-1] / filtered.iloc[0] - 1)
            st.dataframe(
                stats.style.format({
                    "Mean": "${:.2f}",
                    "Std Dev": "${:.2f}",
                    "Min": "${:.2f}",
                    "Max": "${:.2f}",
                    "Return": "{:.1%}",
                }),
                use_container_width=True,
            )

    # ================================================================== #
    # Tab 2: Factor Signals
    # ================================================================== #
    with tab_factor:
        st.markdown("#### Factor Signal Explorer")

        factor_data = _get_factor_signals(prices)
        factor_names = factor_data["factor"].unique().tolist()

        fc1, fc2 = st.columns(2)
        with fc1:
            sel_factor = st.selectbox("Select Factor", factor_names)
        with fc2:
            view_mode = st.radio(
                "View",
                ["Distribution", "Cross-Sectional Ranking"],
                horizontal=True,
            )

        factor_slice = factor_data[factor_data["factor"] == sel_factor]

        if view_mode == "Distribution":
            fig_dist = px.histogram(
                factor_slice,
                x="z_score",
                nbins=40,
                color_discrete_sequence=["#636efa"],
                labels={"z_score": "Z-Score"},
            )
            fig_dist.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_dist.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Z-Score",
                yaxis_title="Count",
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            ranked = factor_slice.sort_values("z_score", ascending=False).head(20)
            fig_rank = px.bar(
                ranked,
                x="z_score",
                y="symbol",
                orientation="h",
                color="z_score",
                color_continuous_scale=["#ff4b4b", "#161b22", "#00d4aa"],
                color_continuous_midpoint=0,
            )
            fig_rank.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Z-Score",
                yaxis_title="",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_rank, use_container_width=True)

        # Summary stats
        st.markdown("##### Factor Summary")
        fsummary = factor_data.groupby("factor").agg(
            mean_z=("z_score", "mean"),
            std_z=("z_score", "std"),
            min_z=("z_score", "min"),
            max_z=("z_score", "max"),
            count=("z_score", "count"),
        ).round(3)
        st.dataframe(fsummary, use_container_width=True)

    # ================================================================== #
    # Tab 3: Macro Indicators
    # ================================================================== #
    with tab_macro:
        st.markdown("#### Macro Indicator Trends")

        macro = _get_macro_indicators()
        indicator_names = macro.columns.tolist()

        sel_indicators = st.multiselect(
            "Select Indicators",
            indicator_names,
            default=indicator_names[:3],
        )

        if sel_indicators:
            fig_macro = px.line(
                macro[sel_indicators],
                labels={"value": "", "variable": "Indicator"},
            )
            fig_macro.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=1.05, x=0),
            )
            st.plotly_chart(fig_macro, use_container_width=True)

            # Latest values
            st.markdown("##### Latest Values")
            latest = macro.iloc[-1][sel_indicators].to_frame("Value").T
            st.dataframe(
                latest.style.format("{:.2f}"),
                use_container_width=True,
            )

    # ================================================================== #
    # Tab 4: Data Quality
    # ================================================================== #
    with tab_quality:
        st.markdown("#### Data Quality Dashboard")

        qlog = _get_quality_log()

        # Summary metrics
        n_pass = (qlog["status"] == "pass").sum()
        n_warn = (qlog["status"] == "warning").sum()
        n_fail = (qlog["status"] == "fail").sum()
        total = len(qlog)

        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("Total Checks", total)
        qc2.metric("Passed", n_pass)
        qc3.metric("Warnings", n_warn)
        qc4.metric("Failed", n_fail)

        st.divider()

        # Status-colored table
        def _status_color(val: str) -> str:
            if val == "pass":
                return "color: #00d4aa"
            elif val == "warning":
                return "color: #ffa726"
            return "color: #ff4b4b"

        st.dataframe(
            qlog.style.map(_status_color, subset=["status"]),
            use_container_width=True,
            hide_index=True,
        )

        # Quality score gauge
        score = n_pass / total * 100 if total > 0 else 0
        st.markdown(f"##### Overall Quality Score: **{score:.0f}%**")
        st.progress(score / 100)
