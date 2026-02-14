"""Portfolio Overview page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.sample_data import get_sample_portfolio


def render() -> None:
    """Render portfolio overview page."""

    st.markdown("## Portfolio Overview")

    data = get_sample_portfolio()
    holdings: pd.DataFrame = data["holdings"]
    port_value: pd.Series = data["portfolio_value"]
    bench_value: pd.Series = data["benchmark_value"]
    port_ret: pd.Series = data["portfolio_returns"]
    trades: pd.DataFrame = data["trades"]

    # ------------------------------------------------------------------
    # KPI metric cards
    # ------------------------------------------------------------------
    total_value = holdings["Market Value"].sum()
    total_pnl = holdings["P&L"].sum()
    total_pnl_pct = (total_pnl / (total_value - total_pnl)) * 100
    ann_ret = ((1 + port_ret).prod()) ** (252 / len(port_ret)) - 1
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio Value", f"${total_value:,.0f}")
    c2.metric("Total P&L", f"${total_pnl:,.0f}", f"{total_pnl_pct:+.1f}%")
    c3.metric("Ann. Return", f"{ann_ret:.1%}")
    c4.metric("Ann. Volatility", f"{ann_vol:.1%}")
    c5.metric("Sharpe Ratio", f"{sharpe:.2f}")

    st.divider()

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------
    col_chart, col_alloc = st.columns([3, 1])

    with col_chart:
        st.markdown("#### Equity Curve")
        eq_df = pd.DataFrame({
            "Portfolio": port_value,
            "Benchmark": bench_value,
        })
        fig = px.line(
            eq_df,
            labels={"value": "Value ($)", "variable": ""},
            color_discrete_sequence=["#00d4aa", "#636efa"],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Sector allocation pie
    # ------------------------------------------------------------------
    with col_alloc:
        st.markdown("#### Sector Allocation")
        sector_weights = holdings.groupby("Sector")["Weight"].sum().reset_index()
        fig_pie = px.pie(
            sector_weights,
            values="Weight",
            names="Sector",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True,
            legend=dict(font=dict(size=10)),
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # Holdings table & Recent trades
    # ------------------------------------------------------------------
    tab_hold, tab_trades = st.tabs(["Holdings", "Recent Trades"])

    with tab_hold:
        display_cols = [
            "Symbol", "Sector", "Shares", "Avg Cost", "Current Price",
            "Market Value", "P&L", "P&L %", "Weight",
        ]
        styled = holdings[display_cols].style.format({
            "Avg Cost": "${:.2f}",
            "Current Price": "${:.2f}",
            "Market Value": "${:,.0f}",
            "P&L": "${:,.0f}",
            "P&L %": "{:+.2f}%",
            "Weight": "{:.2f}%",
        }).map(
            lambda v: "color: #00d4aa" if isinstance(v, (int, float)) and v > 0
            else ("color: #ff4b4b" if isinstance(v, (int, float)) and v < 0 else ""),
            subset=["P&L", "P&L %"],
        )
        st.dataframe(styled, use_container_width=True, height=400)

    with tab_trades:
        st.dataframe(
            trades.style.format({
                "Price": "${:.2f}",
            }).map(
                lambda v: "color: #00d4aa" if v == "BUY" else "color: #ff4b4b",
                subset=["Side"],
            ),
            use_container_width=True,
        )

    # ------------------------------------------------------------------
    # Rolling performance
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("#### Rolling 63-Day Metrics")

    rolling_ret = port_ret.rolling(63).apply(lambda x: (1 + x).prod() - 1, raw=False)
    rolling_vol = port_ret.rolling(63).std() * np.sqrt(252)
    rolling_sharpe = rolling_ret / rolling_vol

    roll_df = pd.DataFrame({
        "Rolling Return": rolling_ret,
        "Rolling Volatility": rolling_vol,
        "Rolling Sharpe": rolling_sharpe,
    }).dropna()

    rc1, rc2 = st.columns(2)
    with rc1:
        fig_rr = px.line(
            roll_df[["Rolling Return", "Rolling Volatility"]],
            color_discrete_sequence=["#00d4aa", "#ff4b4b"],
        )
        fig_rr.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=250,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.05, x=0),
        )
        st.plotly_chart(fig_rr, use_container_width=True)

    with rc2:
        fig_rs = px.line(
            roll_df[["Rolling Sharpe"]],
            color_discrete_sequence=["#636efa"],
        )
        fig_rs.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_rs.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=250,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.05, x=0),
        )
        st.plotly_chart(fig_rs, use_container_width=True)
