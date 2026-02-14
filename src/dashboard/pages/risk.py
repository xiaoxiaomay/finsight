"""Risk Dashboard page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.sample_data import get_sample_portfolio, get_sample_prices


def _compute_risk_metrics(
    port_ret: pd.Series,
    bench_ret: pd.Series,
) -> dict:
    """Compute risk metrics, using real engine when available."""
    try:
        from src.quant.portfolio.risk import (
            conditional_var,
            portfolio_beta,
            value_at_risk,
        )

        var_95 = value_at_risk(port_ret, confidence=0.95)
        var_99 = value_at_risk(port_ret, confidence=0.99)
        cvar_95 = conditional_var(port_ret, confidence=0.95)
        beta = portfolio_beta(port_ret, bench_ret)
    except Exception:
        var_95 = float(np.percentile(port_ret, 5))
        var_99 = float(np.percentile(port_ret, 1))
        cvar_95 = float(port_ret[port_ret <= np.percentile(port_ret, 5)].mean())
        cov = np.cov(port_ret, bench_ret)
        beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 1.0

    ann_vol = float(port_ret.std() * np.sqrt(252))
    bench_vol = float(bench_ret.std() * np.sqrt(252))
    tracking_error = float((port_ret - bench_ret).std() * np.sqrt(252))

    cum = (1 + port_ret).cumprod()
    drawdown_series = cum / cum.cummax() - 1
    max_dd = float(drawdown_series.min())

    # Drawdown duration
    in_dd = drawdown_series < 0
    if in_dd.any():
        groups = (~in_dd).cumsum()
        dd_lens = in_dd.groupby(groups).sum()
        max_dd_duration = int(dd_lens.max())
    else:
        max_dd_duration = 0

    # Downside deviation
    downside = port_ret[port_ret < 0]
    downside_dev = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else 0.0

    return {
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "beta": beta,
        "ann_volatility": ann_vol,
        "bench_volatility": bench_vol,
        "tracking_error": tracking_error,
        "max_drawdown": max_dd,
        "max_dd_duration_days": max_dd_duration,
        "downside_deviation": downside_dev,
        "drawdown_series": drawdown_series,
    }


def render() -> None:
    """Render Risk Dashboard page."""

    st.markdown("## Risk Dashboard")

    data = get_sample_portfolio()
    port_ret = data["portfolio_returns"]
    bench_ret = data["benchmark_returns"]
    holdings = data["holdings"]

    risk = _compute_risk_metrics(port_ret, bench_ret)

    # ------------------------------------------------------------------
    # KPI metric cards
    # ------------------------------------------------------------------
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("VaR (95%)", f"{risk['var_95']:.2%}")
    c2.metric("CVaR (95%)", f"{risk['cvar_95']:.2%}")
    c3.metric("VaR (99%)", f"{risk['var_99']:.2%}")
    c4.metric("Beta", f"{risk['beta']:.2f}")
    c5.metric("Ann. Volatility", f"{risk['ann_volatility']:.1%}")
    c6.metric("Max Drawdown", f"{risk['max_drawdown']:.1%}")

    st.divider()

    # ------------------------------------------------------------------
    # Two-column layout: Drawdown + Return Distribution
    # ------------------------------------------------------------------
    col_dd, col_dist = st.columns(2)

    with col_dd:
        st.markdown("#### Drawdown Analysis")
        dd = risk["drawdown_series"]
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index,
            y=dd.values,
            fill="tozeroy",
            fillcolor="rgba(255,75,75,0.3)",
            line=dict(color="#ff4b4b", width=1),
            name="Drawdown",
        ))
        fig_dd.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_tickformat=".1%",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        st.caption(
            f"Max drawdown duration: **{risk['max_dd_duration_days']} days** "
            f"| Tracking error: **{risk['tracking_error']:.1%}** "
            f"| Downside deviation: **{risk['downside_deviation']:.1%}**"
        )

    with col_dist:
        st.markdown("#### Return Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=port_ret.values * 100,
            nbinsx=60,
            marker_color="#636efa",
            opacity=0.7,
            name="Portfolio",
        ))
        fig_hist.add_trace(go.Histogram(
            x=bench_ret.values * 100,
            nbinsx=60,
            marker_color="#00d4aa",
            opacity=0.5,
            name="Benchmark",
        ))
        # VaR lines
        fig_hist.add_vline(
            x=risk["var_95"] * 100,
            line_dash="dash",
            line_color="#ffa726",
            annotation_text="VaR 95%",
        )
        fig_hist.add_vline(
            x=risk["var_99"] * 100,
            line_dash="dash",
            line_color="#ff4b4b",
            annotation_text="VaR 99%",
        )
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            barmode="overlay",
            legend=dict(orientation="h", y=1.05, x=0),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # Sector concentration & Correlation
    # ------------------------------------------------------------------
    col_sec, col_corr = st.columns(2)

    with col_sec:
        st.markdown("#### Sector Concentration")
        sector_data = holdings.groupby("Sector")["Weight"].sum().sort_values(
            ascending=True
        ).reset_index()
        fig_bar = px.bar(
            sector_data,
            x="Weight",
            y="Sector",
            orientation="h",
            color="Weight",
            color_continuous_scale=["#1e1e2f", "#00d4aa"],
        )
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Weight (%)",
            yaxis_title="",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_corr:
        st.markdown("#### Cross-Asset Correlation")
        # Use sample prices for correlation
        prices = get_sample_prices(n_symbols=10, n_days=252)
        returns = prices.pct_change().dropna()
        corr = returns.corr()

        fig_corr = px.imshow(
            corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            color_continuous_scale=["#ff4b4b", "#1e1e2f", "#00d4aa"],
            color_continuous_midpoint=0,
            aspect="auto",
            zmin=-1,
            zmax=1,
        )
        fig_corr.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # Rolling volatility & beta
    # ------------------------------------------------------------------
    st.markdown("#### Rolling Risk Metrics (63-Day)")

    roll_vol = port_ret.rolling(63).std() * np.sqrt(252)
    roll_bench_vol = bench_ret.rolling(63).std() * np.sqrt(252)

    # Rolling beta
    cov_roll = port_ret.rolling(63).cov(bench_ret)
    var_roll = bench_ret.rolling(63).var()
    roll_beta = cov_roll / var_roll

    rc1, rc2 = st.columns(2)

    with rc1:
        vol_df = pd.DataFrame({
            "Portfolio Vol": roll_vol,
            "Benchmark Vol": roll_bench_vol,
        }).dropna()
        fig_vol = px.line(
            vol_df,
            color_discrete_sequence=["#ff4b4b", "#636efa"],
        )
        fig_vol.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Annualized Vol",
            yaxis_tickformat=".0%",
            legend=dict(orientation="h", y=1.05, x=0),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with rc2:
        beta_df = pd.DataFrame({"Rolling Beta": roll_beta}).dropna()
        fig_beta = px.line(
            beta_df,
            color_discrete_sequence=["#ffa726"],
        )
        fig_beta.add_hline(y=1.0, line_dash="dash", line_color="gray")
        fig_beta.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Beta",
            legend=dict(orientation="h", y=1.05, x=0),
        )
        st.plotly_chart(fig_beta, use_container_width=True)

    # ------------------------------------------------------------------
    # Risk summary table
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("#### Risk Summary")

    risk_table = pd.DataFrame({
        "Metric": [
            "VaR (95%)", "VaR (99%)", "CVaR (95%)", "Beta",
            "Portfolio Vol", "Benchmark Vol", "Tracking Error",
            "Max Drawdown", "Max DD Duration", "Downside Dev",
        ],
        "Value": [
            f"{risk['var_95']:.2%}",
            f"{risk['var_99']:.2%}",
            f"{risk['cvar_95']:.2%}",
            f"{risk['beta']:.3f}",
            f"{risk['ann_volatility']:.1%}",
            f"{risk['bench_volatility']:.1%}",
            f"{risk['tracking_error']:.1%}",
            f"{risk['max_drawdown']:.1%}",
            f"{risk['max_dd_duration_days']} days",
            f"{risk['downside_deviation']:.1%}",
        ],
    })
    st.dataframe(risk_table, use_container_width=True, hide_index=True)
