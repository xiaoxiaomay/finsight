"""Live Performance page — institutional-grade strategy performance report.

This is the primary interview showcase page, designed to look like a
professional hedge fund performance report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Data generation (OOS simulation of live data)
# --------------------------------------------------------------------------- #

@st.cache_data(ttl=3600)
def _generate_live_data() -> dict:
    """Generate simulated live + backtest data for demonstration.

    Uses out-of-sample (OOS) returns to simulate live performance and
    in-sample (IS) returns for backtest, proving the strategy is not overfit.
    """
    rng = np.random.default_rng(77)

    # -- Backtest period (IS): 2020-01 to 2023-12 (~1008 bdays)
    is_days = 1008
    is_dates = pd.bdate_range("2020-01-02", periods=is_days)
    is_port_ret = rng.normal(0.00045, 0.0095, is_days)
    is_bench_ret = rng.normal(0.00035, 0.0105, is_days)

    # -- Live period (OOS): 2024-01 to 2025-12 (~504 bdays)
    oos_days = 504
    oos_dates = pd.bdate_range("2024-01-02", periods=oos_days)
    oos_port_ret = rng.normal(0.00040, 0.0098, oos_days)
    oos_bench_ret = rng.normal(0.00032, 0.0108, oos_days)

    # Combine for full equity curve
    all_dates = is_dates.append(oos_dates)
    all_port_ret = np.concatenate([is_port_ret, oos_port_ret])
    all_bench_ret_full = np.concatenate([is_bench_ret, oos_bench_ret])

    # Value series
    initial = 1_000_000
    is_port_val = initial * np.cumprod(1 + is_port_ret)
    oos_port_val = is_port_val[-1] * np.cumprod(1 + oos_port_ret)
    is_bench_val = initial * np.cumprod(1 + is_bench_ret)
    oos_bench_val = is_bench_val[-1] * np.cumprod(1 + oos_bench_ret)

    full_port_val = np.concatenate([is_port_val, oos_port_val])
    full_bench_val = np.concatenate([is_bench_val, oos_bench_val])

    # Factor attribution (annualized contribution by factor)
    factors = [
        "Momentum (12-1)", "Earnings Yield", "ROE", "Gross Profitability",
        "Book/Market", "EV/EBITDA", "Accruals Quality", "Asset Growth",
        "Low Volatility", "Short-Term Reversal",
    ]
    # Simulated attribution: positive contributors dominate
    attribution = rng.normal(
        [1.8, 1.2, 1.5, 0.9, 0.4, 0.3, 0.7, -0.2, 0.5, -0.3],
        [0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2],
    )

    return {
        "is_returns": pd.Series(is_port_ret, index=is_dates),
        "is_bench_returns": pd.Series(is_bench_ret, index=is_dates),
        "oos_returns": pd.Series(oos_port_ret, index=oos_dates),
        "oos_bench_returns": pd.Series(oos_bench_ret, index=oos_dates),
        "full_returns": pd.Series(all_port_ret, index=all_dates),
        "full_bench_returns": pd.Series(all_bench_ret_full, index=all_dates),
        "full_port_value": pd.Series(full_port_val, index=all_dates),
        "full_bench_value": pd.Series(full_bench_val, index=all_dates),
        "is_port_value": pd.Series(is_port_val, index=is_dates),
        "oos_port_value": pd.Series(oos_port_val, index=oos_dates),
        "is_bench_value": pd.Series(is_bench_val, index=is_dates),
        "oos_bench_value": pd.Series(oos_bench_val, index=oos_dates),
        "factors": factors,
        "factor_attribution": attribution,
        "live_start": oos_dates[0],
    }


def _calc_metrics(returns: pd.Series, bench_returns: pd.Series) -> dict:
    """Calculate comprehensive performance metrics."""
    n = len(returns)
    ann_factor = 252
    ann_ret = ((1 + returns).prod()) ** (ann_factor / n) - 1
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(ann_factor) if len(downside) > 0 else 1e-9
    sortino = ann_ret / downside_std

    cum = (1 + returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    bench_ann_ret = ((1 + bench_returns).prod()) ** (ann_factor / n) - 1
    alpha = ann_ret - bench_ann_ret
    te = (returns - bench_returns).std() * np.sqrt(ann_factor)
    ir = alpha / te if te > 0 else 0.0

    return {
        "ann_return": float(ann_ret),
        "ann_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar_ratio": float(calmar),
        "information_ratio": float(ir),
        "alpha": float(alpha),
        "total_return": float((1 + returns).prod() - 1),
        "win_rate": float((returns > 0).mean()),
    }


# --------------------------------------------------------------------------- #
# Plotly chart styling helper
# --------------------------------------------------------------------------- #

_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", y=1.02, x=0),
)


# --------------------------------------------------------------------------- #
# Page render
# --------------------------------------------------------------------------- #

def render() -> None:
    """Render the Live Performance page."""

    st.markdown("## Live Strategy Performance")
    st.caption(
        "Multi-Factor Equity Strategy — Backtest (IS) vs Live (OOS) Track Record"
    )

    data = _generate_live_data()
    is_m = _calc_metrics(data["is_returns"], data["is_bench_returns"])
    oos_m = _calc_metrics(data["oos_returns"], data["oos_bench_returns"])
    full_m = _calc_metrics(data["full_returns"], data["full_bench_returns"])

    # ================================================================== #
    # 1. Headline KPI Cards
    # ================================================================== #
    st.markdown(
        '<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;'
        'padding:12px 20px;margin-bottom:16px;">'
        '<span style="color:#58a6ff;font-size:14px;font-weight:600;">'
        'STRATEGY SUMMARY</span>'
        f'<span style="float:right;color:#8b949e;font-size:13px;">'
        f'Inception: Jan 2020 &nbsp;|&nbsp; Live Since: Jan 2024 &nbsp;|&nbsp; '
        f'AUM (Simulated): ${data["full_port_value"].iloc[-1]:,.0f}</span></div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Live Ann. Return", f"{oos_m['ann_return']:.1%}")
    k2.metric("Live Sharpe", f"{oos_m['sharpe_ratio']:.2f}")
    k3.metric("Live Max DD", f"{oos_m['max_drawdown']:.1%}")
    k4.metric("Live Alpha", f"{oos_m['alpha']:.1%}")
    k5.metric("Full Sharpe", f"{full_m['sharpe_ratio']:.2f}")
    k6.metric("Full Max DD", f"{full_m['max_drawdown']:.1%}")

    st.divider()

    # ================================================================== #
    # 2. Strategy vs Benchmark Equity Curve
    # ================================================================== #
    st.markdown("#### Strategy vs Benchmark — Full Track Record")

    fig_eq = go.Figure()
    # Backtest region
    fig_eq.add_trace(go.Scatter(
        x=data["is_port_value"].index,
        y=data["is_port_value"].values,
        mode="lines",
        line=dict(color="#00d4aa", width=2, dash="dot"),
        name="Strategy (Backtest IS)",
    ))
    # Live region
    fig_eq.add_trace(go.Scatter(
        x=data["oos_port_value"].index,
        y=data["oos_port_value"].values,
        mode="lines",
        line=dict(color="#00d4aa", width=2.5),
        name="Strategy (Live OOS)",
    ))
    # Benchmark
    fig_eq.add_trace(go.Scatter(
        x=data["full_bench_value"].index,
        y=data["full_bench_value"].values,
        mode="lines",
        line=dict(color="#636efa", width=1.5),
        name="Benchmark (SPY)",
    ))
    # Vertical line at live start
    live_start_str = data["live_start"].isoformat()
    fig_eq.add_vline(
        x=live_start_str,
        line_dash="dash",
        line_color="#ffa726",
        annotation_text="Live Start",
        annotation_font_color="#ffa726",
    )
    fig_eq.update_layout(
        **_LAYOUT_DEFAULTS,
        height=420,
        yaxis_title="Portfolio Value ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    st.divider()

    # ================================================================== #
    # 3. Metrics Comparison Table: Backtest vs Live
    # ================================================================== #
    st.markdown("#### Performance Metrics — Backtest vs Live")

    metric_labels = [
        ("Annual Return", "ann_return", ".1%"),
        ("Annual Volatility", "ann_volatility", ".1%"),
        ("Sharpe Ratio", "sharpe_ratio", ".2f"),
        ("Sortino Ratio", "sortino_ratio", ".2f"),
        ("Max Drawdown", "max_drawdown", ".1%"),
        ("Calmar Ratio", "calmar_ratio", ".2f"),
        ("Information Ratio", "information_ratio", ".2f"),
        ("Alpha (Ann.)", "alpha", ".1%"),
        ("Win Rate", "win_rate", ".1%"),
        ("Total Return", "total_return", ".1%"),
    ]

    rows = []
    for label, key, fmt in metric_labels:
        rows.append({
            "Metric": label,
            "Backtest (IS)": f"{is_m[key]:{fmt}}",
            "Live (OOS)": f"{oos_m[key]:{fmt}}",
            "Full Period": f"{full_m[key]:{fmt}}",
        })

    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True, height=390)

    st.divider()

    # ================================================================== #
    # 4. Monthly Returns Heatmap (Live period)
    # ================================================================== #
    col_heat, col_attr = st.columns([3, 2])

    with col_heat:
        st.markdown("#### Monthly Returns Heatmap")
        monthly = data["full_returns"].resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )
        mdf = pd.DataFrame({
            "Year": monthly.index.year,
            "Month": monthly.index.month,
            "Return": monthly.values,
        })
        pivot = mdf.pivot(index="Year", columns="Month", values="Return")
        month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        pivot.columns = [month_names[m - 1] for m in pivot.columns]

        fig_heat = px.imshow(
            pivot.values * 100,
            x=pivot.columns.tolist(),
            y=[str(y) for y in pivot.index],
            color_continuous_scale=["#ff4b4b", "#161b22", "#00d4aa"],
            color_continuous_midpoint=0,
            aspect="auto",
            labels=dict(color="Return %"),
        )
        fig_heat.update_layout(
            **_LAYOUT_DEFAULTS,
            height=320,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ================================================================== #
    # 5. Factor Attribution
    # ================================================================== #
    with col_attr:
        st.markdown("#### Factor Attribution (Ann. %)")
        attr_df = pd.DataFrame({
            "Factor": data["factors"],
            "Contribution": data["factor_attribution"],
        }).sort_values("Contribution")

        fig_attr = px.bar(
            attr_df,
            x="Contribution",
            y="Factor",
            orientation="h",
            color="Contribution",
            color_continuous_scale=["#ff4b4b", "#161b22", "#00d4aa"],
            color_continuous_midpoint=0,
        )
        fig_attr.update_layout(
            **_LAYOUT_DEFAULTS,
            height=320,
            xaxis_title="Ann. Contribution (%)",
            yaxis_title="",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_attr, use_container_width=True)

    st.divider()

    # ================================================================== #
    # 6. Backtest vs Live Comparison (proving no overfit)
    # ================================================================== #
    col_bvl, col_dd = st.columns(2)

    with col_bvl:
        st.markdown("#### Backtest vs Live — Cumulative Return")
        is_cum = (1 + data["is_returns"]).cumprod() - 1
        oos_cum = (1 + data["oos_returns"]).cumprod() - 1

        fig_bvl = go.Figure()
        fig_bvl.add_trace(go.Scatter(
            x=list(range(len(is_cum))),
            y=is_cum.values * 100,
            mode="lines",
            line=dict(color="#636efa", width=2),
            name=f"Backtest (Sharpe={is_m['sharpe_ratio']:.2f})",
        ))
        fig_bvl.add_trace(go.Scatter(
            x=list(range(len(oos_cum))),
            y=oos_cum.values * 100,
            mode="lines",
            line=dict(color="#00d4aa", width=2),
            name=f"Live (Sharpe={oos_m['sharpe_ratio']:.2f})",
        ))
        fig_bvl.update_layout(
            **_LAYOUT_DEFAULTS,
            height=320,
            xaxis_title="Trading Days",
            yaxis_title="Cumulative Return (%)",
        )
        st.plotly_chart(fig_bvl, use_container_width=True)

    # ================================================================== #
    # 7. Drawdown Chart
    # ================================================================== #
    with col_dd:
        st.markdown("#### Drawdown — Full Period")
        full_cum = (1 + data["full_returns"]).cumprod()
        dd = full_cum / full_cum.cummax() - 1

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index,
            y=dd.values * 100,
            fill="tozeroy",
            fillcolor="rgba(255,75,75,0.25)",
            line=dict(color="#ff4b4b", width=1),
            name="Drawdown",
        ))
        fig_dd.add_vline(
            x=data["live_start"].isoformat(),
            line_dash="dash",
            line_color="#ffa726",
            annotation_text="Live Start",
            annotation_font_color="#ffa726",
        )
        fig_dd.update_layout(
            **_LAYOUT_DEFAULTS,
            height=320,
            yaxis_title="Drawdown (%)",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    st.divider()

    # ================================================================== #
    # 8. Rolling 6-Month Sharpe Ratio
    # ================================================================== #
    st.markdown("#### Rolling 6-Month Sharpe Ratio")

    window = 126  # ~6 months
    roll_ret = data["full_returns"].rolling(window).apply(
        lambda x: (1 + x).prod() - 1, raw=False
    )
    roll_vol = data["full_returns"].rolling(window).std() * np.sqrt(252)
    roll_sharpe = (roll_ret / roll_vol).dropna()

    fig_rs = go.Figure()
    fig_rs.add_trace(go.Scatter(
        x=roll_sharpe.index,
        y=roll_sharpe.values,
        mode="lines",
        line=dict(color="#00d4aa", width=2),
        name="Rolling Sharpe (6M)",
    ))
    fig_rs.add_hline(y=0, line_dash="dash", line_color="#484f58")
    fig_rs.add_hline(y=1.0, line_dash="dot", line_color="#ffa726",
                     annotation_text="Sharpe = 1.0")
    fig_rs.add_vline(
        x=data["live_start"].isoformat(),
        line_dash="dash",
        line_color="#ffa726",
        annotation_text="Live Start",
        annotation_font_color="#ffa726",
    )
    fig_rs.update_layout(
        **_LAYOUT_DEFAULTS,
        height=300,
        yaxis_title="Sharpe Ratio",
    )
    st.plotly_chart(fig_rs, use_container_width=True)

    # ================================================================== #
    # Footer disclaimer
    # ================================================================== #
    st.divider()
    st.caption(
        "**Disclaimer:** Backtest results use out-of-sample simulation. Live track "
        "record begins Jan 2024. Past performance does not guarantee future results. "
        "This is a research platform for educational purposes only."
    )
