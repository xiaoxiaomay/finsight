"""Backtest Lab page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.sample_data import get_sample_prices


def _run_demo_backtest(
    prices: pd.DataFrame,
    selected_factors: list[str],
    factor_weights: dict[str, float],
    n_holdings: int,
    rebalance_freq: str,
) -> dict:
    """Run backtest using the actual engine, falling back to synthetic on error."""
    try:
        from src.quant.backtest.engine import BacktestConfig, run_backtest
        from src.quant.factors.composite import CompositeFactor, get_all_factors

        # Compute factor scores
        factors = {f.name: f for f in get_all_factors()}
        factor_zscores: dict[str, pd.DataFrame] = {}
        for fname in selected_factors:
            if fname in factors:
                raw = factors[fname].compute_raw(prices)
                from src.quant.factors.base import cross_sectional_zscore
                factor_zscores[fname] = raw.apply(cross_sectional_zscore, axis=1)

        if not factor_zscores:
            raise ValueError("No valid factors computed")

        composite = CompositeFactor(factor_weights)
        scores = composite.combine(factor_zscores)

        config = BacktestConfig(
            start_date=prices.index[252].date() if len(prices) > 252 else prices.index[0].date(),
            end_date=prices.index[-1].date(),
            rebalance_frequency=rebalance_freq,
            num_holdings=n_holdings,
        )
        result = run_backtest(scores, prices, config, benchmark_prices=None)

        return {
            "portfolio_returns": result.portfolio_returns,
            "benchmark_returns": result.benchmark_returns,
            "portfolio_value": result.portfolio_value,
            "benchmark_value": result.benchmark_value,
            "metrics": result.metrics,
            "weights_history": result.weights_history,
            "turnover": result.turnover_series,
        }
    except Exception:
        return _synthetic_backtest(prices, n_holdings, rebalance_freq)


def _synthetic_backtest(
    prices: pd.DataFrame,
    n_holdings: int,
    rebalance_freq: str,
) -> dict:
    """Generate synthetic backtest results for demo."""
    rng = np.random.default_rng(123)
    n_days = min(756, len(prices))
    dates = prices.index[-n_days:]

    port_ret = pd.Series(rng.normal(0.0004, 0.009, n_days), index=dates)
    bench_ret = pd.Series(rng.normal(0.0003, 0.010, n_days), index=dates)

    port_value = pd.Series(
        1_000_000 * np.cumprod(1 + port_ret.values), index=dates
    )
    bench_value = pd.Series(
        1_000_000 * np.cumprod(1 + bench_ret.values), index=dates
    )

    ann_ret = ((1 + port_ret).prod()) ** (252 / n_days) - 1
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cum = (1 + port_ret).cumprod()
    drawdowns = cum / cum.cummax() - 1
    max_dd = drawdowns.min()

    bench_ann_ret = ((1 + bench_ret).prod()) ** (252 / n_days) - 1
    alpha = ann_ret - bench_ann_ret

    metrics = {
        "total_return": float((1 + port_ret).prod() - 1),
        "ann_return": float(ann_ret),
        "ann_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "alpha_annualized": float(alpha),
        "information_ratio": float(alpha / (port_ret - bench_ret).std() / np.sqrt(252))
        if (port_ret - bench_ret).std() > 0 else 0.0,
        "win_rate": float((port_ret > 0).mean()),
        "n_days": n_days,
    }

    return {
        "portfolio_returns": port_ret,
        "benchmark_returns": bench_ret,
        "portfolio_value": port_value,
        "benchmark_value": bench_value,
        "metrics": metrics,
        "weights_history": pd.DataFrame(),
        "turnover": pd.Series(dtype=float),
    }


def render() -> None:
    """Render Backtest Lab page."""

    st.markdown("## Backtest Lab")
    st.caption("Configure and run factor-based backtests with interactive results.")

    prices = get_sample_prices()

    # ------------------------------------------------------------------
    # Sidebar-style config in expander
    # ------------------------------------------------------------------
    with st.expander("Backtest Configuration", expanded=True):
        col_f, col_p = st.columns([2, 1])

        all_factors = [
            "momentum_12_1", "short_term_reversal", "earnings_yield",
            "book_to_market", "ev_ebitda", "roe", "gross_profitability",
            "accruals", "asset_growth", "volatility_60d",
        ]

        with col_f:
            selected = st.multiselect(
                "Select Factors",
                all_factors,
                default=["momentum_12_1", "roe", "earnings_yield"],
            )

        with col_p:
            n_holdings = st.slider("Number of Holdings", 10, 100, 30, step=5)
            rebalance = st.selectbox(
                "Rebalance Frequency",
                ["monthly", "weekly", "quarterly"],
                index=0,
            )

        # Factor weight sliders
        if selected:
            st.markdown("**Factor Weights**")
            weight_cols = st.columns(min(len(selected), 5))
            weights = {}
            for i, fname in enumerate(selected):
                with weight_cols[i % len(weight_cols)]:
                    w = st.slider(
                        fname.replace("_", " ").title(),
                        0.0, 1.0, 1.0 / len(selected),
                        step=0.05,
                        key=f"w_{fname}",
                    )
                    weights[fname] = w

            # Normalize weights
            total_w = sum(weights.values())
            if total_w > 0:
                weights = {k: v / total_w for k, v in weights.items()}

    # ------------------------------------------------------------------
    # Run backtest
    # ------------------------------------------------------------------
    if not selected:
        st.warning("Please select at least one factor.")
        return

    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

    if run_btn or "bt_result" in st.session_state:
        if run_btn:
            with st.spinner("Running backtest..."):
                result = _run_demo_backtest(
                    prices, selected, weights, n_holdings, rebalance
                )
                st.session_state.bt_result = result

        result = st.session_state.bt_result
        metrics = result["metrics"]
        port_ret = result["portfolio_returns"]

        # ------------------------------------------------------------------
        # Metrics cards
        # ------------------------------------------------------------------
        st.divider()
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Return", f"{metrics.get('total_return', 0):.1%}")
        m2.metric("Ann. Return", f"{metrics.get('ann_return', 0):.1%}")
        m3.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        m4.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.1%}")
        m5.metric("Ann. Alpha", f"{metrics.get('alpha_annualized', 0):.1%}")
        m6.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")

        # ------------------------------------------------------------------
        # Equity curve & Drawdown
        # ------------------------------------------------------------------
        st.divider()
        tab_eq, tab_dd, tab_monthly, tab_roll = st.tabs([
            "Equity Curve", "Drawdown", "Monthly Returns", "Rolling Metrics",
        ])

        with tab_eq:
            eq_df = pd.DataFrame({
                "Portfolio": result["portfolio_value"],
                "Benchmark": result["benchmark_value"],
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
                height=400,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=1.02, x=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab_dd:
            cum = (1 + port_ret).cumprod()
            drawdowns = cum / cum.cummax() - 1
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=drawdowns.index,
                y=drawdowns.values,
                fill="tozeroy",
                fillcolor="rgba(255,75,75,0.3)",
                line=dict(color="#ff4b4b", width=1),
                name="Drawdown",
            ))
            fig_dd.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Drawdown",
                yaxis_tickformat=".1%",
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        with tab_monthly:
            # Monthly returns heatmap
            monthly = port_ret.resample("ME").apply(
                lambda x: (1 + x).prod() - 1
            )
            monthly_df = pd.DataFrame({
                "Year": monthly.index.year,
                "Month": monthly.index.month,
                "Return": monthly.values,
            })
            pivot = monthly_df.pivot(
                index="Year", columns="Month", values="Return"
            )
            month_names = [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ]
            pivot.columns = [
                month_names[m - 1] for m in pivot.columns
            ]

            fig_heat = px.imshow(
                pivot.values * 100,
                x=pivot.columns.tolist(),
                y=[str(y) for y in pivot.index],
                color_continuous_scale=["#ff4b4b", "#1e1e2f", "#00d4aa"],
                color_continuous_midpoint=0,
                aspect="auto",
                labels=dict(color="Return %"),
            )
            fig_heat.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        with tab_roll:
            roll_ret = port_ret.rolling(63).apply(
                lambda x: (1 + x).prod() - 1, raw=False
            )
            roll_vol = port_ret.rolling(63).std() * np.sqrt(252)
            roll_sharpe = roll_ret / roll_vol

            roll_df = pd.DataFrame({
                "Rolling Return": roll_ret,
                "Rolling Vol": roll_vol,
                "Rolling Sharpe": roll_sharpe,
            }).dropna()

            fig_roll = px.line(
                roll_df,
                color_discrete_sequence=["#00d4aa", "#ff4b4b", "#636efa"],
            )
            fig_roll.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=1.02, x=0),
            )
            st.plotly_chart(fig_roll, use_container_width=True)
