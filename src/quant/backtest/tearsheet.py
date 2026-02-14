"""Performance tearsheet generator.

Produces an HTML report with:
- Equity curve (portfolio vs benchmark)
- Drawdown chart
- Monthly returns heatmap
- Key metrics summary table
- Walk-forward fold results (if available)
"""

from pathlib import Path

import pandas as pd

from src.config.logging_config import get_logger
from src.quant.backtest.engine import BacktestResult
from src.quant.backtest.statistics import (
    alpha_significance,
    drawdown_analysis,
    monthly_returns_table,
)
from src.quant.backtest.walk_forward import WalkForwardResult

logger = get_logger("backtest.tearsheet")

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FinSight Performance Tearsheet</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafbfc; color: #24292e; }}
  h1 {{ color: #0366d6; border-bottom: 2px solid #0366d6; padding-bottom: 8px; }}
  h2 {{ color: #24292e; margin-top: 40px; border-bottom: 1px solid #e1e4e8; padding-bottom: 6px; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }}
  .metric-card {{ background: white; border: 1px solid #e1e4e8; border-radius: 6px; padding: 16px; text-align: center; }}
  .metric-value {{ font-size: 24px; font-weight: bold; color: #0366d6; }}
  .metric-label {{ font-size: 12px; color: #586069; margin-top: 4px; }}
  .positive {{ color: #22863a; }}
  .negative {{ color: #cb2431; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; background: white; }}
  th, td {{ border: 1px solid #e1e4e8; padding: 8px 12px; text-align: right; }}
  th {{ background: #f6f8fa; font-weight: 600; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .heatmap-pos {{ background: rgba(34, 134, 58, 0.15); }}
  .heatmap-neg {{ background: rgba(203, 36, 49, 0.15); }}
  .heatmap-strong-pos {{ background: rgba(34, 134, 58, 0.35); }}
  .heatmap-strong-neg {{ background: rgba(203, 36, 49, 0.35); }}
  .chart {{ width: 100%; height: 300px; background: white; border: 1px solid #e1e4e8; border-radius: 6px; padding: 16px; margin: 16px 0; overflow-x: auto; }}
  .svg-chart {{ width: 100%; height: 280px; }}
  .disclaimer {{ font-size: 11px; color: #586069; margin-top: 40px; padding: 12px; background: #f6f8fa; border-radius: 6px; }}
  .fold-table th {{ text-align: center; }}
  .fold-table td {{ text-align: center; }}
</style>
</head>
<body>

<h1>FinSight Performance Tearsheet</h1>
<p><strong>Strategy:</strong> {strategy_name} &nbsp;|&nbsp;
   <strong>Period:</strong> {start_date} to {end_date} &nbsp;|&nbsp;
   <strong>Generated:</strong> {generated_date}</p>

<h2>Key Metrics</h2>
<div class="metrics-grid">
{metrics_cards}
</div>

<h2>Equity Curve</h2>
<div class="chart">
{equity_chart}
</div>

<h2>Drawdown</h2>
<div class="chart">
{drawdown_chart}
</div>

<h2>Statistical Significance</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{significance_rows}
</table>

<h2>Monthly Returns (%)</h2>
{monthly_table}

{walk_forward_section}

<div class="disclaimer">
FinSight is an AI-powered research tool for educational purposes only.
Past performance does not guarantee future results. Not investment advice.
</div>

</body>
</html>"""


def generate_tearsheet(
    result: BacktestResult,
    strategy_name: str = "Multi-Factor Equity",
    wf_result: WalkForwardResult | None = None,
    output_path: str | Path | None = None,
) -> str:
    """Generate an HTML performance tearsheet.

    Args:
        result: BacktestResult from a completed backtest.
        strategy_name: Name of the strategy.
        wf_result: Optional walk-forward results.
        output_path: Where to save the HTML file.

    Returns:
        HTML string.
    """
    metrics = result.metrics
    returns = result.portfolio_returns
    bench_returns = result.benchmark_returns

    # Metric cards
    cards = [
        _metric_card("Ann. Return", f"{metrics.get('ann_return', 0):.1%}",
                      "positive" if metrics.get("ann_return", 0) > 0 else "negative"),
        _metric_card("Ann. Volatility", f"{metrics.get('ann_volatility', 0):.1%}", ""),
        _metric_card("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}",
                      "positive" if metrics.get("sharpe_ratio", 0) > 0.5 else ""),
        _metric_card("Max Drawdown", f"{metrics.get('max_drawdown', 0):.1%}", "negative"),
        _metric_card("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}", ""),
        _metric_card("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.2f}", ""),
        _metric_card("Info Ratio", f"{metrics.get('information_ratio', 0):.2f}", ""),
        _metric_card("Win Rate", f"{metrics.get('win_rate', 0):.1%}", ""),
    ]
    metrics_cards = "\n".join(cards)

    # Equity curve (SVG)
    equity_chart = _render_equity_svg(result.portfolio_value, result.benchmark_value)

    # Drawdown chart
    dd = drawdown_analysis(returns)
    drawdown_chart = _render_drawdown_svg(dd["drawdown_series"])

    # Statistical significance
    sig = alpha_significance(returns, bench_returns)
    sig_rows = "\n".join([
        f"<tr><td>Alpha (annualized)</td><td>{sig['alpha_annualized']:.2%}</td></tr>",
        f"<tr><td>t-statistic</td><td>{sig['t_stat']:.2f}</td></tr>",
        f"<tr><td>p-value</td><td>{sig['p_value']:.4f}</td></tr>",
        f"<tr><td>Significant (5%)</td><td>{'Yes' if sig['significant'] else 'No'}</td></tr>",
        f"<tr><td>Observations</td><td>{sig['n_obs']}</td></tr>",
    ])

    # Monthly returns heatmap
    monthly = monthly_returns_table(returns)
    monthly_html = _render_monthly_table(monthly)

    # Walk-forward section
    wf_section = ""
    if wf_result:
        wf_section = _render_walk_forward_section(wf_result)

    # Determine dates
    start_date = returns.index[0].strftime("%Y-%m-%d") if len(returns) > 0 else "N/A"
    end_date = returns.index[-1].strftime("%Y-%m-%d") if len(returns) > 0 else "N/A"

    html = HTML_TEMPLATE.format(
        strategy_name=strategy_name,
        start_date=start_date,
        end_date=end_date,
        generated_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        metrics_cards=metrics_cards,
        equity_chart=equity_chart,
        drawdown_chart=drawdown_chart,
        significance_rows=sig_rows,
        monthly_table=monthly_html,
        walk_forward_section=wf_section,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)
        logger.info("tearsheet_saved", path=str(output_path))

    return html


def _metric_card(label: str, value: str, css_class: str) -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-value {css_class}">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>'
    )


def _render_equity_svg(
    portfolio_value: pd.Series,
    benchmark_value: pd.Series,
) -> str:
    """Render equity curve as inline SVG."""
    if portfolio_value.empty:
        return "<p>No data available.</p>"

    width, height = 1060, 260
    margin = 40

    # Normalize both to start at same value
    pv = portfolio_value / portfolio_value.iloc[0] * 100
    bv = benchmark_value / benchmark_value.iloc[0] * 100 if not benchmark_value.empty else pv * 0

    all_vals = pd.concat([pv, bv]).dropna()
    y_min = all_vals.min() * 0.95
    y_max = all_vals.max() * 1.05

    def to_svg_coords(series: pd.Series) -> str:
        n = len(series)
        points = []
        for i, (_, v) in enumerate(series.items()):
            x = margin + (i / max(n - 1, 1)) * (width - 2 * margin)
            y = height - margin - ((v - y_min) / max(y_max - y_min, 1)) * (height - 2 * margin)
            points.append(f"{x:.1f},{y:.1f}")
        return " ".join(points)

    port_points = to_svg_coords(pv)
    bench_points = to_svg_coords(bv) if not bv.empty else ""

    svg = f'<svg class="svg-chart" viewBox="0 0 {width} {height}">\n'
    if bench_points:
        svg += f'  <polyline fill="none" stroke="#959da5" stroke-width="1.5" points="{bench_points}"/>\n'
    svg += f'  <polyline fill="none" stroke="#0366d6" stroke-width="2" points="{port_points}"/>\n'
    svg += f'  <text x="{width - margin}" y="20" font-size="11" fill="#0366d6" text-anchor="end">Portfolio</text>\n'
    if bench_points:
        svg += f'  <text x="{width - margin}" y="35" font-size="11" fill="#959da5" text-anchor="end">Benchmark</text>\n'
    svg += '</svg>'

    return svg


def _render_drawdown_svg(drawdown: pd.Series) -> str:
    """Render drawdown chart as inline SVG."""
    if drawdown.empty:
        return "<p>No data available.</p>"

    width, height = 1060, 260
    margin = 40
    n = len(drawdown)
    y_min = drawdown.min() * 1.1

    points = []
    for i, (_, v) in enumerate(drawdown.items()):
        x = margin + (i / max(n - 1, 1)) * (width - 2 * margin)
        y = margin + ((-v) / max(-y_min, 0.01)) * (height - 2 * margin)
        points.append(f"{x:.1f},{y:.1f}")

    area_points = f"{margin},{margin} " + " ".join(points) + f" {width - margin},{margin}"

    svg = f'<svg class="svg-chart" viewBox="0 0 {width} {height}">\n'
    svg += f'  <polygon fill="rgba(203,36,49,0.15)" points="{area_points}"/>\n'
    svg += f'  <polyline fill="none" stroke="#cb2431" stroke-width="1.5" points="{" ".join(points)}"/>\n'
    svg += f'  <line x1="{margin}" y1="{margin}" x2="{width-margin}" y2="{margin}" stroke="#e1e4e8" stroke-width="1"/>\n'
    svg += f'  <text x="{margin + 5}" y="{height - 10}" font-size="11" fill="#cb2431">Max DD: {drawdown.min():.1%}</text>\n'
    svg += '</svg>'

    return svg


def _render_monthly_table(monthly: pd.DataFrame) -> str:
    """Render monthly returns as HTML table with heatmap coloring."""
    html = '<table>\n<tr><th>Year</th>'
    for col in monthly.columns:
        html += f'<th>{col}</th>'
    html += '</tr>\n'

    for year in monthly.index:
        html += f'<tr><td><strong>{year}</strong></td>'
        for col in monthly.columns:
            val = monthly.loc[year, col]
            if pd.isna(val):
                html += '<td></td>'
            else:
                css = _heatmap_class(val)
                html += f'<td class="{css}">{val:.1%}</td>'
        html += '</tr>\n'

    html += '</table>'
    return html


def _heatmap_class(val: float) -> str:
    if val > 0.05:
        return "heatmap-strong-pos"
    elif val > 0:
        return "heatmap-pos"
    elif val > -0.05:
        return "heatmap-neg"
    else:
        return "heatmap-strong-neg"


def _render_walk_forward_section(wf_result: WalkForwardResult) -> str:
    """Render walk-forward validation results."""
    html = '<h2>Walk-Forward Validation</h2>\n'

    # Summary
    oos = wf_result.oos_combined_metrics
    html += f'<p><strong>Combined OOS Sharpe:</strong> {oos.get("sharpe_ratio", "N/A")} &nbsp;|&nbsp; '
    html += f'<strong>OOS Ann. Return:</strong> {oos.get("ann_return", 0):.1%} &nbsp;|&nbsp; '
    html += f'<strong>OOS Max Drawdown:</strong> {oos.get("max_drawdown", 0):.1%}</p>\n'

    # Per-fold table
    html += '<table class="fold-table">\n'
    html += '<tr><th>Fold</th><th>IS Sharpe</th><th>OOS Sharpe</th>'
    html += '<th>IS Return</th><th>OOS Return</th><th>OOS Max DD</th></tr>\n'

    for fm in wf_result.per_fold_metrics:
        html += '<tr>'
        html += f'<td>{fm["fold"]}</td>'
        html += f'<td>{fm["is_sharpe"]:.2f}</td>'
        html += f'<td>{fm["oos_sharpe"]:.2f}</td>'
        html += f'<td>{fm["is_return"]:.1%}</td>'
        html += f'<td>{fm["oos_return"]:.1%}</td>'
        html += f'<td>{fm["oos_max_dd"]:.1%}</td>'
        html += '</tr>\n'

    html += '</table>\n'
    return html
