#!/usr/bin/env python3
"""Generate a standalone HTML performance tearsheet.

Usage:
    python scripts/generate_tearsheet.py
    python scripts/generate_tearsheet.py --output reports/tearsheet.html
    python scripts/generate_tearsheet.py --strategy "Custom Alpha" --email "you@example.com"
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _generate_data() -> dict:
    """Generate simulated strategy data for the tearsheet."""
    rng = np.random.default_rng(77)

    # Backtest (IS) + Live (OOS)
    is_days, oos_days = 1008, 504
    is_dates = pd.bdate_range("2020-01-02", periods=is_days)
    oos_dates = pd.bdate_range("2024-01-02", periods=oos_days)
    all_dates = is_dates.append(oos_dates)

    is_port = rng.normal(0.00045, 0.0095, is_days)
    oos_port = rng.normal(0.00040, 0.0098, oos_days)
    is_bench = rng.normal(0.00035, 0.0105, is_days)
    oos_bench = rng.normal(0.00032, 0.0108, oos_days)

    port_ret = np.concatenate([is_port, oos_port])
    bench_ret = np.concatenate([is_bench, oos_bench])

    initial = 1_000_000
    port_val = initial * np.cumprod(1 + port_ret)
    bench_val = initial * np.cumprod(1 + bench_ret)

    factors = [
        "Momentum (12-1)", "Earnings Yield", "ROE", "Gross Profitability",
        "Book/Market", "EV/EBITDA", "Accruals", "Asset Growth",
        "Low Volatility", "Reversal",
    ]
    attribution = rng.normal(
        [1.8, 1.2, 1.5, 0.9, 0.4, 0.3, 0.7, -0.2, 0.5, -0.3],
        [0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2],
    )

    return {
        "port_ret": pd.Series(port_ret, index=all_dates),
        "bench_ret": pd.Series(bench_ret, index=all_dates),
        "port_val": pd.Series(port_val, index=all_dates),
        "bench_val": pd.Series(bench_val, index=all_dates),
        "is_ret": pd.Series(is_port, index=is_dates),
        "oos_ret": pd.Series(oos_port, index=oos_dates),
        "is_bench": pd.Series(is_bench, index=is_dates),
        "oos_bench": pd.Series(oos_bench, index=oos_dates),
        "factors": factors,
        "attribution": attribution,
        "live_start": oos_dates[0],
    }


def _calc(ret: pd.Series, bench: pd.Series) -> dict:
    n = len(ret)
    ann_ret = ((1 + ret).prod()) ** (252 / n) - 1
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    ds = ret[ret < 0]
    ds_std = ds.std() * np.sqrt(252) if len(ds) > 0 else 1e-9
    sortino = ann_ret / ds_std
    cum = (1 + ret).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    b_ann = ((1 + bench).prod()) ** (252 / n) - 1
    alpha = ann_ret - b_ann
    te = (ret - bench).std() * np.sqrt(252)
    ir = alpha / te if te > 0 else 0
    return {
        "ann_return": ann_ret, "ann_volatility": ann_vol, "sharpe": sharpe,
        "sortino": sortino, "max_drawdown": max_dd, "calmar": calmar,
        "info_ratio": ir, "alpha": alpha, "total_return": (1 + ret).prod() - 1,
        "win_rate": (ret > 0).mean(),
    }


def _heatmap_style(val: float) -> str:
    if val > 5:
        return "background:rgba(0,212,170,0.35);color:#fff"
    if val > 0:
        return "background:rgba(0,212,170,0.15);color:#e0e0e0"
    if val > -5:
        return "background:rgba(255,75,75,0.15);color:#e0e0e0"
    return "background:rgba(255,75,75,0.35);color:#fff"


def _svg_line(values: np.ndarray, w: int, h: int, color: str, stroke: float = 2) -> str:
    n = len(values)
    y_min, y_max = values.min() * 0.95, values.max() * 1.05
    rng = y_max - y_min if y_max != y_min else 1
    pts = []
    for i, v in enumerate(values):
        x = 40 + (i / max(n - 1, 1)) * (w - 80)
        y = h - 30 - ((v - y_min) / rng) * (h - 60)
        pts.append(f"{x:.1f},{y:.1f}")
    return f'<polyline fill="none" stroke="{color}" stroke-width="{stroke}" points="{" ".join(pts)}"/>'


def generate_tearsheet_html(
    strategy_name: str = "Multi-Factor Equity",
    email: str = "",
) -> str:
    """Generate standalone HTML tearsheet."""
    data = _generate_data()
    full = _calc(data["port_ret"], data["bench_ret"])
    is_m = _calc(data["is_ret"], data["is_bench"])
    oos_m = _calc(data["oos_ret"], data["oos_bench"])

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    start = data["port_ret"].index[0].strftime("%Y-%m-%d")
    end = data["port_ret"].index[-1].strftime("%Y-%m-%d")

    # Monthly returns heatmap
    monthly = data["port_ret"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    mdf = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values * 100,
    })
    pivot = mdf.pivot(index="Year", columns="Month", values="Return")
    mnames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    monthly_html = '<table><tr><th>Year</th>'
    for m in range(1, 13):
        monthly_html += f'<th>{mnames[m-1]}</th>'
    monthly_html += '<th>Annual</th></tr>\n'
    for year in pivot.index:
        monthly_html += f'<tr><td><b>{year}</b></td>'
        yr_total = 0
        for m in range(1, 13):
            if m in pivot.columns:
                v = pivot.loc[year, m]
                if pd.notna(v):
                    monthly_html += f'<td style="{_heatmap_style(v)}">{v:+.1f}%</td>'
                    yr_total += v
                else:
                    monthly_html += '<td></td>'
            else:
                monthly_html += '<td></td>'
        monthly_html += f'<td style="font-weight:bold;{_heatmap_style(yr_total)}">{yr_total:+.1f}%</td>'
        monthly_html += '</tr>\n'
    monthly_html += '</table>'

    # Equity SVG
    w, h = 1060, 280
    pv = data["port_val"].values / data["port_val"].values[0] * 100
    bv = data["bench_val"].values / data["bench_val"].values[0] * 100
    eq_svg = f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:{h}px;">\n'
    eq_svg += _svg_line(bv, w, h, "#636efa", 1.5) + '\n'
    eq_svg += _svg_line(pv, w, h, "#00d4aa", 2) + '\n'
    # Live start marker
    n_is = len(data["is_ret"])
    n_total = len(data["port_ret"])
    x_live = 40 + (n_is / max(n_total - 1, 1)) * (w - 80)
    eq_svg += f'<line x1="{x_live:.0f}" y1="10" x2="{x_live:.0f}" y2="{h-20}" stroke="#ffa726" stroke-dasharray="4" stroke-width="1.5"/>\n'
    eq_svg += f'<text x="{x_live+4:.0f}" y="20" font-size="11" fill="#ffa726">Live Start</text>\n'
    eq_svg += f'<text x="{w-40}" y="20" font-size="11" fill="#00d4aa" text-anchor="end">Strategy</text>\n'
    eq_svg += f'<text x="{w-40}" y="35" font-size="11" fill="#636efa" text-anchor="end">Benchmark</text>\n'
    eq_svg += '</svg>'

    # Drawdown SVG
    cum = (1 + data["port_ret"]).cumprod()
    dd = (cum / cum.cummax() - 1).values
    dd_svg = f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:{h}px;">\n'
    dd_min = dd.min() * 1.1
    n = len(dd)
    pts = []
    for i, v in enumerate(dd):
        x = 40 + (i / max(n - 1, 1)) * (w - 80)
        y = 30 + ((-v) / max(-dd_min, 0.01)) * (h - 60)
        pts.append(f"{x:.1f},{y:.1f}")
    area = "40,30 " + " ".join(pts) + f" {w-40},30"
    dd_svg += f'<polygon fill="rgba(255,75,75,0.15)" points="{area}"/>\n'
    dd_svg += f'<polyline fill="none" stroke="#ff4b4b" stroke-width="1.5" points="{" ".join(pts)}"/>\n'
    dd_svg += f'<line x1="40" y1="30" x2="{w-40}" y2="30" stroke="#333" stroke-width="0.5"/>\n'
    dd_svg += f'<text x="45" y="{h-10}" font-size="11" fill="#ff4b4b">Max DD: {dd.min():.1%}</text>\n'
    dd_svg += '</svg>'

    # Factor attribution
    fa = sorted(zip(data["factors"], data["attribution"], strict=False), key=lambda x: x[1], reverse=True)
    attr_html = '<table><tr><th>Factor</th><th>Contribution (Ann. %)</th><th></th></tr>\n'
    max_abs = max(abs(a) for _, a in fa) if fa else 1
    for name, val in fa:
        color = "#00d4aa" if val >= 0 else "#ff4b4b"
        bar_w = abs(val) / max_abs * 200
        attr_html += f'<tr><td>{name}</td><td style="color:{color}">{val:+.2f}%</td>'
        attr_html += f'<td><div style="background:{color};width:{bar_w:.0f}px;height:14px;border-radius:2px;"></div></td></tr>\n'
    attr_html += '</table>'

    # Build metrics row helper
    def _mc(label: str, value: str, cls: str = "") -> str:
        return (f'<div class="mc"><div class="mv {cls}">{value}</div>'
                f'<div class="ml">{label}</div></div>')

    contact = f'<br>Contact: {email}' if email else ''

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FinSight Performance Tearsheet — {strategy_name}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 24px;
         background: #0d1117; color: #c9d1d9; }}
  h1 {{ color: #58a6ff; border-bottom: 2px solid #21262d; padding-bottom: 8px; margin-top: 0; }}
  h2 {{ color: #c9d1d9; border-bottom: 1px solid #21262d; padding-bottom: 6px; margin-top: 36px; }}
  .header {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px;
             padding: 20px 24px; margin-bottom: 24px; }}
  .header p {{ color: #8b949e; margin: 4px 0; }}
  .mg {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 16px 0; }}
  .mc {{ background: #161b22; border: 1px solid #21262d; border-radius: 6px;
         padding: 14px; text-align: center; }}
  .mv {{ font-size: 22px; font-weight: bold; color: #58a6ff; }}
  .mv.pos {{ color: #3fb950; }}
  .mv.neg {{ color: #f85149; }}
  .ml {{ font-size: 11px; color: #8b949e; margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #21262d; padding: 8px 12px; text-align: right; }}
  th {{ background: #161b22; font-weight: 600; color: #c9d1d9; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .chart {{ background: #161b22; border: 1px solid #21262d; border-radius: 6px;
            padding: 16px; margin: 12px 0; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .disclaimer {{ font-size: 11px; color: #484f58; margin-top: 32px; padding: 12px;
                 background: #161b22; border-radius: 6px; }}
  .footer {{ text-align: center; color: #484f58; font-size: 11px; margin-top: 16px; }}
  @media print {{
    body {{ background: #fff; color: #24292e; }}
    .mc, .chart, .header {{ border-color: #d1d5da; background: #f6f8fa; }}
    .mv {{ color: #0366d6; }}
    th {{ background: #f6f8fa; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>FinSight Quantitative Research Platform</h1>
  <p><strong>Strategy:</strong> {strategy_name} &nbsp;|&nbsp;
     <strong>Period:</strong> {start} to {end} &nbsp;|&nbsp;
     <strong>Live Since:</strong> Jan 2024</p>
  <p><strong>Generated:</strong> {now}{contact}</p>
</div>

<h2>Key Performance Metrics</h2>
<div class="mg">
  {_mc("Ann. Return (Full)", f"{full['ann_return']:.1%}", "pos" if full['ann_return'] > 0 else "neg")}
  {_mc("Sharpe Ratio", f"{full['sharpe']:.2f}", "pos" if full['sharpe'] > 0.5 else "")}
  {_mc("Max Drawdown", f"{full['max_drawdown']:.1%}", "neg")}
  {_mc("Sortino Ratio", f"{full['sortino']:.2f}", "")}
  {_mc("Calmar Ratio", f"{full['calmar']:.2f}", "")}
</div>
<div class="mg">
  {_mc("Alpha (Ann.)", f"{full['alpha']:.1%}", "pos" if full['alpha'] > 0 else "neg")}
  {_mc("Info Ratio", f"{full['info_ratio']:.2f}", "")}
  {_mc("Win Rate", f"{full['win_rate']:.1%}", "")}
  {_mc("Ann. Volatility", f"{full['ann_volatility']:.1%}", "")}
  {_mc("Total Return", f"{full['total_return']:.1%}", "pos")}
</div>

<h2>Backtest vs Live Comparison</h2>
<table>
<tr><th>Metric</th><th>Backtest (IS)</th><th>Live (OOS)</th><th>Full Period</th></tr>
<tr><td>Annual Return</td><td>{is_m['ann_return']:.1%}</td><td>{oos_m['ann_return']:.1%}</td><td>{full['ann_return']:.1%}</td></tr>
<tr><td>Sharpe Ratio</td><td>{is_m['sharpe']:.2f}</td><td>{oos_m['sharpe']:.2f}</td><td>{full['sharpe']:.2f}</td></tr>
<tr><td>Sortino Ratio</td><td>{is_m['sortino']:.2f}</td><td>{oos_m['sortino']:.2f}</td><td>{full['sortino']:.2f}</td></tr>
<tr><td>Max Drawdown</td><td>{is_m['max_drawdown']:.1%}</td><td>{oos_m['max_drawdown']:.1%}</td><td>{full['max_drawdown']:.1%}</td></tr>
<tr><td>Calmar Ratio</td><td>{is_m['calmar']:.2f}</td><td>{oos_m['calmar']:.2f}</td><td>{full['calmar']:.2f}</td></tr>
<tr><td>Information Ratio</td><td>{is_m['info_ratio']:.2f}</td><td>{oos_m['info_ratio']:.2f}</td><td>{full['info_ratio']:.2f}</td></tr>
<tr><td>Alpha (Ann.)</td><td>{is_m['alpha']:.1%}</td><td>{oos_m['alpha']:.1%}</td><td>{full['alpha']:.1%}</td></tr>
</table>

<h2>Equity Curve — Strategy vs Benchmark</h2>
<div class="chart">
{eq_svg}
</div>

<h2>Drawdown</h2>
<div class="chart">
{dd_svg}
</div>

<h2>Monthly Returns (%)</h2>
{monthly_html}

<h2>Factor Attribution</h2>
{attr_html}

<div class="disclaimer">
<strong>Disclaimer:</strong> FinSight is an AI-powered research and analysis tool for
educational and informational purposes only. It does not constitute investment advice.
Past performance does not guarantee future results. Backtest results include simulated
out-of-sample period to validate strategy robustness.
</div>

<div class="footer">
  FinSight Quantitative Research Platform &mdash; Generated on {now}{contact}
</div>

</body>
</html>"""

    return html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FinSight performance tearsheet")
    parser.add_argument(
        "--output", "-o",
        default="reports/tearsheet.html",
        help="Output HTML file path (default: reports/tearsheet.html)",
    )
    parser.add_argument(
        "--strategy", "-s",
        default="Multi-Factor Equity",
        help="Strategy name for the report header",
    )
    parser.add_argument(
        "--email", "-e",
        default="",
        help="Contact email for the footer",
    )
    args = parser.parse_args()

    html = generate_tearsheet_html(
        strategy_name=args.strategy,
        email=args.email,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(f"Tearsheet generated: {output} ({len(html):,} bytes)")
    print(f"Open in browser: file://{output.resolve()}")


if __name__ == "__main__":
    main()
