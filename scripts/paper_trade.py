#!/usr/bin/env python3
"""Paper trading signal generator.

Reads the latest portfolio signals and generates trade recommendations.
Records trades to a log file for track record building.

Usage:
    python scripts/paper_trade.py
    python scripts/paper_trade.py --capital 500000
    python scripts/paper_trade.py --max-position-pct 5
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd


def load_latest_snapshot() -> dict | None:
    """Load the most recent portfolio snapshot."""
    snapshot_dir = Path("data/snapshots")
    if not snapshot_dir.exists():
        return None
    files = sorted(snapshot_dir.glob("snapshot_*.json"), reverse=True)
    if not files:
        return None
    return json.loads(files[0].read_text())


def generate_trades(
    snapshot: dict,
    capital: float = 1_000_000,
    max_position_pct: float = 5.0,
) -> list[dict]:
    """Generate trade recommendations from signals."""
    max_alloc = capital * (max_position_pct / 100)
    n_long = len(snapshot.get("long", {}))
    per_position = min(capital / max(n_long, 1), max_alloc)

    trades = []
    for symbol, score in snapshot.get("long", {}).items():
        # Estimate shares (use placeholder price for demo)
        est_price = 150.0  # Would come from market data in production
        shares = int(per_position / est_price)
        if shares > 0:
            trades.append({
                "date": str(date.today()),
                "symbol": symbol,
                "side": "BUY",
                "shares": shares,
                "est_price": est_price,
                "notional": round(shares * est_price, 2),
                "signal_score": round(score, 4),
                "reason": "Long — top factor composite score",
            })

    return trades


def record_trades(trades: list[dict]) -> None:
    """Append trades to the trade log."""
    log_dir = Path("data/trades")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "paper_trades.jsonl"
    with open(log_file, "a") as f:
        for trade in trades:
            f.write(json.dumps(trade) + "\n")

    print(f"Recorded {len(trades)} trades to {log_file}")


def print_trade_summary(trades: list[dict]) -> None:
    """Display trade summary."""
    if not trades:
        print("No trades generated.")
        return

    df = pd.DataFrame(trades)
    total_notional = df["notional"].sum()

    print("\n" + "=" * 70)
    print(f"PAPER TRADE RECOMMENDATIONS — {date.today()}")
    print("=" * 70)
    print(f"{'Symbol':<8} {'Side':<6} {'Shares':>8} {'Price':>10} {'Notional':>12} {'Score':>8}")
    print("-" * 70)
    for t in trades:
        print(
            f"{t['symbol']:<8} {t['side']:<6} {t['shares']:>8} "
            f"${t['est_price']:>9.2f} ${t['notional']:>11,.2f} {t['signal_score']:>8.4f}"
        )
    print("-" * 70)
    print(f"{'Total':>36} ${total_notional:>11,.2f}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="FinSight paper trade generator")
    parser.add_argument(
        "--capital", type=float, default=1_000_000,
        help="Total capital allocation (default: $1M)",
    )
    parser.add_argument(
        "--max-position-pct", type=float, default=5.0,
        help="Max position size as %% of capital (default: 5%%)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show trades without recording",
    )
    args = parser.parse_args()

    snapshot = load_latest_snapshot()
    if not snapshot:
        print("No portfolio snapshot found. Run `python scripts/daily_update.py` first.")
        return

    print(f"Loading snapshot from: {snapshot.get('date', 'unknown')}")
    print(f"Capital: ${args.capital:,.0f} | Max position: {args.max_position_pct}%")

    trades = generate_trades(
        snapshot,
        capital=args.capital,
        max_position_pct=args.max_position_pct,
    )

    print_trade_summary(trades)

    if not args.dry_run and trades:
        record_trades(trades)
    elif args.dry_run:
        print("\n(Dry run — trades not recorded)")


if __name__ == "__main__":
    main()
