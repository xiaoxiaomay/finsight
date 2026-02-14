"""CLI for running factor backtests.

Usage:
    python scripts/run_backtest.py --factor momentum_12_1
    python scripts/run_backtest.py --factor composite --start 2021-01-01
    python scripts/run_backtest.py --factor momentum_12_1 --walk-forward
    python scripts/run_backtest.py --list-factors

Generates an HTML tearsheet in reports/ directory.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import get_logger, setup_logging

logger = get_logger("run_backtest")


def generate_sample_data(
    n_symbols: int = 100,
    n_days: int = 1260,  # ~5 years
    start_date: str = "2020-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Generate realistic synthetic market data for backtesting.

    Creates correlated stock prices with sector structure and varying
    volatilities to simulate a realistic equity universe.
    """
    np.random.seed(42)
    dates = pd.bdate_range(start_date, periods=n_days)
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]

    # Sector structure (10 sectors, ~10 stocks each)
    sectors = {}
    sector_names = [
        "Tech", "Health", "Finance", "Consumer", "Energy",
        "Industrial", "Materials", "Utilities", "RealEstate", "Telecom",
    ]
    for i, sym in enumerate(symbols):
        sectors[sym] = sector_names[i % len(sector_names)]

    # Generate correlated returns
    # Market factor + sector factors + idiosyncratic
    market_returns = np.random.normal(0.0003, 0.01, n_days)  # ~7.5% annual

    all_returns = np.zeros((n_days, n_symbols))
    for j, _sym in enumerate(symbols):
        sector_returns = np.random.normal(0, 0.005, n_days)

        # Varying beta (0.5 to 1.5)
        beta = 0.7 + (j % 10) * 0.1
        idio_vol = 0.005 + (j % 5) * 0.003

        stock_returns = (
            beta * market_returns
            + 0.3 * sector_returns
            + np.random.normal(0.00005, idio_vol, n_days)
        )
        all_returns[:, j] = stock_returns

    # Convert to prices
    prices_matrix = 100 * np.exp(np.cumsum(all_returns, axis=0))
    prices = pd.DataFrame(prices_matrix, index=dates, columns=symbols)

    # Generate fundamentals
    fund_rows = []
    for sym in symbols:
        for year in range(2019, 2026):
            fund_rows.append({
                "symbol": sym,
                "filing_date": pd.Timestamp(f"{year}-03-15"),
                "report_date": pd.Timestamp(f"{year - 1}-12-31"),
                "period_type": "annual",
                "eps": np.random.uniform(2, 15),
                "book_value_per_share": np.random.uniform(10, 80),
                "net_income": np.random.uniform(1e8, 5e9),
                "total_equity": np.random.uniform(5e8, 50e9),
                "total_assets": np.random.uniform(1e9, 100e9),
                "total_liabilities": np.random.uniform(5e8, 60e9),
                "gross_profit": np.random.uniform(5e8, 20e9),
                "operating_income": np.random.uniform(2e8, 10e9),
                "operating_cash_flow": np.random.uniform(3e8, 15e9),
                "shares_outstanding": np.random.randint(1e8, 5e9),
            })

    fundamentals = pd.DataFrame(fund_rows)

    return prices, fundamentals


def run_single_factor_backtest(
    factor_name: str,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    start_date: date,
    end_date: date,
    do_walk_forward: bool = False,
) -> None:
    """Run a backtest for a single factor and generate tearsheet."""
    from src.quant.backtest.engine import BacktestConfig, run_backtest
    from src.quant.backtest.tearsheet import generate_tearsheet
    from src.quant.backtest.walk_forward import WalkForwardConfig, run_walk_forward
    from src.quant.factors.composite import CompositeFactor, get_all_factors

    # Compute factor scores
    print(f"Computing factor: {factor_name}...")

    if factor_name == "composite":
        all_factors = get_all_factors()
        factor_zscores = {}
        for f in all_factors:
            raw = f.compute_raw(prices, fundamentals)
            if not raw.empty:
                from src.quant.factors.base import cross_sectional_zscore
                z = raw.apply(cross_sectional_zscore, axis=1)
                factor_zscores[f.name] = z
                print(f"  {f.name}: computed ({len(z)} dates)")
            else:
                print(f"  {f.name}: no data (skipped)")

        composite = CompositeFactor()
        scores = composite.combine(factor_zscores)
    else:
        factor = _get_factor_by_name(factor_name)
        if factor is None:
            print(f"ERROR: Unknown factor '{factor_name}'")
            sys.exit(1)

        factor_data = factor.compute(prices, fundamentals)
        if factor_data.empty:
            print("ERROR: Factor computation returned no data")
            sys.exit(1)

        # Convert long format back to wide for backtest engine
        scores = factor_data.pivot(
            index="date", columns="symbol", values="z_score"
        )

    if scores.empty:
        print("ERROR: No factor scores computed")
        sys.exit(1)

    print(f"Factor scores: {len(scores)} dates × {len(scores.columns)} symbols")

    # Backtest config
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        num_holdings=50,
        max_position_weight=0.05,
    )

    # Generate benchmark (equal-weight of all stocks)
    bench_returns = prices.pct_change().mean(axis=1)
    bench_prices = (1 + bench_returns).cumprod() * 100

    # Run backtest
    print("Running backtest...")
    result = run_backtest(scores, prices, config, bench_prices)

    # Print key metrics
    m = result.metrics
    print(f"\n{'='*50}")
    print(f"  Results: {factor_name}")
    print(f"{'='*50}")
    print(f"  Ann. Return:    {m.get('ann_return', 0):>8.1%}")
    print(f"  Ann. Volatility:{m.get('ann_volatility', 0):>8.1%}")
    print(f"  Sharpe Ratio:   {m.get('sharpe_ratio', 0):>8.2f}")
    print(f"  Max Drawdown:   {m.get('max_drawdown', 0):>8.1%}")
    print(f"  Info Ratio:     {m.get('information_ratio', 0):>8.2f}")
    print(f"  Alpha t-stat:   {m.get('alpha_t_stat', 0):>8.2f}")
    print(f"{'='*50}\n")

    # Walk-forward validation
    wf_result = None
    if do_walk_forward:
        print("Running walk-forward validation (5 folds)...")
        wf_config = WalkForwardConfig(
            n_folds=5,
            in_sample_months=36,
            out_of_sample_months=12,
            purge_gap_days=5,
        )
        wf_result = run_walk_forward(
            scores, prices, wf_config, config, bench_prices
        )

        print("\nWalk-Forward Results:")
        for fm in wf_result.per_fold_metrics:
            print(
                f"  Fold {fm['fold']}: IS Sharpe={fm['is_sharpe']:.2f}, "
                f"OOS Sharpe={fm['oos_sharpe']:.2f}"
            )

        oos = wf_result.oos_combined_metrics
        print(f"\n  Combined OOS Sharpe: {oos.get('sharpe_ratio', 'N/A')}")
        print(f"  Combined OOS Return: {oos.get('ann_return', 0):.1%}")

    # Generate tearsheet
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"tearsheet_{factor_name}.html"

    generate_tearsheet(
        result=result,
        strategy_name=f"{factor_name.replace('_', ' ').title()} Factor Strategy",
        wf_result=wf_result,
        output_path=output_path,
    )

    print(f"\nTearsheet saved: {output_path}")


def _get_factor_by_name(name: str):  # type: ignore[return]
    """Get a factor instance by name."""
    from src.quant.factors.composite import get_all_factors

    for f in get_all_factors():
        if f.name == name:
            return f
    return None


def list_factors() -> None:
    """Print available factors."""
    from src.quant.factors.composite import get_all_factors

    print("\nAvailable factors:")
    print(f"  {'Name':<25} {'Category':<12}")
    print(f"  {'-'*25} {'-'*12}")

    for f in get_all_factors():
        print(f"  {f.name:<25} {f.category:<12}")

    print(f"  {'composite':<25} {'composite':<12}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FinSight Factor Backtest CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--factor", type=str, default="momentum_12_1",
                        help="Factor name to backtest (default: momentum_12_1)")
    parser.add_argument("--start", type=str, default="2021-01-01",
                        help="Backtest start date (default: 2021-01-01)")
    parser.add_argument("--end", type=str, default="2025-12-31",
                        help="Backtest end date (default: 2025-12-31)")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward validation")
    parser.add_argument("--list-factors", action="store_true",
                        help="List available factors")
    parser.add_argument("--use-db", action="store_true",
                        help="Use database data (requires running DB)")

    args = parser.parse_args()

    setup_logging()

    if args.list_factors:
        list_factors()
        return

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if args.use_db:
        print("Loading data from database...")
        from src.data.database import sync_engine
        from src.data.db_utils import execute_query
        from src.quant.factors.base import prepare_price_matrix

        market_df = execute_query(sync_engine, """
            SELECT symbol, date, adj_close FROM market_data
            WHERE date >= :start AND date <= :end
            ORDER BY date, symbol
        """, {"start": str(start), "end": str(end)})

        prices = prepare_price_matrix(market_df)

        fund_df = execute_query(sync_engine, """
            SELECT * FROM fundamentals ORDER BY filing_date
        """)
        fundamentals = fund_df if not fund_df.empty else None
    else:
        print("Using synthetic data (pass --use-db for real data)...")
        prices, fundamentals = generate_sample_data()

    run_single_factor_backtest(
        factor_name=args.factor,
        prices=prices,
        fundamentals=fundamentals,
        start_date=start,
        end_date=end,
        do_walk_forward=args.walk_forward,
    )


if __name__ == "__main__":
    main()
