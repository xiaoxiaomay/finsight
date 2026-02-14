"""Seed script: load historical data into the database.

Usage:
    python scripts/seed_data.py                     # Full SP500 load
    python scripts/seed_data.py --symbols AAPL MSFT # Specific symbols only
    python scripts/seed_data.py --universe TSX60    # TSX60 only
    python scripts/seed_data.py --skip-fundamentals # Skip FMP calls (save quota)
    python scripts/seed_data.py --quick             # 10 symbols, for testing

This script performs a one-time historical data load:
1. Market data (yfinance) — 2020-01-01 to today
2. Fundamental data (FMP API) — annual financials
3. Macro data (FRED) — all configured indicators
4. Quality checks — validates everything loaded correctly
"""

import argparse
import asyncio
import sys
import time
from datetime import date
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import get_logger, setup_logging
from src.config.settings import get_settings

logger = get_logger("seed_data")

# Quick test subset
QUICK_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "JNJ", "V",
]


async def seed_market_data(symbols: list[str], start_date: date) -> dict:
    """Load historical market data from yfinance."""
    from src.data.ingestion.market_data import MarketDataIngestor

    logger.info("seeding_market_data", symbols=len(symbols), start=str(start_date))

    ingestor = MarketDataIngestor()
    end_date = date.today()

    result = await ingestor.ingest(symbols, start_date, end_date)
    return result


async def seed_fundamentals(symbols: list[str], start_date: date) -> dict:
    """Load fundamental data from FMP API."""
    from src.data.ingestion.fundamentals import FundamentalsIngestor

    # Filter US symbols only (FMP coverage)
    us_symbols = [s for s in symbols if not s.endswith(".TO")]

    # FMP free tier: 250 calls/day, 3 per symbol = ~83 symbols
    # Be conservative: process up to 80 symbols
    batch = us_symbols[:80]
    logger.info(
        "seeding_fundamentals",
        total_us=len(us_symbols),
        batch_size=len(batch),
        note="FMP free tier limits to ~80 symbols/day",
    )

    ingestor = FundamentalsIngestor()
    result = await ingestor.ingest(batch, start_date, date.today())
    return result


async def seed_macro(start_date: date) -> dict:
    """Load macro indicators from FRED."""
    from src.data.ingestion.macro import MacroIngestor

    logger.info("seeding_macro_data")

    ingestor = MacroIngestor()
    result = await ingestor.ingest([], start_date, date.today())
    return result


def run_quality_checks() -> list[dict]:
    """Run post-seed quality checks."""
    from src.data.database import sync_engine
    from src.data.quality.monitors import DataQualityMonitor

    logger.info("running_quality_checks")
    monitor = DataQualityMonitor(sync_engine)
    return monitor.run_all_checks()


def print_db_summary() -> None:
    """Print summary of data loaded into the database."""
    from src.data.database import sync_engine
    from src.data.db_utils import get_row_count

    tables = [
        "market_data",
        "fundamentals",
        "macro_indicators",
        "data_quality_log",
    ]

    print("\n" + "=" * 60)
    print("  Database Summary")
    print("=" * 60)

    for table in tables:
        try:
            count = get_row_count(sync_engine, table)
            print(f"  {table:<25} {count:>10,} rows")
        except Exception:
            print(f"  {table:<25} {'ERROR':>10}")

    print("=" * 60 + "\n")


async def main(args: argparse.Namespace) -> None:
    """Main seed workflow."""
    setup_logging()
    settings = get_settings()

    start_date = settings.market_data_start_date
    start_time = time.time()

    # Determine symbols
    if args.symbols:
        symbols = args.symbols
        logger.info("using_custom_symbols", symbols=symbols)
    elif args.quick:
        symbols = QUICK_SYMBOLS
        logger.info("using_quick_test_set", symbols=len(symbols))
    else:
        from src.data.ingestion.universes import get_universe
        symbols = get_universe(args.universe)
        logger.info("using_universe", universe=args.universe, symbols=len(symbols))

    print("\nFinSight Data Seeding")
    print(f"  Universe: {args.universe} ({len(symbols)} symbols)")
    print(f"  Start date: {start_date}")
    print(f"  Skip fundamentals: {args.skip_fundamentals}")
    print()

    # Step 1: Market Data
    print("[1/4] Loading market data (yfinance)...")
    try:
        market_result = await seed_market_data(symbols, start_date)
        print(f"      Fetched {market_result['rows_fetched']:,} rows, "
              f"stored {market_result['rows_stored']:,}")
        if market_result["issues"]:
            print(f"      Issues: {len(market_result['issues'])}")
    except Exception as e:
        print(f"      ERROR: {e}")
        logger.error("market_data_seed_failed", error=str(e))

    # Step 2: Macro Data
    print("[2/4] Loading macro indicators (FRED)...")
    try:
        macro_result = await seed_macro(start_date)
        print(f"      Fetched {macro_result['rows_fetched']:,} rows, "
              f"stored {macro_result['rows_stored']:,}")
    except Exception as e:
        print(f"      ERROR: {e}")
        logger.error("macro_seed_failed", error=str(e))

    # Step 3: Fundamentals (optional)
    if not args.skip_fundamentals:
        print("[3/4] Loading fundamentals (FMP)...")
        try:
            fund_result = await seed_fundamentals(symbols, start_date)
            print(f"      Fetched {fund_result['rows_fetched']:,} rows, "
                  f"stored {fund_result['rows_stored']:,}")
            if fund_result["issues"]:
                print(f"      Issues: {len(fund_result['issues'])}")
        except Exception as e:
            print(f"      ERROR: {e}")
            logger.error("fundamentals_seed_failed", error=str(e))
    else:
        print("[3/4] Skipping fundamentals (--skip-fundamentals)")

    # Step 4: Quality Checks
    print("[4/4] Running quality checks...")
    try:
        results = run_quality_checks()
        passed = sum(1 for r in results if r["status"] == "pass")
        warned = sum(1 for r in results if r["status"] == "warn")
        failed = sum(1 for r in results if r["status"] == "fail")
        print(f"      Results: {passed} passed, {warned} warnings, {failed} failed")
    except Exception as e:
        print(f"      ERROR: {e}")
        logger.error("quality_checks_failed", error=str(e))

    # Summary
    elapsed = time.time() - start_time
    print_db_summary()
    print(f"Completed in {elapsed:.1f} seconds.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FinSight: Load historical financial data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/seed_data.py                     # Full SP500 load
  python scripts/seed_data.py --quick             # 10 symbols (fast test)
  python scripts/seed_data.py --symbols AAPL MSFT # Specific symbols
  python scripts/seed_data.py --skip-fundamentals # Save FMP API quota
        """,
    )

    parser.add_argument(
        "--symbols", nargs="+",
        help="Specific symbols to load (overrides universe)",
    )
    parser.add_argument(
        "--universe", default="SP500",
        choices=["SP500", "TSX60", "ALL"],
        help="Stock universe to load (default: SP500)",
    )
    parser.add_argument(
        "--skip-fundamentals", action="store_true",
        help="Skip FMP fundamental data (saves daily API quota)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test: load only 10 symbols",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
