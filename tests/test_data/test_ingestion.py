"""Tests for data ingestion modules."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest


class TestMarketDataIngestor:
    """Test yfinance market data ingestion."""

    def test_reshape_single_ticker(self) -> None:
        """Test reshaping single-ticker yfinance output."""
        from src.data.ingestion.market_data import _reshape_single

        raw = pd.DataFrame({
            "Date": pd.date_range("2024-01-02", periods=3),
            "Open": [185.0, 186.0, 187.0],
            "High": [187.0, 188.0, 189.0],
            "Low": [184.0, 185.0, 186.0],
            "Close": [186.0, 187.0, 188.0],
            "Adj Close": [186.0, 187.0, 188.0],
            "Volume": [50_000_000, 48_000_000, 52_000_000],
        }).set_index("Date")

        result = _reshape_single(raw, "AAPL")

        assert len(result) == 3
        assert list(result.columns) == [
            "symbol", "date", "open", "high", "low",
            "close", "adj_close", "volume", "source",
        ]
        assert (result["symbol"] == "AAPL").all()
        assert (result["source"] == "yfinance").all()

    def test_validate_removes_all_null_prices(self) -> None:
        """Validation should remove rows with all-null prices."""
        from src.data.ingestion.market_data import MarketDataIngestor

        ingestor = MarketDataIngestor()

        df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "date": ["2024-01-02", "2024-01-03"],
            "open": [185.0, None],
            "high": [187.0, None],
            "low": [184.0, None],
            "close": [186.0, None],
            "adj_close": [186.0, None],
            "volume": [50_000_000, None],
        })

        clean, issues = ingestor.validate(df)
        assert len(clean) == 1
        assert any("all-null" in i for i in issues)

    def test_validate_flags_extreme_moves(self) -> None:
        """Validation should flag >50% intraday moves."""
        from src.data.ingestion.market_data import MarketDataIngestor

        ingestor = MarketDataIngestor()

        df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "date": ["2024-01-02", "2024-01-03"],
            "open": [100.0, 100.0],
            "high": [200.0, 105.0],
            "low": [99.0, 99.0],
            "close": [180.0, 103.0],  # 80% move on first day
            "adj_close": [180.0, 103.0],
            "volume": [50_000_000, 48_000_000],
        })

        clean, issues = ingestor.validate(df)
        assert any("50%" in i for i in issues)


class TestFundamentalsIngestor:
    """Test FMP fundamental data ingestion."""

    def test_parse_income_statement(self) -> None:
        """Test parsing FMP income statement response."""
        from src.data.ingestion.fundamentals import FundamentalsIngestor

        ingestor = FundamentalsIngestor()

        data = [
            {
                "date": "2024-09-30",
                "fillingDate": "2024-11-01",
                "acceptedDate": "2024-10-31",
                "period": "FY",
                "revenue": 391_035_000_000,
                "costOfRevenue": 210_000_000_000,
                "grossProfit": 181_035_000_000,
                "operatingIncome": 123_000_000_000,
                "netIncome": 93_736_000_000,
                "eps": 6.13,
                "weightedAverageShsOut": 15_287_000_000,
            },
        ]

        df = ingestor._parse_income(data, "AAPL")

        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "AAPL"
        assert df.iloc[0]["filing_date"] == "2024-11-01"
        assert df.iloc[0]["period_type"] == "annual"
        assert df.iloc[0]["revenue"] == 391_035_000_000

    def test_parse_balance_sheet(self) -> None:
        """Test parsing FMP balance sheet response."""
        from src.data.ingestion.fundamentals import FundamentalsIngestor

        ingestor = FundamentalsIngestor()

        data = [
            {
                "date": "2024-09-30",
                "period": "FY",
                "totalAssets": 352_583_000_000,
                "totalLiabilities": 290_437_000_000,
                "totalStockholdersEquity": 62_146_000_000,
                "commonStock": 15_000_000_000,
            },
        ]

        df = ingestor._parse_balance(data, "AAPL")

        assert len(df) == 1
        assert df.iloc[0]["total_assets"] == 352_583_000_000
        assert df.iloc[0]["total_equity"] == 62_146_000_000

    def test_validate_point_in_time(self) -> None:
        """Validation should flag filing_date < report_date."""
        from src.data.ingestion.fundamentals import FundamentalsIngestor

        ingestor = FundamentalsIngestor()

        df = pd.DataFrame({
            "symbol": ["AAPL"],
            "report_date": ["2024-09-30"],
            "filing_date": ["2024-08-01"],  # Before report date — bad
            "period_type": ["annual"],
            "revenue": [391_000_000_000],
        })

        clean, issues = ingestor.validate(df)
        assert any("filing_date" in i for i in issues)


class TestMacroIngestor:
    """Test FRED macro data ingestion."""

    @pytest.mark.asyncio
    async def test_parse_fred_response(self) -> None:
        """Test parsing FRED API observations."""
        from src.data.ingestion.macro import MacroIngestor

        ingestor = MacroIngestor()

        # Mock a single series fetch
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "observations": [
                {"date": "2024-01-01", "value": "5.33"},
                {"date": "2024-02-01", "value": "5.33"},
                {"date": "2024-03-01", "value": "."},  # Missing value
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        df = await ingestor._fetch_series(
            mock_client, "DFF", date(2024, 1, 1), date(2024, 12, 31)
        )

        assert df is not None
        assert len(df) == 2  # "." value should be skipped
        assert df.iloc[0]["indicator_id"] == "DFF"
        assert df.iloc[0]["value"] == 5.33

    def test_validate_removes_duplicates(self) -> None:
        """Validation should remove duplicate (indicator_id, date) pairs."""
        from src.data.ingestion.macro import MacroIngestor

        ingestor = MacroIngestor()

        df = pd.DataFrame({
            "indicator_id": ["DFF", "DFF", "DGS10"],
            "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "value": [5.33, 5.34, 4.50],
            "source": ["FRED", "FRED", "FRED"],
        })

        clean, issues = ingestor.validate(df)
        assert len(clean) == 2  # One duplicate removed
        assert any("duplicate" in i.lower() for i in issues)


class TestUniverses:
    """Test symbol universe functions."""

    def test_tsx60_list(self) -> None:
        """TSX60 list should have reasonable number of symbols."""
        from src.data.ingestion.universes import get_tsx60_symbols

        symbols = get_tsx60_symbols()
        assert len(symbols) >= 40
        assert all(s.endswith(".TO") for s in symbols)

    def test_sp500_fallback(self) -> None:
        """SP500 fallback list should have ~500 symbols."""
        from src.data.ingestion.universes import _SP500_FALLBACK

        assert len(_SP500_FALLBACK) >= 450
        assert "AAPL" in _SP500_FALLBACK
        assert "MSFT" in _SP500_FALLBACK

    def test_get_universe_invalid(self) -> None:
        """Should raise ValueError for unknown universe."""
        from src.data.ingestion.universes import get_universe

        with pytest.raises(ValueError, match="Unknown universe"):
            get_universe("INVALID")
