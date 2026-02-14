"""Tests for paper trading script."""

from scripts.paper_trade import generate_trades, load_latest_snapshot


class TestGenerateTrades:
    def test_returns_list(self):
        snapshot = {
            "date": "2026-01-01",
            "long": {"AAPL": 1.5, "MSFT": 0.8},
            "short": {"TSLA": -1.2},
        }
        trades = generate_trades(snapshot)
        assert isinstance(trades, list)

    def test_trade_count_matches_long(self):
        snapshot = {
            "date": "2026-01-01",
            "long": {"AAPL": 1.5, "MSFT": 0.8, "GOOGL": 0.3},
            "short": {},
        }
        trades = generate_trades(snapshot)
        assert len(trades) == 3

    def test_trade_has_required_fields(self):
        snapshot = {
            "date": "2026-01-01",
            "long": {"AAPL": 1.5},
            "short": {},
        }
        trades = generate_trades(snapshot)
        assert len(trades) == 1
        t = trades[0]
        assert t["symbol"] == "AAPL"
        assert t["side"] == "BUY"
        assert t["shares"] > 0
        assert t["notional"] > 0

    def test_respects_max_position_size(self):
        snapshot = {
            "date": "2026-01-01",
            "long": {"AAPL": 1.5},
            "short": {},
        }
        trades = generate_trades(snapshot, capital=100_000, max_position_pct=5.0)
        t = trades[0]
        # Max position = 5000, at $150/share = ~33 shares
        assert t["notional"] <= 5500  # Some rounding tolerance

    def test_empty_long(self):
        snapshot = {"date": "2026-01-01", "long": {}, "short": {}}
        trades = generate_trades(snapshot)
        assert trades == []


class TestLoadLatestSnapshot:
    def test_returns_none_when_no_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = load_latest_snapshot()
        assert result is None
