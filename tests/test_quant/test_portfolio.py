"""Tests for portfolio construction and risk modules."""

import numpy as np
import pandas as pd


class TestPortfolioOptimizer:
    """Test portfolio optimization methods."""

    def test_equal_weight_basic(self) -> None:
        """Equal weight should select top N and assign 1/N weights."""
        from src.quant.portfolio.optimizer import equal_weight

        scores = pd.Series(
            {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0}
        )

        weights = equal_weight(scores, n_holdings=3)

        assert len(weights) == 3
        assert abs(weights.sum() - 1.0) < 1e-10
        assert "A" in weights.index
        assert "B" in weights.index
        assert "C" in weights.index

    def test_equal_weight_max_constraint(self) -> None:
        """No position should exceed max_weight after normalization."""
        from src.quant.portfolio.optimizer import equal_weight

        scores = pd.Series({f"S{i}": float(i) for i in range(100)})
        weights = equal_weight(scores, n_holdings=50, max_weight=0.05)

        assert weights.max() <= 0.051  # Allow small float error
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_risk_parity_basic(self) -> None:
        """Risk parity should weight inversely to volatility."""
        from src.quant.portfolio.optimizer import risk_parity

        scores = pd.Series({"A": 2.0, "B": 1.0, "C": 0.5})
        vols = pd.Series({"A": 0.30, "B": 0.15, "C": 0.10})

        weights = risk_parity(scores, vols, n_holdings=3, max_weight=1.0)

        assert abs(weights.sum() - 1.0) < 1e-10
        # Lower vol should get higher weight
        assert weights["C"] > weights["A"]

    def test_risk_parity_fallback(self) -> None:
        """Should fall back to equal weight if vol data is missing."""
        from src.quant.portfolio.optimizer import risk_parity

        scores = pd.Series({"A": 2.0, "B": 1.0})
        vols = pd.Series(dtype=float)

        weights = risk_parity(scores, vols, n_holdings=2)

        assert len(weights) == 2
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_sector_constraint(self) -> None:
        """Sector constraint should limit weight per sector."""
        from src.quant.portfolio.optimizer import apply_sector_constraint

        weights = pd.Series({
            "A": 0.25, "B": 0.25,  # Tech: 0.50
            "C": 0.15, "D": 0.10,  # Health: 0.25
            "E": 0.15, "F": 0.10,  # Finance: 0.25
        })
        sector_map = {
            "A": "Tech", "B": "Tech",
            "C": "Health", "D": "Health",
            "E": "Finance", "F": "Finance",
        }

        adjusted = apply_sector_constraint(weights, sector_map, max_sector_weight=0.35)

        # Tech should be capped at ~0.35
        tech_weight = adjusted[["A", "B"]].sum()

        assert tech_weight <= 0.36
        assert abs(adjusted.sum() - 1.0) < 0.01

    def test_turnover_constraint(self) -> None:
        """Turnover constraint should blend weights if turnover exceeds limit."""
        from src.quant.portfolio.optimizer import apply_turnover_constraint

        current = pd.Series({"A": 0.5, "B": 0.5})
        target = pd.Series({"C": 0.5, "D": 0.5})  # 100% turnover

        blended = apply_turnover_constraint(target, current, max_turnover=0.30)

        # Should be a blend, not fully target
        assert abs(blended.sum() - 1.0) < 0.01
        # Should still hold some of A and B
        all_syms = set(blended[blended > 0.01].index)
        assert len(all_syms) > 2


class TestRiskMetrics:
    """Test portfolio risk metric computations."""

    def test_var_historical(self) -> None:
        """Historical VaR should be at expected percentile."""
        from src.quant.portfolio.risk import value_at_risk

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 1000))

        var_95 = value_at_risk(returns, confidence=0.95, method="historical")

        # 95% VaR for N(0, 0.01) should be ~0.0165
        assert 0.005 < var_95 < 0.03

    def test_var_parametric(self) -> None:
        """Parametric VaR should be based on normal distribution."""
        from src.quant.portfolio.risk import value_at_risk

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 1000))

        var = value_at_risk(returns, confidence=0.95, method="parametric")

        assert var > 0

    def test_cvar(self) -> None:
        """CVaR should be larger than VaR (more conservative)."""
        from src.quant.portfolio.risk import conditional_var, value_at_risk

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 1000))

        var = value_at_risk(returns, confidence=0.95)
        cvar = conditional_var(returns, confidence=0.95)

        assert cvar >= var

    def test_portfolio_beta(self) -> None:
        """Beta of market should be ~1.0."""
        from src.quant.portfolio.risk import portfolio_beta

        np.random.seed(42)
        n = 500
        market = pd.Series(np.random.normal(0.0003, 0.01, n))
        port = market * 1.2 + pd.Series(np.random.normal(0, 0.005, n))

        beta = portfolio_beta(port, market)

        # Should be roughly 1.2
        assert 0.8 < beta < 1.6
