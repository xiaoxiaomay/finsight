"""Tests for Regime Detector."""

from datetime import date

import numpy as np
import pandas as pd


class TestRegimeDetector:
    """Test rule-based regime detection."""

    def _make_sp500_series(self, trend: str = "up", n: int = 250) -> pd.Series:
        """Generate synthetic SP500 price series."""
        rng = np.random.default_rng(42)
        if trend == "up":
            base = np.linspace(4000, 5000, n) + rng.normal(0, 20, n)
        elif trend == "down":
            base = np.linspace(5000, 4000, n) + rng.normal(0, 20, n)
        elif trend == "recovery":
            # First go down, then cross above
            base = np.concatenate([
                np.linspace(5000, 4200, n - 20),
                np.linspace(4200, 4800, 20),
            ]) + rng.normal(0, 10, n)
        else:
            base = np.full(n, 4500) + rng.normal(0, 20, n)
        return pd.Series(base)

    def test_bull_regime(self) -> None:
        from src.agents.regime_detector import RegimeDetector

        detector = RegimeDetector()
        sp500 = self._make_sp500_series("up")

        result = detector.detect(
            sp500_prices=sp500,
            vix_level=15.0,
            yield_curve_spread=1.0,
            assessment_date=date(2024, 6, 15),
        )

        assert result.regime == "bull"
        assert result.defensive_tilt is False
        assert result.suggested_equity_weight > 0.6

    def test_bear_regime(self) -> None:
        from src.agents.regime_detector import RegimeDetector

        detector = RegimeDetector()
        sp500 = self._make_sp500_series("down")

        result = detector.detect(
            sp500_prices=sp500,
            vix_level=28.0,
            yield_curve_spread=-0.5,
            assessment_date=date(2024, 6, 15),
        )

        assert result.regime == "bear"
        assert result.defensive_tilt is True
        assert result.suggested_equity_weight < 0.5

    def test_high_vol_regime(self) -> None:
        from src.agents.regime_detector import RegimeDetector

        detector = RegimeDetector()
        sp500 = self._make_sp500_series("up")

        result = detector.detect(
            sp500_prices=sp500,
            vix_level=35.0,
            yield_curve_spread=0.5,
        )

        assert result.regime == "high_vol"
        assert result.defensive_tilt is True

    def test_factor_tilt_bull(self) -> None:
        from src.agents.regime_detector import RegimeDetector

        detector = RegimeDetector()
        sp500 = self._make_sp500_series("up")

        result = detector.detect(
            sp500_prices=sp500,
            vix_level=15.0,
            yield_curve_spread=1.0,
        )

        assert "momentum_12_1" in result.recommended_factors
        assert result.factor_tilt != ""

    def test_factor_tilt_bear(self) -> None:
        from src.agents.regime_detector import RegimeDetector

        detector = RegimeDetector()
        sp500 = self._make_sp500_series("down")

        result = detector.detect(
            sp500_prices=sp500,
            vix_level=28.0,
            yield_curve_spread=-0.5,
        )

        assert "volatility_60d" in result.recommended_factors
        assert "momentum_12_1" in result.factors_to_avoid

    def test_reasoning_included(self) -> None:
        from src.agents.regime_detector import RegimeDetector

        detector = RegimeDetector()
        sp500 = self._make_sp500_series("up")

        result = detector.detect(
            sp500_prices=sp500,
            vix_level=15.0,
            yield_curve_spread=1.0,
        )

        assert len(result.reasoning) > 0
        assert "200-day" in result.reasoning

    def test_no_sp500_data(self) -> None:
        from src.agents.regime_detector import RegimeDetector

        detector = RegimeDetector()
        result = detector.detect(
            sp500_prices=None,
            vix_level=18.0,
            yield_curve_spread=0.5,
        )

        assert result.regime in ("bull", "bear", "high_vol", "recovery", "neutral")

    def test_detect_from_data_convenience(self) -> None:
        from src.agents.regime_detector import RegimeDetector

        data = list(np.linspace(4000, 5000, 250))
        result = RegimeDetector.detect_from_data(
            sp500_data=data,
            vix=15.0,
            yield_spread=1.0,
        )

        assert result.regime == "bull"
        assert result.vix_level == 15.0

    def test_confidence_higher_with_strong_signals(self) -> None:
        from src.agents.regime_detector import RegimeDetector

        detector = RegimeDetector()
        sp500 = self._make_sp500_series("up")

        # Strong bull: low VIX, high above DMA, normal curve
        result = detector.detect(
            sp500_prices=sp500,
            vix_level=12.0,
            yield_curve_spread=1.5,
        )

        assert result.confidence >= 0.6
