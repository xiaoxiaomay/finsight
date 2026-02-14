"""Market Regime Detector.

Rule-based regime detection using macro indicators:
- Bull: SP500 > 200DMA AND VIX < 20 AND yield curve > 0
- Bear: SP500 < 200DMA AND VIX > 25
- High Vol: VIX > 30
- Recovery: SP500 crossing above 200DMA
- Neutral: none of the above

Provides factor tilt recommendations per regime.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.agents.schemas.regime import RegimeOutput
from src.config.logging_config import get_logger

logger = get_logger("regime_detector")

# Factor tilt recommendations by regime
REGIME_TILTS: dict[str, dict] = {
    "bull": {
        "factor_tilt": "momentum + growth",
        "recommended_factors": ["momentum_12_1", "earnings_yield", "roe", "gross_profitability"],
        "factors_to_avoid": ["volatility_60d", "short_term_reversal"],
        "suggested_equity_weight": 0.75,
        "defensive_tilt": False,
    },
    "bear": {
        "factor_tilt": "quality + low_vol",
        "recommended_factors": ["volatility_60d", "gross_profitability", "accruals", "roe"],
        "factors_to_avoid": ["momentum_12_1", "earnings_yield"],
        "suggested_equity_weight": 0.35,
        "defensive_tilt": True,
    },
    "high_vol": {
        "factor_tilt": "low_vol + quality",
        "recommended_factors": ["volatility_60d", "accruals", "gross_profitability"],
        "factors_to_avoid": ["momentum_12_1", "short_term_reversal", "book_to_market"],
        "suggested_equity_weight": 0.30,
        "defensive_tilt": True,
    },
    "recovery": {
        "factor_tilt": "value + reversal",
        "recommended_factors": ["book_to_market", "earnings_yield", "short_term_reversal", "ev_ebitda"],
        "factors_to_avoid": ["volatility_60d"],
        "suggested_equity_weight": 0.65,
        "defensive_tilt": False,
    },
    "neutral": {
        "factor_tilt": "balanced",
        "recommended_factors": ["roe", "gross_profitability", "earnings_yield", "momentum_12_1"],
        "factors_to_avoid": [],
        "suggested_equity_weight": 0.60,
        "defensive_tilt": False,
    },
}


class RegimeDetector:
    """Rule-based market regime detector.

    Uses SP500 price vs 200DMA, VIX level, and yield curve spread
    to classify the current market regime.
    """

    def __init__(
        self,
        sp500_lookback: int = 200,
        vix_bull_threshold: float = 20.0,
        vix_bear_threshold: float = 25.0,
        vix_highvol_threshold: float = 30.0,
    ) -> None:
        self.sp500_lookback = sp500_lookback
        self.vix_bull_threshold = vix_bull_threshold
        self.vix_bear_threshold = vix_bear_threshold
        self.vix_highvol_threshold = vix_highvol_threshold

    def detect(
        self,
        sp500_prices: pd.Series | None = None,
        vix_level: float | None = None,
        yield_curve_spread: float | None = None,
        assessment_date: date | None = None,
    ) -> RegimeOutput:
        """Detect current market regime.

        Args:
            sp500_prices: Series of SP500 close prices (DatetimeIndex).
            vix_level: Current VIX level.
            yield_curve_spread: 10Y-2Y Treasury spread in percentage points.
            assessment_date: Date of assessment.

        Returns:
            RegimeOutput with regime label and factor recommendations.
        """
        if assessment_date is None:
            assessment_date = date.today()

        # Compute SP500 vs 200DMA
        sp500_vs_200dma = 0.0
        sp500_trend = "flat"
        sp500_above_dma = None

        if sp500_prices is not None and len(sp500_prices) >= self.sp500_lookback:
            dma200 = sp500_prices.rolling(window=self.sp500_lookback).mean()
            current_price = float(sp500_prices.iloc[-1])
            current_dma = float(dma200.iloc[-1])

            if current_dma > 0:
                sp500_vs_200dma = ((current_price / current_dma) - 1) * 100
                sp500_above_dma = current_price > current_dma

            # Check trend: is SP500 recently crossing the 200DMA?
            if len(dma200.dropna()) >= 5:
                prev_above = float(sp500_prices.iloc[-5]) > float(dma200.iloc[-5])
                curr_above = sp500_above_dma
                if curr_above and not prev_above:
                    sp500_trend = "crossing_up"
                elif not curr_above and prev_above:
                    sp500_trend = "crossing_down"
                elif sp500_vs_200dma > 3:
                    sp500_trend = "up"
                elif sp500_vs_200dma < -3:
                    sp500_trend = "down"

        # Default values
        if vix_level is None:
            vix_level = 18.0
        if yield_curve_spread is None:
            yield_curve_spread = 0.5

        # Classify regime
        regime = self._classify(
            sp500_above_dma=sp500_above_dma,
            sp500_trend=sp500_trend,
            vix=vix_level,
            yield_spread=yield_curve_spread,
        )

        # Get confidence
        confidence = self._compute_confidence(
            regime, sp500_vs_200dma, vix_level, yield_curve_spread,
        )

        # Get factor tilts
        tilts = REGIME_TILTS.get(regime, REGIME_TILTS["neutral"])

        reasoning = self._build_reasoning(
            regime, sp500_vs_200dma, vix_level, yield_curve_spread, sp500_trend,
        )

        return RegimeOutput(
            assessment_date=assessment_date,
            regime=regime,
            confidence=confidence,
            sp500_vs_200dma=round(sp500_vs_200dma, 2),
            vix_level=vix_level,
            yield_curve_spread=yield_curve_spread,
            sp500_trend=sp500_trend if sp500_trend != "crossing_up" else "up",
            factor_tilt=tilts["factor_tilt"],
            recommended_factors=tilts["recommended_factors"],
            factors_to_avoid=tilts["factors_to_avoid"],
            suggested_equity_weight=tilts["suggested_equity_weight"],
            defensive_tilt=tilts["defensive_tilt"],
            reasoning=reasoning,
        )

    def _classify(
        self,
        sp500_above_dma: bool | None,
        sp500_trend: str,
        vix: float,
        yield_spread: float,
    ) -> str:
        """Classify market regime from indicators."""
        # High volatility takes priority
        if vix > self.vix_highvol_threshold:
            return "high_vol"

        # Recovery: SP500 crossing above 200DMA
        if sp500_trend == "crossing_up":
            return "recovery"

        # Bear market
        if sp500_above_dma is False and vix > self.vix_bear_threshold:
            return "bear"

        # Bull market
        if sp500_above_dma is True and vix < self.vix_bull_threshold and yield_spread > 0:
            return "bull"

        # Mild bear (below DMA but low vol)
        if sp500_above_dma is False:
            return "bear"

        # Mild bull (above DMA but elevated vol or inverted curve)
        if sp500_above_dma is True:
            return "bull"

        return "neutral"

    @staticmethod
    def _compute_confidence(
        regime: str,
        sp500_vs_dma: float,
        vix: float,
        yield_spread: float,
    ) -> float:
        """Compute confidence in regime classification."""
        # Base confidence
        conf = 0.5

        if regime == "bull":
            if sp500_vs_dma > 5:
                conf += 0.15
            if vix < 15:
                conf += 0.1
            if yield_spread > 0.5:
                conf += 0.1
        elif regime == "bear":
            if sp500_vs_dma < -5:
                conf += 0.15
            if vix > 30:
                conf += 0.1
            if yield_spread < -0.5:
                conf += 0.1
        elif regime == "high_vol":
            if vix > 35:
                conf += 0.2
        elif regime == "recovery":
            conf += 0.1

        return min(conf, 0.95)

    @staticmethod
    def _build_reasoning(
        regime: str,
        sp500_vs_dma: float,
        vix: float,
        yield_spread: float,
        trend: str,
    ) -> str:
        """Build human-readable reasoning for the regime classification."""
        parts = [f"Market classified as '{regime}'."]

        if sp500_vs_dma > 0:
            parts.append(f"S&P 500 is {sp500_vs_dma:.1f}% above its 200-day moving average.")
        else:
            parts.append(f"S&P 500 is {abs(sp500_vs_dma):.1f}% below its 200-day moving average.")

        parts.append(f"VIX at {vix:.1f}.")

        if yield_spread > 0:
            parts.append(f"Yield curve normal (10Y-2Y spread: {yield_spread:.2f}%).")
        elif yield_spread < 0:
            parts.append(f"Yield curve inverted (10Y-2Y spread: {yield_spread:.2f}%).")
        else:
            parts.append("Yield curve flat.")

        if trend == "crossing_up":
            parts.append("S&P 500 recently crossed above 200DMA, suggesting recovery.")

        return " ".join(parts)

    @staticmethod
    def detect_from_data(
        sp500_data: np.ndarray | list[float],
        vix: float,
        yield_spread: float,
        assessment_date: date | None = None,
    ) -> RegimeOutput:
        """Convenience method: detect regime from raw arrays."""
        detector = RegimeDetector()
        sp500_series = pd.Series(sp500_data) if sp500_data is not None else None
        return detector.detect(
            sp500_prices=sp500_series,
            vix_level=vix,
            yield_curve_spread=yield_spread,
            assessment_date=assessment_date,
        )
