"""Quant Signal Agent.

Queries factor scores for a given stock, computes composite signal,
and provides quantitative analysis with historical backtest context.
Produces QuantSignalOutput.
"""

from __future__ import annotations

import json
from datetime import date

import anthropic

from src.agents.prompts.quant_signal import (
    QUANT_SIGNAL_PROMPT,
    QUANT_SIGNAL_SYSTEM_PROMPT,
)
from src.agents.schemas.quant_signal import (
    BacktestStats,
    FactorScore,
    QuantSignalOutput,
)
from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger("agent.quant_signal")

# Default factor weights matching composite.py
DEFAULT_WEIGHTS: dict[str, float] = {
    "momentum_12_1": 0.15,
    "short_term_reversal": 0.05,
    "earnings_yield": 0.10,
    "book_to_market": 0.10,
    "ev_ebitda": 0.10,
    "roe": 0.10,
    "gross_profitability": 0.10,
    "accruals": 0.10,
    "asset_growth": 0.10,
    "volatility_60d": 0.10,
}


class QuantSignalAgent:
    """Agent for quantitative factor signal analysis.

    Uses:
    - Factor signals from DB or in-memory data
    - Composite scoring with configurable weights
    - Claude Sonnet for signal interpretation
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        settings = get_settings()
        self.client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self.weights = weights or DEFAULT_WEIGHTS

    def analyze(
        self,
        symbol: str,
        factor_data: dict[str, dict] | None = None,
        backtest_stats: dict | None = None,
        sector: str = "",
        analysis_date: date | None = None,
    ) -> QuantSignalOutput:
        """Run quant signal analysis for a stock.

        Args:
            symbol: Stock ticker.
            factor_data: Dict of factor_name -> {raw_value, z_score, percentile}.
                If None, returns minimal output.
            backtest_stats: Optional historical backtest performance dict.
            sector: GICS sector name.
            analysis_date: Date of analysis.

        Returns:
            QuantSignalOutput Pydantic model.
        """
        if analysis_date is None:
            analysis_date = date.today()

        if not factor_data:
            logger.warning("no_factor_data", symbol=symbol)
            return self._empty_result(symbol, analysis_date)

        # Step 1: Build factor scores
        factor_scores = self._build_factor_scores(factor_data)

        # Step 2: Compute composite
        composite_score, composite_pctile = self._compute_composite(factor_data)

        # Step 3: Determine signal from composite
        signal = self._determine_signal(composite_pctile)

        # Step 4: Identify strongest/weakest
        sorted_factors = sorted(
            factor_scores, key=lambda f: f.z_score, reverse=True,
        )
        strongest = [f.factor_name for f in sorted_factors[:3]]
        weakest = [f.factor_name for f in sorted_factors[-3:]]

        # Step 5: Build backtest context
        bt_stats = None
        bt_context = "No backtest data available."
        if backtest_stats:
            bt_stats = BacktestStats(**backtest_stats)
            bt_context = self._format_backtest(backtest_stats)

        # Step 6: Call LLM for interpretation
        factor_text = self._format_factors(factor_scores)
        prompt = QUANT_SIGNAL_PROMPT.format(
            symbol=symbol,
            analysis_date=str(analysis_date),
            factor_data=factor_text,
            composite_score=f"{composite_score:.3f}",
            composite_percentile=f"{composite_pctile:.1f}",
            backtest_context=bt_context,
            sector=sector,
        )

        try:
            response_text = self._call_llm(prompt)
            parsed = self._parse_response(response_text)
            confidence = parsed.get("confidence_level", "medium")
        except Exception:
            logger.warning("llm_interpretation_failed", symbol=symbol)
            confidence = "medium"

        return QuantSignalOutput(
            symbol=symbol,
            analysis_date=analysis_date,
            factor_scores=factor_scores,
            composite_score=composite_score,
            composite_percentile=composite_pctile,
            signal_direction=signal["direction"],
            signal_strength=signal["strength"],
            backtest_stats=bt_stats,
            strongest_factors=strongest,
            weakest_factors=weakest,
            sector=sector,
            confidence_level=confidence,
        )

    def _build_factor_scores(
        self, factor_data: dict[str, dict],
    ) -> list[FactorScore]:
        """Convert factor data dict to FactorScore list."""
        scores = []
        for name, vals in factor_data.items():
            scores.append(FactorScore(
                factor_name=name,
                raw_value=vals.get("raw_value", 0.0),
                z_score=vals.get("z_score", 0.0),
                percentile=vals.get("percentile", 50.0),
            ))
        return scores

    def _compute_composite(
        self, factor_data: dict[str, dict],
    ) -> tuple[float, float]:
        """Compute weighted composite z-score and percentile."""
        total_weight = 0.0
        weighted_z = 0.0

        for name, vals in factor_data.items():
            w = self.weights.get(name, 0.0)
            weighted_z += w * vals.get("z_score", 0.0)
            total_weight += w

        composite_z = weighted_z / total_weight if total_weight > 0 else 0.0

        # Approximate percentile from z-score (standard normal CDF)
        composite_pctile = self._z_to_percentile(composite_z)

        return round(composite_z, 4), round(composite_pctile, 1)

    @staticmethod
    def _z_to_percentile(z: float) -> float:
        """Approximate percentile from z-score using logistic approximation."""
        import math
        return 100.0 / (1.0 + math.exp(-1.7 * z))

    @staticmethod
    def _determine_signal(percentile: float) -> dict[str, object]:
        """Determine signal direction and strength from percentile."""
        if percentile >= 80:
            return {"direction": "strong_long", "strength": min(1.0, (percentile - 80) / 20 + 0.7)}
        if percentile >= 60:
            return {"direction": "long", "strength": 0.4 + (percentile - 60) / 50}
        if percentile >= 40:
            return {"direction": "neutral", "strength": 0.2}
        if percentile >= 20:
            return {"direction": "short", "strength": 0.4 + (40 - percentile) / 50}
        return {"direction": "strong_short", "strength": min(1.0, (20 - percentile) / 20 + 0.7)}

    @staticmethod
    def _format_factors(factor_scores: list[FactorScore]) -> str:
        """Format factor scores for LLM prompt."""
        lines = []
        for f in factor_scores:
            lines.append(
                f"- {f.factor_name}: z={f.z_score:+.2f}, "
                f"percentile={f.percentile:.0f}%"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_backtest(stats: dict) -> str:
        """Format backtest stats for prompt."""
        lines = [
            f"Total Return: {stats.get('total_return', 0):.1%}",
            f"Annualized Return: {stats.get('annualized_return', 0):.1%}",
            f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}",
            f"Max Drawdown: {stats.get('max_drawdown', 0):.1%}",
        ]
        return "\n".join(lines)

    def _call_llm(self, prompt: str) -> str:
        """Call Claude Sonnet."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=QUANT_SIGNAL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _parse_response(self, response_text: str) -> dict:
        """Parse LLM JSON response."""
        json_str = _extract_json(response_text)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _empty_result(symbol: str, analysis_date: date) -> QuantSignalOutput:
        """Return empty result when no data."""
        return QuantSignalOutput(
            symbol=symbol,
            analysis_date=analysis_date,
            signal_direction="neutral",
            signal_strength=0.0,
            confidence_level="low",
        )


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response."""
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        return text[brace_start:brace_end + 1]
    return text
