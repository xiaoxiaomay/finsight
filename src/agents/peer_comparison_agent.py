"""Peer Comparison Agent.

Compares a target company against sector peers on valuation,
profitability, and growth metrics. Produces PeerComparisonOutput.
"""

from __future__ import annotations

import json
from datetime import date

import anthropic

from src.agents.prompts.peer_comparison import (
    PEER_COMPARISON_PROMPT,
    PEER_COMPARISON_SYSTEM_PROMPT,
)
from src.agents.schemas.peer_comparison import (
    PeerComparisonOutput,
    PeerMetrics,
)
from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger("agent.peer_comparison")

# GICS sector -> representative tickers mapping
SECTOR_PEERS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AVGO", "ADBE", "CRM", "ORCL", "CSCO"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "LLY", "MRK", "TMO", "ABT", "DHR", "BMY"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "C", "AXP", "USB"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG"],
    "Consumer Staples": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "GIS"],
    "Industrials": ["CAT", "RTX", "HON", "UNP", "BA", "DE", "GE", "LMT", "MMM", "UPS"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "PXD", "OXY"],
    "Materials": ["LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DOW", "DD", "PPG"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "WEC", "ED"],
    "Real Estate": ["PLD", "AMT", "CCI", "EQIX", "SPG", "PSA", "O", "WELL", "DLR", "AVB"],
    "Communication Services": ["GOOGL", "META", "DIS", "CMCSA", "NFLX", "T", "VZ", "TMUS", "CHTR", "EA"],
}


class PeerComparisonAgent:
    """Agent for peer comparison analysis.

    Uses:
    - GICS sector mapping for peer identification
    - Financial metrics for comparison
    - Claude Sonnet for competitive analysis
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        settings = get_settings()
        self.client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens

    def analyze(
        self,
        symbol: str,
        sector: str = "",
        industry: str = "",
        target_data: dict | None = None,
        peers_data: list[dict] | None = None,
        analysis_date: date | None = None,
    ) -> PeerComparisonOutput:
        """Run peer comparison analysis.

        Args:
            symbol: Target stock ticker.
            sector: GICS sector.
            industry: GICS industry.
            target_data: Dict with target company metrics.
            peers_data: List of dicts with peer metrics.
            analysis_date: Date of analysis.

        Returns:
            PeerComparisonOutput Pydantic model.
        """
        if analysis_date is None:
            analysis_date = date.today()

        if not target_data:
            logger.warning("no_target_data", symbol=symbol)
            return self._empty_result(symbol, analysis_date, sector)

        # Step 1: Build target metrics
        target_metrics = PeerMetrics(symbol=symbol, **target_data)

        # Step 2: Build peer metrics
        peers = []
        if peers_data:
            for p in peers_data:
                if p.get("symbol", "") != symbol:
                    peers.append(PeerMetrics(**p))

        if not peers:
            logger.warning("no_peer_data", symbol=symbol, sector=sector)
            return PeerComparisonOutput(
                symbol=symbol,
                analysis_date=analysis_date,
                sector=sector,
                industry=industry,
                target_metrics=target_metrics,
                confidence_level="low",
            )

        # Step 3: Compute sector averages
        sector_avgs = self._compute_sector_averages(peers)

        # Step 4: Compute relative positioning
        pe_rel = self._relative_position(target_metrics.pe_ratio, sector_avgs.get("pe_ratio"))
        pb_rel = self._relative_position(target_metrics.pb_ratio, sector_avgs.get("pb_ratio"))
        ev_rel = self._relative_position(target_metrics.ev_ebitda, sector_avgs.get("ev_ebitda"))

        # Step 5: Compute rankings
        prof_rank = self._rank_metric(target_metrics.roe, peers, "roe", higher_better=True)
        growth_rank = self._rank_metric(
            target_metrics.revenue_growth_yoy, peers, "revenue_growth_yoy", higher_better=True,
        )
        overall_rank = (prof_rank + growth_rank) // 2

        # Step 6: Call LLM for qualitative analysis
        target_text = self._format_metrics(target_metrics)
        peer_text = "\n\n".join([
            f"### {p.symbol}\n{self._format_metrics(p)}" for p in peers[:8]
        ])
        avg_text = "\n".join([f"- {k}: {v:.2f}" for k, v in sector_avgs.items() if v])

        prompt = PEER_COMPARISON_PROMPT.format(
            symbol=symbol,
            analysis_date=str(analysis_date),
            sector=sector,
            industry=industry,
            target_metrics=target_text,
            peer_data=peer_text,
            sector_averages=avg_text,
        )

        try:
            response_text = self._call_llm(prompt)
            parsed = self._parse_response(response_text)
        except Exception:
            logger.warning("llm_call_failed", agent="peer_comparison")
            parsed = {}

        return PeerComparisonOutput(
            symbol=symbol,
            analysis_date=analysis_date,
            sector=sector,
            industry=industry,
            target_metrics=target_metrics,
            peers=peers,
            peer_count=len(peers),
            pe_vs_peers=pe_rel,
            pb_vs_peers=pb_rel,
            ev_ebitda_vs_peers=ev_rel,
            valuation_summary=parsed.get("valuation_summary", ""),
            profitability_rank=prof_rank,
            growth_rank=growth_rank,
            overall_rank=overall_rank,
            competitive_advantages=parsed.get("competitive_advantages", []),
            competitive_weaknesses=parsed.get("competitive_weaknesses", []),
            peer_comparison_summary=parsed.get("peer_comparison_summary", ""),
            confidence_level=parsed.get("confidence_level", "medium"),
        )

    @staticmethod
    def _compute_sector_averages(peers: list[PeerMetrics]) -> dict[str, float]:
        """Compute average metrics across peers."""
        metrics = ["pe_ratio", "pb_ratio", "ev_ebitda", "roe", "gross_margin",
                    "operating_margin", "net_margin", "revenue_growth_yoy"]
        avgs: dict[str, float] = {}
        for m in metrics:
            vals = [getattr(p, m) for p in peers if getattr(p, m) is not None]
            if vals:
                avgs[m] = sum(vals) / len(vals)
        return avgs

    @staticmethod
    def _relative_position(value: float | None, avg: float | None) -> str:
        """Determine if value is premium/discount/in_line vs average."""
        if value is None or avg is None or avg == 0:
            return "in_line"
        ratio = value / avg
        if ratio > 1.1:
            return "premium"
        if ratio < 0.9:
            return "discount"
        return "in_line"

    @staticmethod
    def _rank_metric(
        value: float | None,
        peers: list[PeerMetrics],
        attr: str,
        higher_better: bool = True,
    ) -> int:
        """Rank target among peers for a given metric."""
        if value is None:
            return len(peers) + 1
        all_vals = [value]
        for p in peers:
            v = getattr(p, attr)
            if v is not None:
                all_vals.append(v)
        all_vals.sort(reverse=higher_better)
        return all_vals.index(value) + 1

    @staticmethod
    def _format_metrics(m: PeerMetrics) -> str:
        """Format metrics for LLM prompt."""
        lines = [f"Symbol: {m.symbol}"]
        if m.pe_ratio is not None:
            lines.append(f"P/E: {m.pe_ratio:.1f}")
        if m.pb_ratio is not None:
            lines.append(f"P/B: {m.pb_ratio:.1f}")
        if m.ev_ebitda is not None:
            lines.append(f"EV/EBITDA: {m.ev_ebitda:.1f}")
        if m.roe is not None:
            lines.append(f"ROE: {m.roe:.1%}")
        if m.gross_margin is not None:
            lines.append(f"Gross Margin: {m.gross_margin:.1%}")
        if m.operating_margin is not None:
            lines.append(f"Operating Margin: {m.operating_margin:.1%}")
        if m.revenue_growth_yoy is not None:
            lines.append(f"Revenue Growth YoY: {m.revenue_growth_yoy:.1%}")
        return "\n".join(lines)

    @staticmethod
    def get_sector_peers(symbol: str, sector: str) -> list[str]:
        """Get list of peer tickers for a given sector."""
        peers = SECTOR_PEERS.get(sector, [])
        return [p for p in peers if p != symbol][:8]

    def _call_llm(self, prompt: str) -> str:
        """Call Claude Sonnet."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=PEER_COMPARISON_SYSTEM_PROMPT,
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
    def _empty_result(
        symbol: str, analysis_date: date, sector: str,
    ) -> PeerComparisonOutput:
        """Return empty result."""
        return PeerComparisonOutput(
            symbol=symbol,
            analysis_date=analysis_date,
            sector=sector,
            target_metrics=PeerMetrics(symbol=symbol),
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
