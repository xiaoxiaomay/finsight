"""Result Aggregator and Quality Checker.

Collects structured outputs from all agents, detects conflicts,
computes composite scores, and generates investment recommendations.
"""

from __future__ import annotations

from src.config.logging_config import get_logger

logger = get_logger("aggregator")

# Recommendation thresholds
RECOMMENDATION_THRESHOLDS = [
    (80, "Strong Buy"),
    (65, "Buy"),
    (45, "Hold"),
    (30, "Sell"),
    (0, "Strong Sell"),
]

# Agent weight in composite score
AGENT_WEIGHTS = {
    "earnings": 0.35,
    "quant_signal": 0.25,
    "news": 0.15,
    "macro": 0.15,
    "peer": 0.10,
}


def aggregate_results(
    symbol: str,
    agent_results: dict[str, dict],
) -> dict:
    """Aggregate all agent results into a unified report.

    Args:
        symbol: Stock ticker.
        agent_results: Dict of agent_name -> result dict.

    Returns:
        Aggregated report dict.
    """
    # Extract component scores
    scores = _extract_scores(agent_results)

    # Compute weighted composite
    composite = _compute_composite_score(scores)

    # Determine recommendation
    recommendation = _score_to_recommendation(composite)

    # Detect conflicts between agents
    conflicts = _detect_conflicts(agent_results)

    # Determine conviction based on agreement
    conviction = _assess_conviction(scores, conflicts)

    # Consolidate risks and catalysts
    key_risks = _consolidate_list(agent_results, "key_risks", limit=5)
    catalysts = _consolidate_list(agent_results, "catalysts", limit=5)
    bull_case = _build_bull_case(agent_results)
    bear_case = _build_bear_case(agent_results)

    # Build agent sources
    agents_used = list(agent_results.keys())
    data_sources = _collect_data_sources(agent_results)

    # Regime context
    regime_result = agent_results.get("regime", {})

    report = {
        "symbol": symbol,
        "recommendation": recommendation,
        "conviction": conviction,
        "composite_score": round(composite, 1),
        "fundamental_score": scores.get("earnings"),
        "quant_score": scores.get("quant_signal"),
        "sentiment_score": scores.get("news"),
        "macro_score": scores.get("macro"),
        "peer_score": scores.get("peer"),
        "earnings_analysis": agent_results.get("earnings"),
        "macro_assessment": agent_results.get("macro"),
        "news_sentiment": agent_results.get("news"),
        "quant_signals": agent_results.get("quant_signal"),
        "peer_comparison": agent_results.get("peer"),
        "conflicts": [c for c in conflicts],
        "agents_used": agents_used,
        "data_sources": data_sources,
        "key_risks": key_risks,
        "catalysts": catalysts,
        "bull_case": bull_case,
        "bear_case": bear_case,
        "market_regime": regime_result.get("regime", ""),
        "regime_factor_tilt": regime_result.get("factor_tilt", ""),
    }

    logger.info(
        "aggregation_complete",
        symbol=symbol,
        composite=composite,
        recommendation=recommendation,
        agents=agents_used,
    )

    return report


def _extract_scores(agent_results: dict[str, dict]) -> dict[str, float]:
    """Extract normalized 0-100 scores from each agent result."""
    scores: dict[str, float] = {}

    # Earnings -> fundamental_score (already 0-100)
    earnings = agent_results.get("earnings", {})
    if earnings and "fundamental_score" in earnings:
        scores["earnings"] = float(earnings["fundamental_score"])

    # Quant signal -> composite_percentile (already 0-100)
    quant = agent_results.get("quant_signal", {})
    if quant and "composite_percentile" in quant:
        scores["quant_signal"] = float(quant["composite_percentile"])

    # News -> overall_sentiment (-1 to +1) -> map to 0-100
    news = agent_results.get("news", {})
    if news and "overall_sentiment" in news:
        sentiment = float(news["overall_sentiment"])
        scores["news"] = (sentiment + 1) * 50  # [-1,1] -> [0,100]

    # Macro -> suggested_equity_allocation (0-1) -> map to 0-100
    macro = agent_results.get("macro", {})
    if macro and "suggested_equity_allocation" in macro:
        alloc = float(macro["suggested_equity_allocation"])
        scores["macro"] = alloc * 100

    # Peer -> overall_rank (1=best) -> map inversely
    peer = agent_results.get("peer", {})
    if peer and "overall_rank" in peer and "peer_count" in peer:
        rank = int(peer["overall_rank"])
        count = int(peer["peer_count"])
        if count > 0 and rank > 0:
            scores["peer"] = max(0, (1 - (rank - 1) / max(count, 1)) * 100)

    return scores


def _compute_composite_score(scores: dict[str, float]) -> float:
    """Compute weighted composite score from agent scores."""
    if not scores:
        return 50.0

    total_weight = 0.0
    weighted_sum = 0.0

    for agent, score in scores.items():
        w = AGENT_WEIGHTS.get(agent, 0.1)
        weighted_sum += w * score
        total_weight += w

    return weighted_sum / total_weight if total_weight > 0 else 50.0


def _score_to_recommendation(score: float) -> str:
    """Convert composite score to recommendation."""
    for threshold, label in RECOMMENDATION_THRESHOLDS:
        if score >= threshold:
            return label
    return "Strong Sell"


def _detect_conflicts(agent_results: dict[str, dict]) -> list[dict]:
    """Detect conflicting signals between agents."""
    conflicts = []

    # Check earnings vs news conflict
    earnings = agent_results.get("earnings", {})
    news = agent_results.get("news", {})
    if earnings and news:
        earnings_score = earnings.get("fundamental_score", 50)
        news_sentiment = news.get("overall_sentiment", 0)
        if earnings_score > 70 and news_sentiment < -0.3:
            conflicts.append({
                "agents": ["earnings", "news"],
                "description": "Earnings analysis is positive but news sentiment is negative",
                "resolution": "Fundamentals suggest strength despite negative press. "
                "Monitor for news-driven selloff opportunity.",
            })
        elif earnings_score < 40 and news_sentiment > 0.3:
            conflicts.append({
                "agents": ["earnings", "news"],
                "description": "Earnings analysis is weak but news sentiment is positive",
                "resolution": "Positive news may not be sustainable given fundamental weakness.",
            })

    # Check quant vs macro conflict
    quant = agent_results.get("quant_signal", {})
    macro = agent_results.get("macro", {})
    if quant and macro:
        quant_pctile = quant.get("composite_percentile", 50)
        macro_outlook = macro.get("equity_outlook", "neutral")
        if quant_pctile > 70 and macro_outlook == "bearish":
            conflicts.append({
                "agents": ["quant_signal", "macro"],
                "description": "Quant signal is bullish but macro outlook is bearish",
                "resolution": "Stock-specific strength may face headwinds from macro environment.",
            })

    return conflicts


def _assess_conviction(
    scores: dict[str, float],
    conflicts: list[dict],
) -> str:
    """Assess conviction level based on score agreement and conflicts."""
    if not scores:
        return "low"

    # Check variance of scores
    vals = list(scores.values())
    if len(vals) < 2:
        return "medium"

    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)

    # High conviction: low variance, no conflicts
    if variance < 200 and len(conflicts) == 0:
        return "high"
    # Low conviction: high variance or multiple conflicts
    if variance > 600 or len(conflicts) > 1:
        return "low"
    return "medium"


def _consolidate_list(
    agent_results: dict[str, dict],
    field: str,
    limit: int = 5,
) -> list[str]:
    """Consolidate list fields from all agents."""
    items: list[str] = []
    for result in agent_results.values():
        if isinstance(result, dict):
            vals = result.get(field, [])
            if isinstance(vals, list):
                items.extend(str(v) for v in vals)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item.lower() not in seen:
            seen.add(item.lower())
            unique.append(item)
    return unique[:limit]


def _build_bull_case(agent_results: dict[str, dict]) -> list[str]:
    """Build bull case from positive signals across agents."""
    points: list[str] = []

    earnings = agent_results.get("earnings", {})
    if earnings.get("fundamental_score", 0) > 60:
        thesis = earnings.get("investment_thesis", "")
        if thesis:
            points.append(thesis[:200])

    quant = agent_results.get("quant_signal", {})
    if quant.get("composite_percentile", 0) > 60:
        strongest = quant.get("strongest_factors", [])
        if strongest:
            points.append(f"Strong factor exposure: {', '.join(strongest[:3])}")

    news = agent_results.get("news", {})
    if news.get("overall_sentiment", 0) > 0.2:
        points.append(f"Positive news sentiment ({news.get('sentiment_label', 'positive')})")

    # Add catalysts
    points.extend(_consolidate_list(agent_results, "catalysts", limit=2))

    return points[:5]


def _build_bear_case(agent_results: dict[str, dict]) -> list[str]:
    """Build bear case from negative signals across agents."""
    points: list[str] = []

    earnings = agent_results.get("earnings", {})
    if earnings.get("fundamental_score", 100) < 50:
        points.append(f"Weak fundamentals (score: {earnings.get('fundamental_score', 0):.0f}/100)")

    quant = agent_results.get("quant_signal", {})
    if quant.get("composite_percentile", 100) < 40:
        weakest = quant.get("weakest_factors", [])
        if weakest:
            points.append(f"Weak factor exposure: {', '.join(weakest[:3])}")

    news = agent_results.get("news", {})
    if news.get("overall_sentiment", 0) < -0.2:
        points.append(f"Negative news sentiment ({news.get('sentiment_label', 'negative')})")

    macro = agent_results.get("macro", {})
    if macro.get("equity_outlook") == "bearish":
        points.append("Bearish macro environment")

    # Add risks
    points.extend(_consolidate_list(agent_results, "key_risks", limit=2))

    return points[:5]


def _collect_data_sources(agent_results: dict[str, dict]) -> list[str]:
    """Collect data source references from all agents."""
    sources: list[str] = []
    source_fields = ["sources_used", "data_sources"]

    for result in agent_results.values():
        if isinstance(result, dict):
            for field in source_fields:
                vals = result.get(field, [])
                if isinstance(vals, list):
                    sources.extend(str(v) for v in vals)

    # Add standard sources based on which agents ran
    if "earnings" in agent_results:
        sources.append("SEC EDGAR Filings")
    if "macro" in agent_results:
        sources.append("Federal Reserve Economic Data (FRED)")
    if "news" in agent_results:
        sources.append("NewsAPI")
    if "quant_signal" in agent_results:
        sources.append("Factor Signal Database")

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for s in sources:
        if s.lower() not in seen:
            seen.add(s.lower())
            unique.append(s)
    return unique
