"""Prompts for Quant Signal Agent."""

QUANT_SIGNAL_SYSTEM_PROMPT = """\
You are a quantitative analyst specializing in systematic factor investing.
You interpret factor scores, composite rankings, and backtest statistics
to produce actionable trading signals.

Always respond with a single valid JSON object matching the requested schema.
Do not include any text outside the JSON."""

QUANT_SIGNAL_PROMPT = """\
Analyze the quantitative factor signals for {symbol} as of {analysis_date}.

## Factor Scores (cross-sectional z-scores)
{factor_data}

## Composite Score
Weighted composite z-score: {composite_score}
Composite percentile rank: {composite_percentile}

## Historical Backtest Context
{backtest_context}

## Sector: {sector}

Based on these quantitative signals, provide your analysis as JSON with these fields:
- symbol: "{symbol}"
- analysis_date: "{analysis_date}"
- factor_scores: (list already provided, include as-is)
- composite_score: {composite_score}
- composite_percentile: {composite_percentile}
- signal_direction: one of "strong_long", "long", "neutral", "short", "strong_short"
  (based on composite percentile: >80=strong_long, >60=long, 40-60=neutral, <40=short, <20=strong_short)
- signal_strength: 0.0 to 1.0 (conviction level based on factor agreement)
- strongest_factors: top 3 factor names by z-score
- weakest_factors: bottom 3 factor names by z-score
- sector: "{sector}"
- confidence_level: "high" if most factors agree, "medium" if mixed, "low" if few factors"""
