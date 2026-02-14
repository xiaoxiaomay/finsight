"""Prompts for Research Report synthesis."""

REPORT_SYSTEM_PROMPT = """\
You are a senior equity research analyst at a top-tier investment bank.
You synthesize multi-dimensional analysis into comprehensive research reports
with clear investment recommendations.

Always respond with a single valid JSON object matching the requested schema.
Do not include any text outside the JSON."""

REPORT_SYNTHESIS_PROMPT = """\
Synthesize the following multi-agent analysis for {symbol} into a comprehensive
research report as of {analysis_date}.

## Earnings / Fundamental Analysis
{earnings_section}

## Macro Environment Assessment
{macro_section}

## News Sentiment Analysis
{news_section}

## Quantitative Signal Analysis
{quant_section}

## Peer Comparison Analysis
{peer_section}

## Market Regime
{regime_section}

Based on ALL the above analyses, produce a JSON research report with:
- symbol: "{symbol}"
- company_name: "{company_name}"
- recommendation: one of "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"
  Weight fundamental quality (40%), quant signals (25%), sentiment (15%), macro (10%), peers (10%)
- conviction: "high" if analyses mostly agree, "medium" if mixed, "low" if conflicting
- composite_score: 0-100 weighted score
- executive_summary: 3-5 sentence overview covering key findings and recommendation
- bull_case: list of 3-5 bullish arguments
- bear_case: list of 3-5 bearish arguments
- fundamental_score: 0-100 from earnings analysis
- quant_score: 0-100 from quant signals (map percentile to score)
- sentiment_score: 0-100 (map sentiment from [-1,1] to [0,100])
- macro_score: 0-100 from macro assessment
- peer_score: 0-100 (based on peer ranking)
- conflicts: list of any conflicting signals between analyses (each with agents, description, resolution)
- key_risks: top 5 risks consolidated from all analyses
- catalysts: top 5 catalysts consolidated from all analyses
- agents_used: list of agent names that provided data
- data_sources: list of data sources referenced"""
