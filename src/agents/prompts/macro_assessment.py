"""Prompt templates for Macro Environment Agent."""

MACRO_SYSTEM_PROMPT = """You are an expert macroeconomist and market strategist. You analyze \
economic indicators, monetary policy, and market conditions to assess the macroeconomic \
environment and its implications for equity markets.

Your analysis must be:
1. Data-driven — reference specific indicator values and trends
2. Historically informed — compare current conditions to past cycles
3. Forward-looking — assess where in the cycle we are heading
4. Actionable — provide clear implications for portfolio positioning"""

MACRO_ASSESSMENT_PROMPT = """Assess the current macroeconomic environment based on the \
following economic indicators.

## Current Macro Data
{macro_data}

## Yield Curve Status
{yield_curve_status}

## Federal Reserve Policy Assessment
{fed_assessment}

## Instructions
Produce a comprehensive macro assessment covering:

1. **Economic Cycle Phase**: Based on GDP growth, employment, industrial production, and \
other indicators, determine the current cycle phase (expansion/peak/contraction/trough). \
Provide a confidence level (0.0 to 1.0).

2. **Key Indicator Trends**: Assess GDP growth, inflation, and employment trends.

3. **Yield Curve & Policy**: Evaluate the yield curve status and Fed policy stance. \
Determine rate direction (hiking/pausing/cutting).

4. **Market Implications**: Based on the macro environment, determine equity outlook \
(bullish/neutral/bearish), preferred sectors, and key risk factors.

5. **Portfolio Positioning**: Suggest equity allocation (0.0 to 1.0), whether to adopt a \
defensive tilt, and provide reasoning.

Respond with a JSON object matching this exact schema:
{{
    "assessment_date": "{assessment_date}",
    "cycle_phase": "expansion|peak|contraction|trough",
    "cycle_confidence": <float 0.0-1.0>,
    "gdp_growth_trend": "accelerating|stable|decelerating|contracting",
    "inflation_trend": "rising|stable|falling",
    "employment_trend": "strengthening|stable|weakening",
    "yield_curve_status": "normal|flat|inverted",
    "fed_policy_stance": "hawkish|neutral|dovish",
    "rate_direction": "hiking|pausing|cutting",
    "equity_outlook": "bullish|neutral|bearish",
    "sector_preferences": [<string>, ...],
    "risk_factors": [<string>, ...],
    "suggested_equity_allocation": <float 0.0-1.0>,
    "defensive_tilt": <bool>,
    "reasoning": "<string>"
}}"""
