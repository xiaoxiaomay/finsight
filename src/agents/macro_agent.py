"""Macro Environment Agent.

Analyzes macroeconomic indicators from FRED to assess the economic
environment and its implications for equity markets.
Produces MacroAssessmentOutput.
"""

from __future__ import annotations

import json
from datetime import date

import anthropic

from src.agents.prompts.macro_assessment import (
    MACRO_ASSESSMENT_PROMPT,
    MACRO_SYSTEM_PROMPT,
)
from src.agents.schemas.macro import MacroAssessmentOutput
from src.agents.tools.fred_api import (
    assess_fed_stance,
    compute_yield_curve_status,
    fetch_macro_snapshot,
)
from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger("agent.macro")


class MacroAgent:
    """Agent for macroeconomic environment assessment.

    Uses:
    - FRED API tool for macro indicators
    - Yield curve analysis
    - Fed policy stance assessment
    - Claude Sonnet for synthesis
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        settings = get_settings()
        self.client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens

    def assess(
        self,
        assessment_date: date | None = None,
        macro_data: dict | None = None,
    ) -> MacroAssessmentOutput:
        """Run macroeconomic assessment.

        Args:
            assessment_date: Date of assessment (defaults to today).
            macro_data: Optional pre-fetched macro data snapshot.
                If None, will fetch from FRED API.

        Returns:
            MacroAssessmentOutput Pydantic model.
        """
        if assessment_date is None:
            assessment_date = date.today()

        # Step 1: Fetch macro data
        if macro_data is None:
            macro_data = fetch_macro_snapshot()

        macro_text = self._format_macro_data(macro_data)

        # Step 2: Yield curve analysis
        yield_curve = compute_yield_curve_status()

        # Step 3: Fed policy assessment
        fed = assess_fed_stance()

        # Step 4: Build prompt
        prompt = MACRO_ASSESSMENT_PROMPT.format(
            assessment_date=str(assessment_date),
            macro_data=macro_text,
            yield_curve_status=yield_curve,
            fed_assessment=json.dumps(fed, indent=2),
        )

        # Step 5: Call Claude
        response_text = self._call_llm(prompt)

        # Step 6: Parse into Pydantic model
        return self._parse_response(response_text, assessment_date)

    def assess_from_data(
        self,
        macro_data: dict,
        yield_curve_status: str = "unknown",
        fed_assessment: dict | None = None,
        assessment_date: date | None = None,
    ) -> MacroAssessmentOutput:
        """Run assessment from pre-fetched data (useful for testing).

        Args:
            macro_data: Pre-formatted macro data dict.
            yield_curve_status: Pre-computed yield curve status.
            fed_assessment: Pre-computed Fed stance.
            assessment_date: Date of assessment.

        Returns:
            MacroAssessmentOutput Pydantic model.
        """
        if assessment_date is None:
            assessment_date = date.today()
        if fed_assessment is None:
            fed_assessment = {"fed_policy_stance": "unknown", "rate_direction": "unknown"}

        macro_text = self._format_macro_data(macro_data)

        prompt = MACRO_ASSESSMENT_PROMPT.format(
            assessment_date=str(assessment_date),
            macro_data=macro_text,
            yield_curve_status=yield_curve_status,
            fed_assessment=json.dumps(fed_assessment, indent=2),
        )

        response_text = self._call_llm(prompt)
        return self._parse_response(response_text, assessment_date)

    def _call_llm(self, prompt: str) -> str:
        """Call Claude Sonnet and return the response text."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=MACRO_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception:
            logger.warning("llm_call_failed", agent="macro")
            raise

    def _parse_response(
        self,
        response_text: str,
        assessment_date: date,
    ) -> MacroAssessmentOutput:
        """Parse LLM response into structured output."""
        json_str = _extract_json(response_text)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("json_parse_failed", agent="macro")
            data = {}

        data.setdefault("assessment_date", str(assessment_date))
        return MacroAssessmentOutput(**data)

    @staticmethod
    def _format_macro_data(macro_data: dict) -> str:
        """Format macro data snapshot for LLM prompt."""
        lines = []
        for sid, info in macro_data.items():
            if isinstance(info, dict):
                desc = info.get("description", sid)
                val = info.get("value", "N/A")
                trend = info.get("trend", "unknown")
                change = info.get("change", 0)
                lines.append(
                    f"- {desc} ({sid}): {val} (change: {change:+.3f}, trend: {trend})"
                )
            else:
                lines.append(f"- {sid}: {info}")
        return "\n".join(lines) if lines else "No macro data available."


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
