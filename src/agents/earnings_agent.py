"""Earnings Analysis Agent.

Analyzes company financials using RAG retrieval from SEC filings
and structured financial data. Produces EarningsAnalysisOutput.

Uses Claude Sonnet for analysis with structured JSON output.
"""

from __future__ import annotations

import json
from datetime import date

import anthropic

from src.agents.prompts.earnings_analysis import (
    EARNINGS_ANALYSIS_PROMPT,
    EARNINGS_SYSTEM_PROMPT,
)
from src.agents.rag.retriever import HybridRetriever
from src.agents.schemas.earnings import EarningsAnalysisOutput
from src.agents.tools.financial_calc import summarize_fundamentals
from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger("agent.earnings")


class EarningsAgent:
    """Agent for financial earnings analysis.

    Uses:
    - RAG retriever for SEC filing context
    - Financial calculation tool for metrics
    - Claude Sonnet for structured analysis
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.retriever = retriever
        settings = get_settings()
        self.client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens

    def analyze(
        self,
        symbol: str,
        fundamentals_df=None,
        analysis_date: date | None = None,
    ) -> EarningsAnalysisOutput:
        """Run earnings analysis for a company.

        Args:
            symbol: Stock ticker (e.g., 'AAPL').
            fundamentals_df: Optional DataFrame with fundamental data.
            analysis_date: Date of analysis (defaults to today).

        Returns:
            EarningsAnalysisOutput Pydantic model.
        """
        if analysis_date is None:
            analysis_date = date.today()

        # Step 1: Gather financial data
        financial_data = "No fundamental data available."
        if fundamentals_df is not None:
            summary = summarize_fundamentals(fundamentals_df, symbol)
            if summary:
                financial_data = json.dumps(summary, indent=2, default=str)

        # Step 2: RAG retrieval for SEC filing context
        rag_context = "No SEC filing context available."
        if self.retriever and self.retriever.total_chunks > 0:
            results = self.retriever.retrieve(
                query=f"{symbol} revenue earnings profitability cash flow",
                top_k=5,
                symbol=symbol,
            )
            if results:
                rag_context = "\n\n---\n\n".join([
                    f"[Source: {r.chunk.metadata.get('section_name', 'unknown')} | "
                    f"Score: {r.score:.3f}]\n{r.chunk.text[:1500]}"
                    for r in results
                ])

        # Step 3: Build prompt
        prompt = EARNINGS_ANALYSIS_PROMPT.format(
            symbol=symbol,
            analysis_date=str(analysis_date),
            financial_data=financial_data,
            rag_context=rag_context,
        )

        # Step 4: Call Claude
        response_text = self._call_llm(prompt)

        # Step 5: Parse into Pydantic model
        return self._parse_response(response_text, symbol, analysis_date)

    def _call_llm(self, prompt: str) -> str:
        """Call Claude Sonnet and return the response text."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=EARNINGS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception:
            logger.warning("llm_call_failed", agent="earnings")
            raise

    def _parse_response(
        self,
        response_text: str,
        symbol: str,
        analysis_date: date,
    ) -> EarningsAnalysisOutput:
        """Parse LLM response into structured output."""
        # Extract JSON from response (handle markdown code blocks)
        json_str = _extract_json(response_text)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("json_parse_failed", agent="earnings")
            # Return a minimal valid output
            data = {}

        # Ensure required fields
        data.setdefault("symbol", symbol)
        data.setdefault("analysis_date", str(analysis_date))

        return EarningsAnalysisOutput(**data)


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to find JSON in code blocks
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()

    # Try to find raw JSON object
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        return text[brace_start:brace_end + 1]

    return text
