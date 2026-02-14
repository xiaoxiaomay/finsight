"""LangGraph Supervisor / Router for FinSight Agent System.

Orchestrates multiple agents using a StateGraph:
- Router node: analyzes query and decides which agents to invoke
- Agent nodes: each wraps a domain-specific agent
- Aggregator node: combines results into final report

Simple queries → single agent
Full analysis → fan-out to all 5 agents in parallel
"""

from __future__ import annotations

from datetime import date
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.config.logging_config import get_logger

logger = get_logger("supervisor")


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """State shared across all nodes in the supervisor graph."""

    # Input
    query: str
    symbol: str
    analysis_date: str

    # Router output
    plan: list[str]  # list of agent names to invoke

    # Agent results (populated by agent nodes)
    earnings_result: dict | None
    macro_result: dict | None
    news_result: dict | None
    quant_result: dict | None
    peer_result: dict | None
    regime_result: dict | None

    # Aggregated output
    final_report: dict | None
    error: str | None


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

QUERY_PATTERNS: dict[str, list[str]] = {
    "quant_signal": [
        "factor", "quant", "signal", "score", "rank", "momentum",
        "value factor", "z-score", "percentile",
    ],
    "earnings": [
        "earnings", "revenue", "profit", "margin", "fundamental",
        "10-k", "financial", "balance sheet", "cash flow", "eps",
    ],
    "macro": [
        "macro", "economy", "gdp", "inflation", "interest rate",
        "fed", "unemployment", "yield curve", "recession",
    ],
    "news": [
        "news", "sentiment", "headline", "article", "media",
        "press", "announcement",
    ],
    "peer": [
        "peer", "competitor", "comparison", "relative", "sector",
        "industry", "valuation vs", "compared to",
    ],
}

# Keywords that trigger full analysis
FULL_ANALYSIS_TRIGGERS = [
    "analyze", "full analysis", "research report", "deep dive",
    "comprehensive", "complete analysis", "full report",
]


def classify_query(query: str) -> list[str]:
    """Determine which agents to invoke based on query content.

    Returns list of agent names to invoke.
    """
    query_lower = query.lower()

    # Check for full analysis triggers
    if any(trigger in query_lower for trigger in FULL_ANALYSIS_TRIGGERS):
        return ["earnings", "macro", "news", "quant_signal", "peer"]

    # Match specific agents
    matched: list[str] = []
    for agent_name, keywords in QUERY_PATTERNS.items():
        if any(kw in query_lower for kw in keywords):
            matched.append(agent_name)

    # Default: if nothing matched, use quant signal
    if not matched:
        matched = ["quant_signal"]

    return matched


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> dict[str, Any]:
    """Analyze query and decide which agents to invoke."""
    query = state.get("query", "")
    plan = classify_query(query)
    logger.info("router_decision", query=query[:100], agents=plan)
    return {"plan": plan}


def earnings_node(state: AgentState) -> dict[str, Any]:
    """Run Earnings Analysis Agent."""
    from src.agents.earnings_agent import EarningsAgent

    symbol = state.get("symbol", "")
    analysis_date_str = state.get("analysis_date", "")
    analysis_date = (
        date.fromisoformat(analysis_date_str) if analysis_date_str else date.today()
    )

    try:
        agent = EarningsAgent()
        result = agent.analyze(symbol=symbol, analysis_date=analysis_date)
        return {"earnings_result": result.model_dump(mode="json")}
    except Exception as e:
        logger.warning("earnings_agent_failed", error=str(e))
        return {"earnings_result": {"error": str(e)}}


def macro_node(state: AgentState) -> dict[str, Any]:
    """Run Macro Environment Agent."""
    from src.agents.macro_agent import MacroAgent

    analysis_date_str = state.get("analysis_date", "")
    analysis_date = (
        date.fromisoformat(analysis_date_str) if analysis_date_str else date.today()
    )

    try:
        agent = MacroAgent()
        result = agent.assess(assessment_date=analysis_date)
        return {"macro_result": result.model_dump(mode="json")}
    except Exception as e:
        logger.warning("macro_agent_failed", error=str(e))
        return {"macro_result": {"error": str(e)}}


def news_node(state: AgentState) -> dict[str, Any]:
    """Run News Sentiment Agent."""
    from src.agents.news_agent import NewsAgent

    symbol = state.get("symbol", "")
    analysis_date_str = state.get("analysis_date", "")
    analysis_date = (
        date.fromisoformat(analysis_date_str) if analysis_date_str else date.today()
    )

    try:
        agent = NewsAgent()
        result = agent.analyze(symbol=symbol, analysis_date=analysis_date)
        return {"news_result": result.model_dump(mode="json")}
    except Exception as e:
        logger.warning("news_agent_failed", error=str(e))
        return {"news_result": {"error": str(e)}}


def quant_signal_node(state: AgentState) -> dict[str, Any]:
    """Run Quant Signal Agent."""
    from src.agents.quant_signal_agent import QuantSignalAgent

    symbol = state.get("symbol", "")
    analysis_date_str = state.get("analysis_date", "")
    analysis_date = (
        date.fromisoformat(analysis_date_str) if analysis_date_str else date.today()
    )

    try:
        agent = QuantSignalAgent()
        result = agent.analyze(symbol=symbol, analysis_date=analysis_date)
        return {"quant_result": result.model_dump(mode="json")}
    except Exception as e:
        logger.warning("quant_signal_agent_failed", error=str(e))
        return {"quant_result": {"error": str(e)}}


def peer_node(state: AgentState) -> dict[str, Any]:
    """Run Peer Comparison Agent."""
    from src.agents.peer_comparison_agent import PeerComparisonAgent

    symbol = state.get("symbol", "")
    analysis_date_str = state.get("analysis_date", "")
    analysis_date = (
        date.fromisoformat(analysis_date_str) if analysis_date_str else date.today()
    )

    try:
        agent = PeerComparisonAgent()
        result = agent.analyze(symbol=symbol, analysis_date=analysis_date)
        return {"peer_result": result.model_dump(mode="json")}
    except Exception as e:
        logger.warning("peer_agent_failed", error=str(e))
        return {"peer_result": {"error": str(e)}}


def aggregator_node(state: AgentState) -> dict[str, Any]:
    """Aggregate all agent results into a final report."""
    from src.agents.aggregator import aggregate_results

    symbol = state.get("symbol", "")
    results = {
        "earnings": state.get("earnings_result"),
        "macro": state.get("macro_result"),
        "news": state.get("news_result"),
        "quant_signal": state.get("quant_result"),
        "peer": state.get("peer_result"),
        "regime": state.get("regime_result"),
    }

    # Filter out None entries
    results = {k: v for k, v in results.items() if v is not None}

    try:
        report = aggregate_results(symbol=symbol, agent_results=results)
        return {"final_report": report}
    except Exception as e:
        logger.warning("aggregator_failed", error=str(e))
        return {"final_report": {"error": str(e)}}


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

AGENT_NODE_MAP = {
    "earnings": "earnings_agent",
    "macro": "macro_agent",
    "news": "news_agent",
    "quant_signal": "quant_signal_agent",
    "peer": "peer_agent",
}


def route_to_agents(state: AgentState) -> list[str]:
    """Route to the appropriate agent nodes based on plan.

    Returns list of node names to invoke (parallel fan-out).
    """
    plan = state.get("plan", [])
    nodes = []
    for agent_name in plan:
        node = AGENT_NODE_MAP.get(agent_name)
        if node:
            nodes.append(node)

    if not nodes:
        nodes = ["quant_signal_agent"]

    return nodes


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_supervisor_graph() -> StateGraph:
    """Build the LangGraph supervisor StateGraph.

    Graph structure:
        router → [agent_1, agent_2, ...] → aggregator → END

    The router decides which agents to call.
    Agents run (conceptually in parallel for full analysis).
    The aggregator combines all results.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("earnings_agent", earnings_node)
    graph.add_node("macro_agent", macro_node)
    graph.add_node("news_agent", news_node)
    graph.add_node("quant_signal_agent", quant_signal_node)
    graph.add_node("peer_agent", peer_node)
    graph.add_node("aggregator", aggregator_node)

    # Router → conditional fan-out to agents
    graph.add_conditional_edges(
        "router",
        route_to_agents,
        {
            "earnings_agent": "earnings_agent",
            "macro_agent": "macro_agent",
            "news_agent": "news_agent",
            "quant_signal_agent": "quant_signal_agent",
            "peer_agent": "peer_agent",
        },
    )

    # Each agent → aggregator
    for node_name in AGENT_NODE_MAP.values():
        graph.add_edge(node_name, "aggregator")

    # Aggregator → END
    graph.add_edge("aggregator", END)

    # Entry point
    graph.set_entry_point("router")

    return graph


def compile_supervisor():
    """Compile and return the supervisor graph ready for invocation."""
    graph = build_supervisor_graph()
    return graph.compile()


def run_analysis(
    query: str,
    symbol: str,
    analysis_date: str | None = None,
) -> dict:
    """Run the full supervisor pipeline.

    Args:
        query: User query (e.g., "Analyze AAPL" or "AAPL factor scores").
        symbol: Stock ticker.
        analysis_date: ISO date string (defaults to today).

    Returns:
        Final report dict.
    """
    if analysis_date is None:
        analysis_date = str(date.today())

    app = compile_supervisor()
    initial_state: AgentState = {
        "query": query,
        "symbol": symbol,
        "analysis_date": analysis_date,
    }

    result = app.invoke(initial_state)
    return result.get("final_report", {})
