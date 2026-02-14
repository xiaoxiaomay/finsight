"""Tests for Supervisor routing and query classification."""


class TestQueryClassification:
    """Test query classification logic."""

    def test_full_analysis_trigger(self) -> None:
        from src.agents.supervisor import classify_query

        agents = classify_query("Analyze AAPL")
        assert len(agents) == 5
        assert "earnings" in agents
        assert "macro" in agents
        assert "news" in agents
        assert "quant_signal" in agents
        assert "peer" in agents

    def test_full_report_trigger(self) -> None:
        from src.agents.supervisor import classify_query

        agents = classify_query("Generate a full research report for MSFT")
        assert len(agents) == 5

    def test_quant_only_query(self) -> None:
        from src.agents.supervisor import classify_query

        agents = classify_query("What are the factor scores for AAPL?")
        assert "quant_signal" in agents
        assert len(agents) < 5

    def test_earnings_only_query(self) -> None:
        from src.agents.supervisor import classify_query

        agents = classify_query("Tell me about AAPL revenue and margins")
        assert "earnings" in agents

    def test_macro_only_query(self) -> None:
        from src.agents.supervisor import classify_query

        agents = classify_query("What is the current GDP growth trend?")
        assert "macro" in agents

    def test_news_only_query(self) -> None:
        from src.agents.supervisor import classify_query

        agents = classify_query("Recent news about Apple")
        assert "news" in agents

    def test_peer_only_query(self) -> None:
        from src.agents.supervisor import classify_query

        agents = classify_query("Compare AAPL to its sector peers")
        assert "peer" in agents

    def test_default_to_quant(self) -> None:
        from src.agents.supervisor import classify_query

        agents = classify_query("AAPL stock price?")
        assert "quant_signal" in agents

    def test_multi_agent_query(self) -> None:
        from src.agents.supervisor import classify_query

        agents = classify_query("AAPL earnings and news sentiment")
        assert "earnings" in agents
        assert "news" in agents


class TestRouterNode:
    """Test router node function."""

    def test_router_sets_plan(self) -> None:
        from src.agents.supervisor import router_node

        state = {"query": "Analyze AAPL", "symbol": "AAPL"}
        result = router_node(state)

        assert "plan" in result
        assert len(result["plan"]) == 5

    def test_router_simple_query(self) -> None:
        from src.agents.supervisor import router_node

        state = {"query": "AAPL factor scores", "symbol": "AAPL"}
        result = router_node(state)

        assert "plan" in result
        assert len(result["plan"]) < 5


class TestRouteToAgents:
    """Test conditional routing function."""

    def test_full_analysis_routes_all(self) -> None:
        from src.agents.supervisor import route_to_agents

        state = {"plan": ["earnings", "macro", "news", "quant_signal", "peer"]}
        nodes = route_to_agents(state)

        assert len(nodes) == 5
        assert "earnings_agent" in nodes
        assert "macro_agent" in nodes

    def test_single_agent_route(self) -> None:
        from src.agents.supervisor import route_to_agents

        state = {"plan": ["quant_signal"]}
        nodes = route_to_agents(state)

        assert nodes == ["quant_signal_agent"]

    def test_empty_plan_defaults(self) -> None:
        from src.agents.supervisor import route_to_agents

        state = {"plan": []}
        nodes = route_to_agents(state)

        assert nodes == ["quant_signal_agent"]


class TestGraphConstruction:
    """Test LangGraph supervisor construction."""

    def test_build_graph(self) -> None:
        from src.agents.supervisor import build_supervisor_graph

        graph = build_supervisor_graph()
        assert graph is not None

    def test_compile_graph(self) -> None:
        from src.agents.supervisor import compile_supervisor

        app = compile_supervisor()
        assert app is not None
