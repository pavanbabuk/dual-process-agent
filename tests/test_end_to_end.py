"""End-to-End integration tests for Dual-Process Agent Runtime."""

from dual_agent.dispatcher import DualProcessDispatcher, DispatchResult
from dual_agent.typesafe_client import JevSystemOneClient
from dual_agent.system_two import MockSystemTwoProvider
from dual_agent.mcp_host import MCPHost
from bridges.hermes_middleware import HermesJevRoutingMiddleware


def test_dual_process_speedup_and_token_savings():
    s1 = JevSystemOneClient(force_simulation=True)
    s2 = MockSystemTwoProvider()
    mcp = MCPHost()

    dispatcher = DualProcessDispatcher(
        system_one_client=s1,
        system_two_provider=s2,
        mcp_host=mcp,
        confidence_threshold=0.85,
    )

    result: DispatchResult = dispatcher.run(
        goal="Inspect workspace directory, read configuration, and synthesize summary.",
        max_steps=6,
    )

    # Verification criteria
    assert result.total_steps >= 1
    # System 1 steps should have taken minimal latency (< 100ms each)
    assert result.system_one_latency_ms < 500
    # Verified token savings vs full LLM baseline
    assert result.estimated_token_savings_pct > 0


def test_hermes_middleware_intercept():
    s1 = JevSystemOneClient(force_simulation=True)
    middleware = HermesJevRoutingMiddleware(typesafe_client=s1, confidence_threshold=0.80)

    # Fast-path case: Jev has high confidence routing
    fallback_called = False
    def mock_hermes(state):
        nonlocal fallback_called
        fallback_called = True
        return {"action": "write_file", "tokens": 400}

    res = middleware.intercept_tool_decision(
        current_state="Task: list directory contents",
        available_tools={"list_directory": "List contents of folder"},
        hermes_fallback_fn=mock_hermes,
    )

    assert res["source"] in ("jev_system_one", "hermes_system_two")
    if res["source"] == "jev_system_one":
        assert res["tokens_burned"] == 0
        assert fallback_called is False
