"""End-to-End integration tests for Dual-Process Agent Runtime."""

from dual_agent.dispatcher import DualProcessDispatcher, DispatchResult
from dual_agent.typesafe_client import JevSystemOneClient
from dual_agent.system_two import MockSystemTwoProvider
from dual_agent.mcp_host import MCPHost
from bridges.hermes_middleware import HermesJevRoutingMiddleware


def test_dual_process_run_reports_measured_values_only():
    """The runtime reports MEASURED values only.

    This test previously asserted `estimated_token_savings_pct > 0`, which was
    satisfied by multiplying the step count by a hardcoded 1500-token baseline —
    it verified arithmetic on a constant, not any real saving. It now asserts the
    opposite property: no invented baseline is reported, the run admits when its
    router was simulated, and simulation latency is disclosed rather than passed
    off as model latency.
    """
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

    assert result.total_steps >= 1

    # No fabricated benchmark fields exist any more.
    assert not hasattr(result, "estimated_token_savings_pct")
    assert not hasattr(result, "speedup_ratio")
    assert not hasattr(result, "estimated_baseline_tokens")

    # The run reports measured latency, and it tells the truth about simulation.
    assert result.total_latency_ms > 0
    assert result.used_simulated_system_one is True
    assert result.system_one_fallback_reason == "force_simulation=True"
    assert result.simulated_latency_ms > 0


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
