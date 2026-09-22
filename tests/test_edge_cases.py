"""Edge case and robustness tests for Dual-Process Agent Runtime."""

import pytest
from dual_agent.state import AgentState, StepRecord, StepType
from dual_agent.mcp_host import MCPHost, MCPToolDefinition, ToolExecutionResult
from dual_agent.typesafe_client import JevSystemOneClient
from dual_agent.system_two import get_system_two_provider, MockSystemTwoProvider
from dual_agent.dispatcher import DualProcessDispatcher, DispatchResult
from dual_agent.evaluator import JevEvaluator
from bridges.mcp_evaluator_server import handle_call_tool


def test_mcp_host_tool_failure_handling():
    """Verify MCP host catches exceptions in tools gracefully without crashing."""
    host = MCPHost()

    def faulty_handler(args):
        raise ValueError("Simulated hardware/network failure")

    host.register_tool(
        MCPToolDefinition(
            name="faulty_tool",
            description="A tool that always fails",
            handler=faulty_handler,
        )
    )

    result: ToolExecutionResult = host.execute_tool("faulty_tool", {})
    assert result.success is False
    assert result.output is None
    assert "Simulated hardware/network failure" in result.error


def test_mcp_host_unregistered_tool():
    """Verify attempting to call an unregistered tool returns a clean error."""
    host = MCPHost()
    result = host.execute_tool("non_existent_tool", {})
    assert result.success is False
    assert "not registered" in result.error


def test_dispatcher_max_steps_limit():
    """Verify dispatcher halts when max_steps is reached without infinite looping."""
    s1 = JevSystemOneClient(force_simulation=True)
    s2 = MockSystemTwoProvider()
    mcp = MCPHost()

    dispatcher = DualProcessDispatcher(
        system_one_client=s1,
        system_two_provider=s2,
        mcp_host=mcp,
    )

    # Force a very low max_steps (e.g. 2) on an open-ended goal
    res: DispatchResult = dispatcher.run(goal="Infinite loop test goal", max_steps=2)
    assert res.total_steps <= 2
    assert res.total_latency_ms > 0


def test_evaluator_rubric_scoring_and_completion():
    """Verify JevEvaluator handles arbitrary strings and scores appropriately."""
    client = JevSystemOneClient(force_simulation=True)
    evaluator = JevEvaluator(client=client)
    
    score_high = evaluator.score_output("Complete detailed valid output content.", rubric="Completeness")
    assert 1 <= score_high <= 5

    score_empty = evaluator.score_output("", rubric="Completeness")
    assert 1 <= score_empty <= 5

    state = AgentState(goal="Test goal")
    is_done = evaluator.check_task_completion(state)
    assert isinstance(is_done, bool)


def test_mcp_evaluator_server_tools():
    """Verify bridges/mcp_evaluator_server handler functions."""
    client = JevSystemOneClient(force_simulation=True)

    # 1. jev_evaluate_noul
    res_noul = handle_call_tool(
        "jev_evaluate_noul",
        {"state": "Tests all passed", "criteria": "Is the system healthy?"},
        client,
    )
    assert "is_met" in res_noul
    assert "probability" in res_noul
    assert "latency_ms" in res_noul

    # 2. jev_score_rubric
    res_score = handle_call_tool(
        "jev_score_rubric",
        {"content": "def calculate(): return 42", "rubric": "Code quality"},
        client,
    )
    assert "score" in res_score
    assert 1 <= res_score["score"] <= 5
    assert res_score["max_score"] == 5

    # 3. Unknown tool raises ValueError
    with pytest.raises(ValueError, match="Unknown tool"):
        handle_call_tool("invalid_tool", {}, client)


def test_system_two_provider_fallback():
    """Verify unknown provider strings fall back safely to Mock provider."""
    provider = get_system_two_provider("non_existent_provider_abc")
    assert isinstance(provider, MockSystemTwoProvider)

    resp = provider.generate_step("Test prompt")
    assert resp.action in ("write_file", "finish_task")
    # No model is called by the mock, so it must NOT report token usage — that
    # number used to be a hardcoded 450, which made canned output look like real
    # inference in the telemetry and in the session store.
    assert resp.tokens_used == 0
    assert resp.is_mock is True
    assert resp.degraded_reason
    assert resp.latency_ms > 0


def test_agent_state_prompt_formatting():
    """Verify AgentState serializes compact System 1 state and rich System 2 prompts."""
    state = AgentState(goal="Build a data scraper")
    state.history.append(
        StepRecord(
            step_index=1,
            step_type=StepType.SYSTEM_ONE_FAST_TOOL,
            action="list_directory",
            action_input={"path": "."},
            output='{"entries": ["data.csv"]}',
            latency_ms=15.0,
        )
    )

    s1_text = state.to_system_one_state()
    assert "GOAL: Build a data scraper" in s1_text
    assert "list_directory" in s1_text

    s2_text = state.to_system_two_prompt("### Tool: `run_shell_command`")
    assert "GOAL: Build a data scraper" in s2_text
    assert "AVAILABLE MCP TOOLS:" in s2_text
    assert "run_shell_command" in s2_text
