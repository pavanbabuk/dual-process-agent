"""Regression test for Fix 8: tool-argument validation on slow path & unknown keys."""

from unittest.mock import MagicMock
from dual_agent.dispatcher import DualProcessDispatcher, validate_tool_args
from dual_agent.typesafe_client import JevDecision
from dual_agent.system_two import SystemTwoResponse
from dual_agent.mcp_host import MCPHost, MCPToolDefinition


def test_validate_tool_args_rejects_unknown_keys():
    host = MCPHost()
    tool = host.get_tool("read_file")
    
    # Valid arg
    ok, _ = validate_tool_args(tool, {"path": "test.txt"})
    assert ok is True

    # Unknown key
    ok, err = validate_tool_args(tool, {"path": "test.txt", "extra_bogus_key": "val"})
    assert ok is False, "validate_tool_args must reject unknown keys"
    assert "unknown argument" in err.lower() or "extra_bogus_key" in err


def test_validate_tool_args_rejects_wrong_types():
    host = MCPHost()
    tool = host.get_tool("read_file")
    
    ok, err = validate_tool_args(tool, {"path": 12345})
    assert ok is False
    assert "must be a string" in err.lower()


def test_slow_path_rejects_unknown_tool_arguments(tmp_path):
    # S1 escalates to S2
    s1 = MagicMock()
    s1.evaluate_state_and_route.return_value = JevDecision(
        is_terminal=False,
        selected_tool="read_file",
        confidence=0.1,
        needs_generation=True,
    )
    s1.force_simulation = False
    s1.simulation_reason = None

    # S2 tries to call read_file with an unknown parameter
    s2 = MagicMock()
    s2.generate_step.return_value = SystemTwoResponse(
        action="read_file",
        arguments={"path": "README.md", "unexpected_option": True},
        thought="reading file with unsupported option",
        latency_ms=100.0,
        tokens_used=50,
        is_mock=False,
    )

    mcp = MCPHost()
    # Spy on execute_tool to ensure it is NOT called with invalid args
    mcp_execute = MagicMock(wraps=mcp.execute_tool)
    mcp.execute_tool = mcp_execute

    dispatcher = DualProcessDispatcher(
        system_one_client=s1,
        system_two_provider=s2,
        mcp_host=mcp,
    )

    res = dispatcher.run(goal="read the readme", max_steps=1)
    
    # execute_tool must NOT have been called because arguments failed schema validation
    mcp_execute.assert_not_called()
    assert len(res.history) if hasattr(res, "history") else len(dispatcher.memory.get_recent_sessions(1)) >= 1
