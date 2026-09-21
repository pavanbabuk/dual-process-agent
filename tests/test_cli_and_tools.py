"""Tests for CLI entrypoint, built-in MCP tools, and Hermes fallback."""

import os
from dual_agent.mcp_host import MCPHost
from dual_agent.cli import run_agent_task
from bridges.hermes_middleware import HermesJevRoutingMiddleware
from dual_agent.typesafe_client import JevSystemOneClient, JevDecision


def test_cli_run_agent_task_execution(capsys):
    """Verify CLI task runner executes without exceptions and prints output."""
    run_agent_task(
        goal="Test CLI inspection task",
        provider="mock",
        threshold=0.85,
        max_steps=3,
        force_simulation=True,
    )
    captured = capsys.readouterr()
    assert "Dual-Process Agent Runtime" in captured.out
    assert "Execution Telemetry & Benchmark" in captured.out


def test_builtin_mcp_tools(tmp_path):
    """Verify read_file, write_file, and run_shell_command tools."""
    host = MCPHost()

    # 1. write_file
    target_file = str(tmp_path / "sample.txt")
    write_res = host.execute_tool("write_file", {"path": target_file, "content": "Hello World!"})
    assert write_res.success is True
    assert os.path.exists(target_file)

    # 2. read_file
    read_res = host.execute_tool("read_file", {"path": target_file})
    assert read_res.success is True
    assert read_res.output == "Hello World!"

    # 3. read_file missing file
    missing_res = host.execute_tool("read_file", {"path": str(tmp_path / "does_not_exist.txt")})
    assert missing_res.success is True
    assert "Error: File" in missing_res.output

    # 4. run_shell_command
    shell_res = host.execute_tool("run_shell_command", {"command": "echo 'Testing Shell Tool'"})
    assert shell_res.success is True
    assert "Testing Shell Tool" in shell_res.output


def test_hermes_middleware_fallback_execution():
    """Verify Hermes fallback is triggered when Jev confidence is below threshold."""
    # Create client that forces low confidence
    class LowConfJevClient(JevSystemOneClient):
        def evaluate_state_and_route(self, state_text, tool_options, allow_escalation=True):
            return JevDecision(
                selected_tool="unknown_tool",
                confidence=0.30,  # Below 0.85 threshold
                latency_ms=10.0,
                needs_generation=True,
            )

    middleware = HermesJevRoutingMiddleware(
        typesafe_client=LowConfJevClient(force_simulation=True),
        confidence_threshold=0.85,
    )

    hermes_called = False
    def mock_hermes_fn(state):
        nonlocal hermes_called
        hermes_called = True
        return {"action": "write_file", "args": {"path": "test.txt"}}

    res = middleware.intercept_tool_decision(
        current_state="Complex creative writing task",
        available_tools={"write_file": "Write a file"},
        hermes_fallback_fn=mock_hermes_fn,
    )

    assert hermes_called is True
    assert res["source"] == "hermes_system_two"
    assert res["action"] == "write_file"
