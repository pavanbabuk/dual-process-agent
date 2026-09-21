"""Tests for MCP Host and Tool Registry."""

import os
from dual_agent.mcp_host import MCPHost, MCPToolDefinition, ToolExecutionResult


def test_mcp_host_default_tools():
    host = MCPHost()
    tool_names = [t.name for t in host.list_tools()]
    assert "list_directory" in tool_names
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "run_shell_command" in tool_names


def test_mcp_host_execute_tool():
    host = MCPHost()
    res: ToolExecutionResult = host.execute_tool("list_directory", {"path": "."})
    assert res.success is True
    assert "entries" in res.output
    assert res.execution_time_ms >= 0.0


def test_mcp_host_custom_tool():
    host = MCPHost()
    host.register_tool(
        MCPToolDefinition(
            name="multiply",
            description="Multiplies two numbers",
            handler=lambda args: args["a"] * args["b"],
        )
    )
    res = host.execute_tool("multiply", {"a": 6, "b": 7})
    assert res.success is True
    assert res.output == 42
