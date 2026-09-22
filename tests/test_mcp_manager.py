"""Tests for MCPManager.

The acceptance tests here drive a REAL MCP stdio server — a subprocess that
speaks the protocol on its own stdin/stdout and computes results itself
(`tests/fixtures/echo_mcp_server.py`). Nothing in this file substitutes a Python
mock for the server, because a mock would prove only that the test's own
hardcoded return value can be read back.

What "real" buys, concretely: the expected tool names and schemas are asserted
against constants written independently in this file, and `add_numbers(17, 25)`
is expected to be `"42"` — a value neither the manager nor the host ever sees
until the server process computes it.
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from dual_agent.mcp_manager import (
    HAS_MCP,
    MCPManager,
    MCPTransportError,
)
from dual_agent.mcp_host import MCPHost

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "echo_mcp_server.py")

# Written from the server's own source, not read back from it.
EXPECTED_ADD_SCHEMA = {
    "type": "object",
    "properties": {
        "a": {"type": "integer", "description": "First addend"},
        "b": {"type": "integer", "description": "Second addend"},
    },
    "required": ["a", "b"],
}
EXPECTED_TOOL_NAMES = {"add_numbers", "reverse_text"}


def _write_config(tmp_path, servers: dict) -> str:
    config_file = tmp_path / "mcp_servers.json"
    config_file.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return str(config_file)


def _echo_server(**overrides) -> dict:
    entry = {"command": sys.executable, "args": [FIXTURE]}
    entry.update(overrides)
    return entry


@pytest.fixture
def manager(tmp_path):
    mgr = MCPManager(config_path=str(tmp_path / "mcp_servers.json"))
    yield mgr
    mgr.shutdown()


# --------------------------------------------------------------------------
# Real server, real discovery, real call
# --------------------------------------------------------------------------


def test_discovers_real_tools_with_real_names_and_schemas(tmp_path, manager):
    """(a) tools/list from a live server registers each tool under its real name."""
    manager.config_path = _write_config(tmp_path, {"echo": _echo_server()})
    host = MCPHost()

    attached = manager.attach_to_host(host)

    # Two discovered tools plus the server-level dispatch handle.
    assert attached == len(EXPECTED_TOOL_NAMES) + 1
    registered = {t.name for t in host.list_tools()}
    assert EXPECTED_TOOL_NAMES <= registered

    add_tool = host.get_tool("add_numbers")
    assert add_tool is not None, "tool must be registered under the server's real name"
    assert add_tool.parameters_schema == EXPECTED_ADD_SCHEMA
    assert add_tool.description == "Add two integers and return the sum."

    # The old placeholder shape must not survive: the dispatch handle is now a
    # live forwarder, not a string.
    dispatch = host.get_tool("mcp_echo_dispatch")
    assert dispatch is not None
    assert "NOT CONNECTED" not in dispatch.description
    assert "UNAVAILABLE" not in dispatch.description
    assert "Dispatched" not in dispatch.description


def test_tool_call_returns_the_servers_own_computed_result(tmp_path, manager):
    """(b) The value comes from the subprocess computing it, verified independently."""
    manager.config_path = _write_config(tmp_path, {"echo": _echo_server()})
    host = MCPHost()
    manager.attach_to_host(host)

    result = host.execute_tool("add_numbers", {"a": 17, "b": 25})

    assert result.success is True, f"call failed: {result.error}"
    assert result.output == "42"
    # Independent check of the same arithmetic, so "42" cannot be a coincidence
    # of a hardcoded constant in the client.
    assert str(17 + 25) == result.output

    reversed_result = host.execute_tool("reverse_text", {"text": "dual-process"})
    assert reversed_result.success is True, f"call failed: {reversed_result.error}"
    assert reversed_result.output == "dual-process"[::-1]


def test_second_call_reuses_the_running_server(tmp_path, manager):
    """Lazy start must mean one process for many calls, not one per call."""
    manager.config_path = _write_config(tmp_path, {"echo": _echo_server()})
    host = MCPHost()
    manager.attach_to_host(host)

    transport = manager.connect("echo")
    first_pid = transport._proc.pid

    assert host.execute_tool("add_numbers", {"a": 1, "b": 1}).output == "2"
    assert host.execute_tool("add_numbers", {"a": 2, "b": 2}).output == "4"

    assert manager.connect("echo")._proc.pid == first_pid


def test_arguments_are_validated_by_the_existing_validator(tmp_path, manager):
    """(3) Arg validation reuses dispatcher.validate_tool_args; bad args never sent."""
    manager.config_path = _write_config(tmp_path, {"echo": _echo_server()})
    host = MCPHost()
    manager.attach_to_host(host)

    missing = host.execute_tool("add_numbers", {"a": 1})
    assert missing.success is True  # the handler returns an error STRING
    assert str(missing.output).startswith("Error:")
    assert "b" in str(missing.output)

    wrong_type = host.execute_tool("add_numbers", {"a": "not-an-int", "b": 2})
    assert str(wrong_type.output).startswith("Error:")

    unknown_key = host.execute_tool("add_numbers", {"a": 1, "b": 2, "c": 3})
    assert str(unknown_key.output).startswith("Error:")
    assert "c" in str(unknown_key.output)


def test_shutdown_terminates_the_subprocess(tmp_path, manager):
    """(4) Clean shutdown: the spawned server does not outlive the manager."""
    manager.config_path = _write_config(tmp_path, {"echo": _echo_server()})
    host = MCPHost()
    manager.attach_to_host(host)
    proc = manager.connect("echo")._proc
    assert proc.poll() is None

    manager.shutdown()

    assert proc.poll() is not None, "server subprocess survived shutdown"


# --------------------------------------------------------------------------
# Failure paths must be loud and named
# --------------------------------------------------------------------------


def test_server_that_fails_to_start_is_named_and_not_a_success(tmp_path, manager):
    """(c) A server that fails to start produces a named error, never success."""
    manager.config_path = _write_config(
        tmp_path, {"broken": _echo_server(args=[FIXTURE, "--fail-startup"])}
    )
    host = MCPHost()

    manager.attach_to_host(host)

    tool = host.get_tool("mcp_broken_dispatch")
    assert tool is not None, "an unreachable server must still be addressable"
    result = host.execute_tool("mcp_broken_dispatch", {"action": "anything"})

    assert result.success is True  # handler returns an error string, not a raise
    assert str(result.output).startswith("Error:")
    assert "broken" in str(result.output)
    assert "3" in str(result.output) or "status" in str(result.output).lower()
    # And no tool from the failed server was registered as if it worked.
    assert host.get_tool("add_numbers") is None


def test_missing_command_is_reported_by_name(tmp_path, manager):
    """A command that does not exist must not be launched or reported as success."""
    manager.config_path = _write_config(
        tmp_path, {"ghost": {"command": "definitely-not-a-real-binary-xyz", "args": []}}
    )
    host = MCPHost()
    manager.attach_to_host(host)

    result = host.execute_tool("mcp_ghost_dispatch", {})
    assert str(result.output).startswith("Error:")
    assert "ghost" in str(result.output)
    assert "not found" in str(result.output).lower()


def test_tool_error_result_is_surfaced_with_error_prefix(tmp_path, manager):
    """(2) A server-side tool failure surfaces as 'Error:'."""
    manager.config_path = _write_config(
        tmp_path, {"grumpy": _echo_server(args=[FIXTURE, "--fail-tool"])}
    )
    host = MCPHost()
    manager.attach_to_host(host)

    result = host.execute_tool("add_numbers", {"a": 1, "b": 2})

    assert str(result.output).startswith("Error:")
    assert "refused on purpose" in str(result.output)


def test_server_that_exits_after_handshake_is_reported_not_hung(tmp_path, manager):
    """A server that dies mid-session names the reason instead of blocking."""
    manager.config_path = _write_config(
        tmp_path, {"flaky": _echo_server(args=[FIXTURE, "--exit-after-init"])}
    )
    host = MCPHost()
    manager.attach_to_host(host)

    # tools/list never answered, so the server is unattachable and says why.
    tool = host.get_tool("mcp_flaky_dispatch")
    assert tool is not None
    output = str(host.execute_tool("mcp_flaky_dispatch", {}).output)
    assert output.startswith("Error:")
    assert "flaky" in output


def test_call_timeout_is_reported_by_name(tmp_path, manager, monkeypatch):
    """(4) A slow server produces a named timeout error, not a false success."""
    monkeypatch.setattr("dual_agent.mcp_manager.CALL_TIMEOUT_S", 0.5)
    manager.config_path = _write_config(
        tmp_path, {"slow": _echo_server(args=[FIXTURE, "--sleep", "5"])}
    )
    host = MCPHost()
    # Startup uses its own, longer budget; only the call timeout is shortened.
    monkeypatch.setattr("dual_agent.mcp_manager.STARTUP_TIMEOUT_S", 30)

    manager.attach_to_host(host)

    started = time.perf_counter()
    result = host.execute_tool("add_numbers", {"a": 1, "b": 2})
    elapsed = time.perf_counter() - started

    assert str(result.output).startswith("Error:")
    assert "slow" in str(result.output)
    assert "did not answer" in str(result.output)
    assert elapsed < 5, "the call timeout must fire before the server's own delay"


def test_connect_on_unknown_server_names_it(manager):
    with pytest.raises(MCPTransportError) as excinfo:
        manager.connect("never-configured")
    assert "never-configured" in str(excinfo.value)


# --------------------------------------------------------------------------
# The removed behaviour cannot come back
# --------------------------------------------------------------------------


def test_fake_dispatch_string_cannot_be_produced(tmp_path, manager):
    """(d) The 'Dispatched to external MCP server' string is unreachable.

    Checked two ways: the source tree no longer contains the lambda, and every
    path this manager can take for a configured server returns either the
    server's real output or a string starting with 'Error:'.
    """
    src_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    offenders = []
    for dirpath, _dirnames, filenames in os.walk(src_root):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            full = os.path.join(dirpath, filename)
            with open(full, "r", encoding="utf-8") as f:
                text = f.read()
            if "Dispatched to external MCP server" in text:
                offenders.append(full)
    assert offenders == [], f"placeholder dispatch string still present in {offenders}"

    manager.config_path = _write_config(
        tmp_path,
        {
            "echo": _echo_server(),
            "broken": _echo_server(args=[FIXTURE, "--fail-startup"]),
        },
    )
    host = MCPHost()
    manager.attach_to_host(host)

    assert host.execute_tool("add_numbers", {"a": 1, "b": 2}).output == "3"

    for tool_name in ("mcp_echo_dispatch", "mcp_broken_dispatch"):
        tool = host.get_tool(tool_name)
        assert tool is not None
        output = str(host.execute_tool(tool_name, {"action": "list_repos"}).output)
        assert "Dispatched to external MCP server" not in output
        assert output.startswith("Error:"), f"{tool_name} returned a non-error: {output}"


def test_disabled_servers_are_not_started(tmp_path, manager):
    manager.config_path = _write_config(
        tmp_path, {"off": _echo_server(disabled=True)}
    )
    host = MCPHost()

    assert manager.attach_to_host(host) == -1
    assert host.get_tool("add_numbers") is None
    assert host.get_tool("mcp_off_dispatch") is None


def test_attach_returns_minus_one_when_nothing_configured(manager):
    """'nothing configured' (-1) must be distinguishable from 'all failed' (0)."""
    host = MCPHost()
    assert manager.attach_to_host(host) == -1
    assert manager.health() == {}


def test_attach_returns_zero_when_every_server_failed(tmp_path, manager):
    manager.config_path = _write_config(
        tmp_path, {"broken": _echo_server(args=[FIXTURE, "--fail-startup"])}
    )
    host = MCPHost()
    assert manager.attach_to_host(host) == 0


def test_health_reports_state_and_reason(tmp_path, manager):
    manager.config_path = _write_config(
        tmp_path,
        {
            "echo": _echo_server(),
            "broken": _echo_server(args=[FIXTURE, "--fail-startup"]),
            "later": _echo_server(),
        },
    )
    host = MCPHost()
    manager.attach_to_host(host)

    report = manager.health()
    assert report["echo"]["running"] is True
    assert report["echo"]["tools"] == len(EXPECTED_TOOL_NAMES)
    assert report["broken"]["running"] is False
    assert "broken" in report["broken"]["error"]
    # `attach_to_host` attempts every enabled server, so both good servers run.
    assert report["later"]["running"] is True
    assert report["later"]["error"] is None


def test_health_leaves_untouched_servers_unstarted(tmp_path, manager):
    """health() must not start anything: an unconnected server is 'not started'."""
    manager.config_path = _write_config(tmp_path, {"later": _echo_server()})

    report = manager.health()

    assert report["later"]["started"] is False
    assert report["later"]["running"] is False
    assert report["later"]["error"] is None


def test_missing_mcp_package_is_reported_for_http_transport(tmp_path, monkeypatch):
    """(6) HAS_MCP degradation is loud and names the dependency."""
    monkeypatch.setattr("dual_agent.mcp_manager.HAS_MCP", False)
    manager = MCPManager(config_path=str(tmp_path / "mcp_servers.json"))
    try:
        manager.config_path = _write_config(
            tmp_path, {"remote": {"url": "https://example.invalid/mcp"}}
        )
        host = MCPHost()
        manager.attach_to_host(host)

        result = host.execute_tool("mcp_remote_dispatch", {})
        assert str(result.output).startswith("Error:")
        assert "'mcp'" in str(result.output)
        assert "not installed" in str(result.output)
    finally:
        manager.shutdown()


# --------------------------------------------------------------------------
# Config-file compatibility
# --------------------------------------------------------------------------


def test_add_and_remove_server_round_trip(tmp_path):
    config_file = str(tmp_path / "mcp_servers.json")
    mgr = MCPManager(config_path=config_file)
    try:
        assert mgr.load_servers() == {}

        mgr.add_server(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"TOKEN": "xyz"},
        )
        servers = mgr.load_servers()
        assert servers["github"].command == "npx"
        assert servers["github"].args == ["-y", "@modelcontextprotocol/server-github"]
        assert servers["github"].env == {"TOKEN": "xyz"}

        assert mgr.remove_server("github") is True
        assert "github" not in mgr.load_servers()
        assert mgr.remove_server("github") is False
    finally:
        mgr.shutdown()


def test_existing_claude_desktop_config_shape_still_loads(tmp_path):
    """The on-disk format is unchanged: no new schema was invented."""
    config_file = tmp_path / "mcp_servers.json"
    config_file.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "filesystem": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                        "env": {"LOG": "debug"},
                        "disabled": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    mgr = MCPManager(config_path=str(config_file))
    try:
        entry = mgr.load_servers()["filesystem"]
        assert entry.command == "npx"
        assert entry.disabled is True
        assert entry.url is None
    finally:
        mgr.shutdown()


def test_has_mcp_flag_follows_typesafe_pattern():
    """HAS_MCP must be a plain bool import guard, like HAS_TYPESAFE_SDK."""
    assert isinstance(HAS_MCP, bool)


def test_registry_keeps_working_without_external_servers(tmp_path, manager):
    """Built-in tools are unaffected when no MCP server is configured."""
    host = MCPHost()
    manager.attach_to_host(host)
    res = host.execute_tool("list_directory", {"path": str(tmp_path)})
    assert res.success is True
    assert "entries" in json.loads(res.output)


# --------------------------------------------------------------------------
# Concurrency: two servers, two processes, no cross-talk
# --------------------------------------------------------------------------


def test_two_servers_do_not_share_a_process_or_answers(tmp_path, manager):
    manager.config_path = _write_config(
        tmp_path,
        {
            "one": _echo_server(args=[FIXTURE, "--sleep", "0.05"]),
            "two": _echo_server(args=[FIXTURE, "--sleep", "0.05"]),
        },
    )
    host = MCPHost()
    manager.attach_to_host(host)

    results = {}

    def _call(tool, args, key):
        results[key] = host.execute_tool(tool, args).output

    threads = [
        threading.Thread(target=_call, args=("add_numbers", {"a": 100, "b": 1}, "a")),
        threading.Thread(target=_call, args=("reverse_text", {"text": "abc"}, "b")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert results["a"] == "101"
    assert results["b"] == "cba"
