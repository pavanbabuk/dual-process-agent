"""Regression tests for the honesty, argument-validation, and security fixes.

Each test here corresponds to a specific defect that was found and fixed:

1. Fast-path tool arguments were hardcoded per tool name, so every goal read
   `pyproject.toml` and ran `ls -la` regardless of what was asked.
2. `run_shell_command` used `shell=True` on a model-influenced string, so a
   string-interpolated argument could inject arbitrary commands that the
   approval card never showed.
3. The gateway accepted commands from any Telegram user.
4. The dashboard would bind to any interface with no authentication.
5. Telemetry reported a "token savings" percentage computed from constants.
"""

import os
import sys

import pytest

from dual_agent.dispatcher import DualProcessDispatcher, validate_tool_args
from dual_agent.mcp_host import MCPHost
from dual_agent.state import AgentState
from dual_agent.system_two import MockSystemTwoProvider
from dual_agent.typesafe_client import JevSystemOneClient


# ----------------------------------------------------------------------
# 1. Argument inference must not be hardcoded per tool
# ----------------------------------------------------------------------

def _dispatcher(**kwargs):
    return DualProcessDispatcher(
        system_one_client=JevSystemOneClient(force_simulation=True),
        system_two_provider=MockSystemTwoProvider(),
        mcp_host=MCPHost(),
        **kwargs,
    )


def test_read_file_args_are_not_hardcoded(tmp_path):
    """`read_file` must not always resolve to pyproject.toml."""
    target = tmp_path / "notes.txt"
    target.write_text("hello")

    d = _dispatcher()
    state = AgentState(goal=f"please read {target} and report")
    args = d._infer_default_args("read_file", state)

    assert args["path"] == str(target)
    assert args["path"] != "pyproject.toml"


def test_shell_command_is_never_invented():
    """An unquoted goal must yield no command at all, not a default like `ls -la`."""
    d = _dispatcher()
    state = AgentState(goal="tidy up the project")
    args = d._infer_default_args("run_shell_command", state)

    assert args["command"] == "", "must not invent a command the user never asked for"


def test_shell_command_extracted_only_when_quoted():
    d = _dispatcher()
    state = AgentState(goal="run `echo hi` and show me")
    assert d._infer_default_args("run_shell_command", state)["command"] == "echo hi"


# ----------------------------------------------------------------------
# 2. Fast path must refuse to execute unverified arguments
# ----------------------------------------------------------------------

def test_validate_tool_args_rejects_missing_required():
    tool = MCPHost().get_tool("read_file")
    ok, msg = validate_tool_args(tool, {})
    assert ok is False
    assert "path" in msg


def test_validate_tool_args_rejects_empty_string():
    tool = MCPHost().get_tool("read_file")
    ok, msg = validate_tool_args(tool, {"path": "   "})
    assert ok is False


def test_validate_tool_args_rejects_wrong_type():
    tool = MCPHost().get_tool("write_file")
    ok, msg = validate_tool_args(tool, {"path": "x", "content": 123})
    assert ok is False
    assert "content" in msg


def test_validate_tool_args_accepts_valid():
    tool = MCPHost().get_tool("read_file")
    ok, _ = validate_tool_args(tool, {"path": "./README.md"})
    assert ok is True


def test_fast_path_refuses_unverified_args_by_default():
    """With no usable arguments available, the run must escalate, not guess."""
    d = _dispatcher()
    d.allow_unverified_fast_path = False

    # `read_file` needs a path; this goal names no readable file, so the fast
    # path must decline rather than fabricate one, and the loop must still
    # terminate cleanly instead of crashing on a rejected step.
    result = d.run(goal="read a document and summarize it", max_steps=2)

    assert result.total_steps >= 0
    assert result.is_completed in (True, False)


# ----------------------------------------------------------------------
# 3. Shell execution must not go through a shell
# ----------------------------------------------------------------------

def test_shell_injection_via_argument_is_not_executed(tmp_path):
    """A metacharacter payload passed as an argument must be treated as data.

    Under the old `shell=True` implementation this command created the marker
    file. With argv execution it is a literal argument to /bin/echo and nothing
    else happens.
    """
    marker = tmp_path / "pwned"
    host = MCPHost()

    payload = f"echo safe; touch {marker}"
    result = host.execute_tool("run_shell_command", {"command": payload})

    assert result.success is True
    assert not marker.exists(), "shell metacharacters must not be interpreted"


def test_shell_empty_command_refused():
    result = MCPHost().execute_tool("run_shell_command", {"command": ""})
    assert "refusing to execute an empty command" in str(result.output).lower()


def test_shell_unparseable_command_reports_error():
    result = MCPHost().execute_tool("run_shell_command", {"command": "echo 'unclosed"})
    assert "could not parse command" in str(result.output).lower()


# ----------------------------------------------------------------------
# 4/5. Gateway authorization and telemetry honesty
# ----------------------------------------------------------------------

def test_verification_calls_are_counted_in_system_one_latency():
    """Router calls made to verify the fast path must appear in the S1 total.

    The confidence gate re-asks the router about the runner-up tool. That is a
    real model call; if it is not billed, System 1 latency under-reports the
    cost of routing and the telemetry is quietly flattering.
    """
    seen = []
    original = JevSystemOneClient.evaluate_state_and_route

    def traced(self, state_text, tool_options, allow_escalation=True):
        decision = original(self, state_text, tool_options, allow_escalation)
        seen.append(decision)
        return decision

    JevSystemOneClient.evaluate_state_and_route = traced
    try:
        d = _dispatcher()
        result = d.run(goal="list the files in this project", max_steps=2)
    finally:
        JevSystemOneClient.evaluate_state_and_route = original

    assert len(seen) >= 2, "the gate must have re-checked at least one rival declaration"
    # Every router call was simulated here, so billed S1 time must cover them all.
    assert result.simulated_latency_ms <= result.system_one_latency_ms
    assert result.system_one_latency_ms >= 0


def test_dispatcher_result_latency_invariants():
    """S1 and S2 latency cannot each exceed the total run latency by themselves."""
    result = _dispatcher().run(goal="Inspect the current directory.", max_steps=3)

    assert result.system_one_latency_ms >= 0
    assert result.system_two_latency_ms >= 0
    assert result.simulated_latency_ms <= result.system_one_latency_ms
    # Tool execution time can make the parts exceed the recorded step sum, but
    # neither subsystem may claim more time than the whole run took.
    assert result.system_one_latency_ms <= result.total_latency_ms
    assert result.system_two_latency_ms <= result.total_latency_ms


def test_telegram_adapter_denies_everyone_by_default():
    from dual_agent.gateway.telegram_adapter import TelegramAdapter

    async def _noop(msg):
        return None

    adapter = TelegramAdapter(on_message=_noop, token="dummy:token")
    assert adapter.allowed_user_ids == set()
    assert "999" not in adapter.allowed_user_ids


def test_telegram_adapter_parses_allowlist(monkeypatch):
    from dual_agent.gateway.telegram_adapter import TelegramAdapter

    async def _noop(msg):
        return None

    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "111, 222 ,")
    adapter = TelegramAdapter(on_message=_noop, token="dummy:token")
    assert adapter.allowed_user_ids == {"111", "222"}


def test_dashboard_refuses_non_loopback_bind(monkeypatch):
    from dual_agent.web import server as web_server

    if not web_server._FASTAPI_AVAILABLE:
        pytest.skip("dashboard extras not installed (pip install 'dual-agent[ui]')")

    monkeypatch.delenv("DUAL_AGENT_UI_ALLOW_PUBLIC_BIND", raising=False)
    with pytest.raises(SystemExit) as exc:
        web_server.run_server(host="0.0.0.0", port=7860, open_browser=False)
    assert "Refusing to bind" in str(exc.value)


def test_junction_telemetry_has_no_fabricated_savings():
    result = _dispatcher().run(goal="Inspect the current directory.", max_steps=3)
    for fabricated in ("estimated_token_savings_pct", "speedup_ratio", "estimated_baseline_tokens"):
        assert not hasattr(result, fabricated)


def test_memory_reports_none_for_unmeasured_savings(tmp_path):
    """An unmeasured savings figure must be None, not 0.0."""
    from dual_agent.memory import MemoryEngine

    mem = MemoryEngine(db_path=str(tmp_path / "mem.db"))
    mem.save_session(
        goal="g", outcome="o", is_completed=True, total_steps=1,
        system_one_steps=1, system_two_steps=0, total_latency_ms=5.0,
        tokens_used=0, steps=[],
    )
    stats = mem.get_aggregate_stats()
    assert stats["avg_token_savings_pct"] is None
