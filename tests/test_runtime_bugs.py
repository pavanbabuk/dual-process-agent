"""Regression tests for bugs found by actually running the project.

Each of these was discovered by executing the documented workflow, not by
reading code:

1. `dual-agent --gateway` — documented in the README, install.sh, the shell
   help text and the scheduler docstring — had no CLI handler and died with
   "unrecognized arguments". The whole gateway package (and the scheduler, which
   needs a long-lived process) was therefore unreachable.
2. Nothing ever read a `.env` file, yet the README and install.sh both instruct
   users to put credentials there. Following the docs left the agent in
   simulation mode with no error at all.
3. A goal the router could not terminate on spun until max_steps, repeating the
   same tool call, burning a paid API call per step to produce nothing.
4. The permission broker prompts on stdin, which cannot be answered in a
   server-side gateway process.
"""

import asyncio
import os
import subprocess
import sys

import pytest

from dual_agent.config import load_env_files
from dual_agent.dispatcher import DualProcessDispatcher
from dual_agent.mcp_host import MCPHost
from dual_agent.permission_broker import ApprovalDecision, PermissionBroker
from dual_agent.system_two import MockSystemTwoProvider
from dual_agent.typesafe_client import JevSystemOneClient


def _dispatcher(**kw):
    return DualProcessDispatcher(
        system_one_client=JevSystemOneClient(force_simulation=True),
        system_two_provider=MockSystemTwoProvider(),
        mcp_host=MCPHost(),
        **kw,
    )


# ----------------------------------------------------------------------
# 1. The documented --gateway command must exist
# ----------------------------------------------------------------------

def test_gateway_flag_is_recognized_by_cli():
    """`--gateway` must not be an argparse error any more."""
    proc = subprocess.run(
        [sys.executable, "-m", "dual_agent.cli", "--gateway", "--no-such-flag-probe"],
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "DUAL_AGENT_HOME": "/tmp/da_gateway_probe"},
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "unrecognized arguments: --gateway" not in combined


def test_gateway_runner_is_importable_and_builds_a_dispatcher(tmp_path):
    """The runner must construct the full stack without a Telegram token."""
    from dual_agent.gateway.runner import GatewayRunner

    runner = GatewayRunner(session_root=str(tmp_path / "sessions"))
    dispatcher = runner.build_dispatcher(session_id="chat-42")

    assert isinstance(dispatcher, DualProcessDispatcher)
    # Per-chat isolation: each session gets its own memory database.
    assert (tmp_path / "sessions" / "chat-42").is_dir()
    assert runner.scheduler is not None


def test_gateway_router_factory_accepts_session_id_kwarg(tmp_path):
    """SessionRouter calls the factory as factory(session_id=...).

    A factory that only accepted positional args would raise TypeError on the
    first inbound message.
    """
    from dual_agent.gateway.runner import GatewayRunner

    runner = GatewayRunner(session_root=str(tmp_path / "s"))
    router = runner.router
    d = router.get_or_create("chat-7")
    assert d is router.get_or_create("chat-7")  # cached per chat
    assert router.active_sessions == 1


def test_gateway_reply_discloses_simulation():
    """A chat reply must not present simulated routing as real Jev work."""
    from dual_agent.gateway.runner import GatewayRunner

    result = _dispatcher().run(goal="Inspect the current directory.", max_steps=2)
    reply = GatewayRunner.format_reply(result)

    assert "SIMULATED" in reply
    assert "Jev" in reply


def test_gateway_denies_risky_tools_without_a_terminal(tmp_path):
    """No TTY behind a chat => deny, never block on stdin."""
    broker = PermissionBroker(non_interactive=True, auto_allow=False)
    decision, _ = broker.request_approval(
        tool_name="run_shell_command", args={"command": "rm -rf /"}, risk_level="high"
    )
    assert decision == ApprovalDecision.DENY


def test_gateway_allows_risky_tools_when_explicitly_opted_in():
    broker = PermissionBroker(non_interactive=True, auto_allow=True)
    decision, _ = broker.request_approval(
        tool_name="write_file", args={"path": "x", "content": "y"}, risk_level="medium"
    )
    assert decision == ApprovalDecision.ALLOW_ONCE


# ----------------------------------------------------------------------
# 2. .env must actually be read
# ----------------------------------------------------------------------

def test_env_file_is_loaded(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        "DUAL_AGENT_TEST_MARKER=from-dot-env\n"
        'QUOTED_MARKER="quoted value"\n'
        "EMPTY_LINE_OK=1\n"
    )
    monkeypatch.delenv("DUAL_AGENT_TEST_MARKER", raising=False)
    monkeypatch.delenv("QUOTED_MARKER", raising=False)
    monkeypatch.chdir(tmp_path)

    loaded = load_env_files()

    assert str(env_file) in loaded
    assert os.environ["DUAL_AGENT_TEST_MARKER"] == "from-dot-env"
    assert os.environ["QUOTED_MARKER"] == "quoted value"


def test_real_env_var_beats_env_file(tmp_path, monkeypatch):
    """An explicitly exported key must never be clobbered by a stale .env."""
    (tmp_path / ".env").write_text("DUAL_AGENT_TEST_MARKER=from-file\n")
    monkeypatch.setenv("DUAL_AGENT_TEST_MARKER", "from-real-env")
    monkeypatch.chdir(tmp_path)

    load_env_files()

    assert os.environ["DUAL_AGENT_TEST_MARKER"] == "from-real-env"


def test_load_env_files_survives_missing_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .env anywhere
    assert isinstance(load_env_files(), list)


# ----------------------------------------------------------------------
# 3. No-progress loop must terminate
# ----------------------------------------------------------------------

def test_stalled_goal_stops_before_max_steps():
    """A goal the router can't resolve must not burn every allowed step."""
    dispatcher = _dispatcher()
    result = dispatcher.run(goal="export the agent config", max_steps=15)

    assert result.total_steps < 15, (
        "run should have stopped on stall detection, not exhausted max_steps"
    )
    assert result.is_completed is True
    assert "not progressing" in (result.final_output or "")


def test_stall_detection_does_not_trip_on_varied_work():
    """Distinct outputs must not be mistaken for a stall."""
    dispatcher = _dispatcher()
    result = dispatcher.run(goal="Inspect the current directory.", max_steps=3)
    if result.final_output:
        assert "not progressing" not in result.final_output or result.total_steps >= 3


def test_stall_reported_without_fabricated_metrics():
    result = _dispatcher().run(goal="export the agent config", max_steps=15)
    for fabricated in ("estimated_token_savings_pct", "speedup_ratio"):
        assert not hasattr(result, fabricated)
