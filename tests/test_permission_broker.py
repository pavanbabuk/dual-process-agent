"""Tests for PermissionBroker (approval card logic)."""

import pytest
from dual_agent.permission_broker import PermissionBroker, ApprovalDecision


@pytest.fixture
def auto_broker():
    """PermissionBroker in auto-allow mode (no interactive prompts)."""
    return PermissionBroker(auto_allow=True)


@pytest.fixture
def broker():
    return PermissionBroker(auto_allow=False)


def test_auto_allow_returns_allow_once(auto_broker):
    decision, args = auto_broker.request_approval(
        tool_name="run_shell_command",
        args={"command": "ls -la"},
        risk_level="high",
    )
    assert decision == ApprovalDecision.ALLOW_ONCE
    assert args == {"command": "ls -la"}


def test_session_allowed_bypasses_prompt(auto_broker):
    # Prime session allow
    auto_broker._session_allowed.add("write_file")
    decision, args = auto_broker.request_approval(
        tool_name="write_file",
        args={"path": "test.py", "content": "hello"},
        risk_level="medium",
    )
    assert decision == ApprovalDecision.ALLOW_ONCE


def test_session_denied_returns_deny(auto_broker):
    auto_broker._session_denied.add("run_shell_command")
    decision, args = auto_broker.request_approval(
        tool_name="run_shell_command",
        args={"command": "rm -rf /"},
        risk_level="high",
    )
    assert decision == ApprovalDecision.DENY


def test_session_allowed_add_and_clear(auto_broker):
    auto_broker._session_allowed.add("write_file")
    assert "write_file" in auto_broker._session_allowed
    auto_broker.reset_session_memory()
    assert "write_file" not in auto_broker._session_allowed


def test_auto_allow_args_passed_through(auto_broker):
    original_args = {"command": "echo hello", "cwd": "/tmp"}
    decision, returned_args = auto_broker.request_approval(
        tool_name="run_shell_command",
        args=original_args,
        risk_level="low",
    )
    assert returned_args == original_args


def test_session_denied_overrides_auto_allow():
    """Even in auto-allow mode, explicit session deny should win."""
    broker = PermissionBroker(auto_allow=True)
    broker._session_denied.add("run_shell_command")
    decision, _ = broker.request_approval("run_shell_command", {}, "high")
    assert decision == ApprovalDecision.DENY
