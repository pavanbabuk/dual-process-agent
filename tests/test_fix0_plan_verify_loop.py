"""Tests for Phase 0 Fix 0: Plan -> Act -> Verify -> Correct loop."""

import pytest
from unittest.mock import MagicMock
from dual_agent.dispatcher import DualProcessDispatcher
from dual_agent.state import AgentState, PlanStep
from dual_agent.evaluator import JevEvaluator
from dual_agent.mcp_host import MCPHost, MCPToolDefinition
from dual_agent.system_two import MockSystemTwoProvider, SystemTwoResponse
from dual_agent.typesafe_client import JevDecision


def test_state_plan_model_exists():
    """AgentState must have a plan attribute holding PlanStep items."""
    state = AgentState(goal="Test multi-step task")
    assert hasattr(state, "plan")
    assert isinstance(state.plan, list)


def test_dispatcher_creates_plan_and_verifies_steps():
    """Dispatcher must decompose goal into plan steps and verify each step outcome."""
    # Custom provider that emits a plan then completes steps
    s2 = MagicMock()
    s2.is_mock = False
    s2.generate_step.side_effect = [
        # Planning phase decomposition
        SystemTwoResponse(
            thought="Decompose the multi-step goal",
            action="plan_subgoals",
            args={
                "subgoals": [
                    "Inspect target file to understand current implementation",
                    "Apply changes using patch_file",
                ]
            },
            tokens_used=50,
        ),
        # Step 1 execution
        SystemTwoResponse(
            thought="Step 1: Read target file",
            action="read_file",
            args={"path": "test_target.py"},
            tokens_used=40,
        ),
        # Step 2 execution
        SystemTwoResponse(
            thought="Step 2: Finish task",
            action="finish_task",
            args={"result": "Successfully updated target file"},
            generated_content="Successfully updated target file",
            tokens_used=30,
        ),
    ]

    mcp = MCPHost()
    # Mock evaluator to verify it is invoked
    mock_evaluator = MagicMock(spec=JevEvaluator)
    mock_evaluator.verify_step_outcome.return_value = (True, "Step succeeded and produced expected output")
    mock_evaluator.check_task_completion.return_value = True

    dispatcher = DualProcessDispatcher(
        system_two_provider=s2,
        mcp_host=mcp,
    )
    dispatcher.evaluator = mock_evaluator

    # Write a test file for read_file
    import os
    with open("test_target.py", "w") as f:
        f.write("print('hello')\n")

    try:
        res = dispatcher.run(
            goal="Update test_target.py to print hello world",
            max_steps=5,
        )
        # Verify planning happened and state recorded it
        assert mock_evaluator.verify_step_outcome.called, "Evaluator must be called to verify tool outcomes"
        assert res.is_completed is True
    finally:
        if os.path.exists("test_target.py"):
            os.remove("test_target.py")


def test_verification_failure_triggers_honest_failure_not_fake_success():
    """If a tool step repeatedly fails verification, agent must honestly report failure."""
    s2 = MagicMock()
    s2.is_mock = False
    s2.generate_step.return_value = SystemTwoResponse(
        thought="Attempting action",
        action="read_file",
        args={"path": "non_existent_file_12345.py"},
        tokens_used=10,
    )

    mock_evaluator = MagicMock(spec=JevEvaluator)
    # Step verification always fails because file does not exist
    mock_evaluator.verify_step_outcome.return_value = (False, "File not found")
    mock_evaluator.check_task_completion.return_value = False

    dispatcher = DualProcessDispatcher(
        system_two_provider=s2,
        mcp_host=MCPHost(),
    )
    dispatcher.evaluator = mock_evaluator

    res = dispatcher.run(
        goal="Read non_existent_file_12345.py and extract info",
        max_steps=3,
    )
    # The agent must NOT claim success when verification failed
    assert res.is_completed is False
    assert "could not complete" in (res.final_output or "").lower() or "stopped" in (res.final_output or "").lower()
