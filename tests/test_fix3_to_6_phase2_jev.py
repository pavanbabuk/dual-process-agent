"""Tests for Phase 2: Fixes 3, 4, 5, and 6 (Jev Vendor Spec Alignment)."""

import pytest
from unittest.mock import MagicMock
from typesafe_sdk import ChoiceAnswer, NoulAnswer
from dual_agent.typesafe_client import (
    JevSystemOneClient,
    JevDecision,
    THRESHOLD_NEEDS_WRITING,
    THRESHOLD_NEEDS_CODE,
    THRESHOLD_OPTION_PROBABILITY,
    THRESHOLD_DISTRIBUTION_CONCENTRATION,
)
from dual_agent.dispatcher import DualProcessDispatcher, validate_tool_args
from dual_agent.mcp_host import MCPHost, MCPToolDefinition
from dual_agent.permission_broker import PermissionBroker
from dual_agent.system_two import SystemTwoResponse


def test_fix3_split_compound_questions_independently_falsifiable():
    """Writing flag and code flag must be independently falsifiable with stub responses."""
    client = JevSystemOneClient(force_simulation=False, api_key="dummy-key")
    mock_sdk = MagicMock()
    client._sdk_client = mock_sdk

    # 1. Writing flag alone trips needs_generation
    mock_sdk.system_one.return_value = MagicMock(
        choices={
            "route": ChoiceAnswer(choice="write_file", confidence=0.95, probabilities={"write_file": 0.95}),
            "evidence_sufficient": ChoiceAnswer(choice="sufficient", confidence=0.99, probabilities={"sufficient": 0.99}),
        },
        nouls={
            "goal_satisfied": NoulAnswer(noul=0.10),
            "needs_writing": NoulAnswer(noul=0.90),  # High writing
            "needs_code": NoulAnswer(noul=0.10),     # Low code
            "needs_explanation": NoulAnswer(noul=0.10),
        },
    )
    decision1 = client.evaluate_state_and_route("test state", {"write_file": "desc"})
    assert decision1.needs_writing is True
    assert decision1.needs_code is False
    assert decision1.needs_generation is True

    # 2. Code flag alone trips needs_generation
    mock_sdk.system_one.return_value = MagicMock(
        choices={
            "route": ChoiceAnswer(choice="write_file", confidence=0.95, probabilities={"write_file": 0.95}),
            "evidence_sufficient": ChoiceAnswer(choice="sufficient", confidence=0.99, probabilities={"sufficient": 0.99}),
        },
        nouls={
            "goal_satisfied": NoulAnswer(noul=0.10),
            "needs_writing": NoulAnswer(noul=0.10),  # Low writing
            "needs_code": NoulAnswer(noul=0.90),     # High code
            "needs_explanation": NoulAnswer(noul=0.10),
        },
    )
    decision2 = client.evaluate_state_and_route("test state", {"write_file": "desc"})
    assert decision2.needs_writing is False
    assert decision2.needs_code is True
    assert decision2.needs_generation is True


def test_fix4_never_invent_confidence_and_flat_distribution_gate():
    """Confidence values are not fabricated; flat distribution does not clear the gate."""
    client = JevSystemOneClient(force_simulation=False, api_key="dummy-key")
    mock_sdk = MagicMock()
    client._sdk_client = mock_sdk

    # 1. Missing probabilities must escalate, never fabricate {selected: confidence}
    mock_sdk.system_one.return_value = MagicMock(
        choices={
            "route": MagicMock(choice="list_directory", confidence=0.85, probabilities=None),
            "evidence_sufficient": ChoiceAnswer(choice="sufficient", confidence=0.90, probabilities={"sufficient": 0.90}),
        },
        nouls={
            "goal_satisfied": NoulAnswer(noul=0.10),
            "needs_writing": NoulAnswer(noul=0.10),
            "needs_code": NoulAnswer(noul=0.10),
            "needs_explanation": NoulAnswer(noul=0.10),
        },
    )
    decision = client.evaluate_state_and_route("state", {"list_directory": "desc"})
    assert decision.needs_generation is True, "Missing probabilities must escalate to System 2"
    assert decision.probabilities == {}

    # 2. Flat distribution (all 0.25) must not clear the two-condition gate
    mock_sdk.system_one.return_value = MagicMock(
        choices={
            "route": ChoiceAnswer(
                choice="list_directory",
                confidence=0.25,
                probabilities={"list_directory": 0.25, "read_file": 0.25, "write_file": 0.25, "finish_task": 0.25},
            ),
            "evidence_sufficient": ChoiceAnswer(choice="sufficient", confidence=0.90, probabilities={"sufficient": 0.90}),
        },
        nouls={
            "goal_satisfied": NoulAnswer(noul=0.10),
            "needs_writing": NoulAnswer(noul=0.10),
            "needs_code": NoulAnswer(noul=0.10),
            "needs_explanation": NoulAnswer(noul=0.10),
        },
    )
    flat_decision = client.evaluate_state_and_route("state", {"list_directory": "desc"})
    assert flat_decision.needs_generation is True, "Flat distribution must not clear the gate"


def test_fix5_per_action_thresholds():
    """Same confidence score clears the gate for read-only tool but not destructive tool."""
    mcp = MCPHost()
    dispatcher = DualProcessDispatcher(
        mcp_host=mcp,
        confidence_threshold=0.85,
    )

    read_thresh = dispatcher._get_tool_confidence_threshold("list_directory")
    write_thresh = dispatcher._get_tool_confidence_threshold("write_file")

    # Read-only tool has a lower bar, destructive has a higher bar
    assert read_thresh < write_thresh
    assert write_thresh >= 0.88

    # Score of 0.80 should clear read-only but fail destructive
    assert 0.80 >= read_thresh
    assert 0.80 < write_thresh


def test_fix6_args_validation_refused_on_both_paths():
    """Out-of-schema arguments are refused on both fast-path and slow-path."""
    mcp = MCPHost()
    tool = mcp.get_tool("read_file")

    # Fast-path validation
    ok, err = validate_tool_args(tool, {"path": "test.txt", "hallucinated_key": "bad"})
    assert ok is False
    assert "unknown argument 'hallucinated_key'" in err

    # Slow-path: System 2 returns out-of-schema args
    s2 = MagicMock()
    s2.is_mock = False
    s2.generate_step.side_effect = [
        # Planning phase
        SystemTwoResponse(
            thought="Plan",
            action="plan_subgoals",
            args={"subgoals": ["read file"]},
        ),
        # Step 1: out-of-schema args
        SystemTwoResponse(
            thought="Executing with bad args",
            action="read_file",
            args={"path": "test.txt", "unknown_arg": 123},
        ),
        # Step 2: conclude
        SystemTwoResponse(
            thought="Conclude",
            action="finish_task",
            args={"result": "done"},
        ),
    ]

    dispatcher = DualProcessDispatcher(
        system_two_provider=s2,
        mcp_host=mcp,
        permission_broker=PermissionBroker(auto_allow=True),
        confidence_threshold=0.99,  # force slow-path
    )

    steps = []
    result = dispatcher.run(goal="read file", max_steps=3, step_callback=lambda s: steps.append(s))
    # The first step had invalid args, so it was rejected and recorded
    rejected_steps = [s for s in steps if "Argument validation failed" in str(s.output)]
    assert len(rejected_steps) >= 1
