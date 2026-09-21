"""Tests for TypeSafe AI Jev System 1 client."""

import pytest
from dual_agent.typesafe_client import JevSystemOneClient, JevDecision


def test_jev_client_simulation_routing():
    client = JevSystemOneClient(force_simulation=True)
    decision: JevDecision = client.evaluate_state_and_route(
        state_text="Current task: Inspect directory and list files.",
        tool_options={"list_directory": "List files in directory", "write_file": "Write file"},
    )

    assert isinstance(decision, JevDecision)
    assert decision.selected_tool in ("list_directory", "write_file", "escalate_to_system_two", "finish_task")
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.latency_ms > 0.0
    assert decision.simulated is True


def test_jev_client_score():
    client = JevSystemOneClient(force_simulation=True)
    score = client.evaluate_output_score("def add(a, b): return a + b", "Python code correctness")
    assert 1 <= score <= 5
