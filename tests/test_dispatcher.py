"""Tests for Dual-Process Dispatcher and routing logic."""

import pytest
from dual_agent.dispatcher import DualProcessDispatcher, DispatchResult
from dual_agent.typesafe_client import JevSystemOneClient
from dual_agent.system_two import MockSystemTwoProvider
from dual_agent.mcp_host import MCPHost


def test_dispatcher_fast_path_and_completion():
    s1 = JevSystemOneClient(force_simulation=True)
    s2 = MockSystemTwoProvider()
    mcp = MCPHost()

    dispatcher = DualProcessDispatcher(
        system_one_client=s1,
        system_two_provider=s2,
        mcp_host=mcp,
        confidence_threshold=0.80,
    )

    result: DispatchResult = dispatcher.run(
        goal="Inspect current directory and list files.",
        max_steps=5,
    )

    assert isinstance(result, DispatchResult)
    assert result.total_steps > 0
    assert result.system_one_latency_ms >= 0
