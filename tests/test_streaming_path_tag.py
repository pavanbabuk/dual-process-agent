"""Regression test for Fix 6: WebSocket step events must tag the real path taken (S2_SLOW when escalated)."""

import asyncio
from unittest.mock import MagicMock

from dual_agent.dispatcher import DualProcessDispatcher
from dual_agent.typesafe_client import JevDecision
from dual_agent.system_two import SystemTwoProvider, SystemTwoResponse
from dual_agent.mcp_host import MCPHost
from dual_agent.web.streaming_dispatcher import StreamingDispatcher


def test_streaming_dispatcher_tags_s2_slow_on_escalation(tmp_path):
    async def _test():
        # S1 escalates to S2
        s1 = MagicMock()
        s1.evaluate_state_and_route.return_value = JevDecision(
            is_terminal=False,
            selected_tool="list_directory",
            confidence=0.20,  # Below threshold -> escalates to S2
            latency_ms=12.0,
            needs_generation=True,
        )
        s1.force_simulation = False
        s1.simulation_reason = None

        # S2 produces a tool call then finishes
        step_call_count = [0]
        s2 = MagicMock()
        def s2_generate(prompt):
            step_call_count[0] += 1
            if step_call_count[0] == 1:
                return SystemTwoResponse(
                    action="list_directory",
                    arguments={"path": "."},
                    thought="I need to list directory",
                    latency_ms=250.0,
                    tokens_used=120,
                    is_mock=False,
                )
            return SystemTwoResponse(
                action="finish_task",
                arguments={},
                thought="finishing",
                generated_content="Completed listing directory.",
                latency_ms=100.0,
                tokens_used=50,
                is_mock=False,
            )
        s2.generate_step.side_effect = s2_generate

        mcp = MCPHost()
        dispatcher = DualProcessDispatcher(
            system_one_client=s1,
            system_two_provider=s2,
            mcp_host=mcp,
            confidence_threshold=0.85,
        )

        events = []
        async def capture_event(event):
            events.append(event)

        streaming = StreamingDispatcher(dispatcher=dispatcher, send_event=capture_event)
        await streaming.run_streaming("list directory contents", max_steps=5)

        step_events = [e for e in events if e.get("type") == "step"]
        assert len(step_events) >= 1, f"Expected step events, got {events}"
        
        # At least one event must be tagged S2_SLOW because it was executed via System 2!
        slow_events = [e for e in step_events if e.get("path") == "S2_SLOW"]
        assert len(slow_events) >= 1, f"Expected at least one 'S2_SLOW' event, but all step events were: {step_events}"

    asyncio.run(_test())
