"""Streaming Dispatcher — wraps DualProcessDispatcher to emit WebSocket events per step.

Instead of blocking until the full run completes, this yields JSON-serialisable
event dicts for every meaningful state change so the browser can render them
in real time.

Event types emitted:
  { "type": "started",    "goal": str }
  { "type": "step",       "step": int, "path": "S1_FAST"|"S2_SLOW"|"TERM",
                          "action": str, "args": dict, "output": str,
                          "latency_ms": float, "tokens_used": int }
  { "type": "permission", "request_id": str, "tool": str,
                          "args": dict, "risk": str }
  { "type": "done",       "is_completed": bool, "final_output": str,
                          "total_steps": int, "system_one_steps": int,
                          "system_two_steps": int, "total_latency_ms": float,
                          "tokens_used": int, "used_simulated_system_one": bool }
  { "type": "error",      "message": str }
"""

from __future__ import annotations
import asyncio
import logging
import uuid
from typing import Dict, Any, Optional

from dual_agent.dispatcher import DualProcessDispatcher
from dual_agent.permission_broker import PermissionBroker, ApprovalDecision

logger = logging.getLogger(__name__)

# Per-connection pending permission requests: {request_id: asyncio.Future}
_pending_approvals: Dict[str, asyncio.Future] = {}


class WebPermissionBroker(PermissionBroker):
    """Permission broker that emits WebSocket events instead of Rich prompts."""

    def __init__(self, send_event, loop: asyncio.AbstractEventLoop):
        super().__init__(auto_allow=False)
        self._send_event = send_event  # async callable(dict)
        self._loop = loop

    def request_approval(
        self,
        tool_name: str,
        args: Dict[str, Any],
        risk_level: str = "high",
    ):
        """Emit a permission event and block until the browser responds."""
        # Session-level shortcuts
        if tool_name in self._session_allowed:
            return ApprovalDecision.ALLOW_ONCE, args
        if tool_name in self._session_denied:
            return ApprovalDecision.DENY, args

        request_id = str(uuid.uuid4())[:8]
        future: asyncio.Future = self._loop.create_future()
        _pending_approvals[request_id] = future

        # Emit permission request event (non-blocking from sync context)
        asyncio.run_coroutine_threadsafe(
            self._send_event({
                "type": "permission",
                "request_id": request_id,
                "tool": tool_name,
                "args": args,
                "risk": risk_level,
            }),
            self._loop,
        )

        # Block (in the executor thread) until the browser responds
        import concurrent.futures
        result_future = concurrent.futures.Future()

        def _cb(f: asyncio.Future):
            try:
                result_future.set_result(f.result())
            except Exception as e:
                result_future.set_exception(e)

        self._loop.call_soon_threadsafe(future.add_done_callback, _cb)
        try:
            decision_str, edited_args = result_future.result(timeout=120)
        except Exception:
            decision_str, edited_args = "deny", args
        finally:
            _pending_approvals.pop(request_id, None)

        # The decision arrives over a WebSocket, so it is untrusted input: a
        # malformed or unrecognised string must not raise out of here and abort
        # the run. Unknown values fall back to deny, which is the safe default
        # for a tool that was awaiting permission.
        try:
            decision = ApprovalDecision(decision_str)
        except ValueError:
            logger.warning(
                f"[WebPermission] Unrecognised approval decision {decision_str!r}; denying."
            )
            decision = ApprovalDecision.DENY

        if decision == ApprovalDecision.ALLOW_SESSION:
            self._session_allowed.add(tool_name)
        return decision, edited_args


async def resolve_permission(request_id: str, decision: str, edited_args: Optional[Dict] = None):
    """Called from the WebSocket handler when the browser sends a permission_response."""
    future = _pending_approvals.get(request_id)
    if future and not future.done():
        future.get_event_loop().call_soon_threadsafe(
            future.set_result, (decision, edited_args or {})
        )


class _HookedList(list):
    """list subclass that fires a callback on every append (Python 3.14 safe)."""
    _callback = None  # set after construction

    def append(self, item):
        super().append(item)
        if self._callback is not None:
            try:
                self._callback(item)
            except Exception:
                pass


class StreamingDispatcher:
    """Runs the DualProcessDispatcher in an executor and yields events as they happen."""

    def __init__(self, dispatcher: DualProcessDispatcher, send_event):
        self.dispatcher = dispatcher
        self._send_event = send_event

    async def run_streaming(self, goal: str, max_steps: int = 15) -> None:
        """Execute goal and stream events via send_event. Non-blocking."""
        await self._send_event({"type": "started", "goal": goal})

        loop = asyncio.get_running_loop()

        # Replace broker with a WebSocket-aware one
        web_broker = WebPermissionBroker(send_event=self._send_event, loop=loop)
        web_broker._session_allowed = self.dispatcher.broker._session_allowed.copy()
        web_broker._session_denied = self.dispatcher.broker._session_denied.copy()

        original_broker = self.dispatcher.broker
        self.dispatcher.broker = web_broker

        try:
            result = await loop.run_in_executor(
                None,
                lambda: self._run_with_hooks(goal, max_steps, loop)
            )

            await self._send_event({
                "type": "done",
                "is_completed": result.is_completed,
                "final_output": result.final_output or "Completed.",
                "total_steps": result.total_steps,
                "system_one_steps": result.system_one_steps,
                "system_two_steps": result.system_two_steps,
                "total_latency_ms": round(result.total_latency_ms, 1),
                "tokens_used": result.tokens_used,
                "used_simulated_system_one": result.used_simulated_system_one,
            })
        except Exception as e:
            logger.exception(f"[StreamingDispatcher] Error: {e}")
            await self._send_event({"type": "error", "message": str(e)})
        finally:
            self.dispatcher.broker = original_broker

    def _run_with_hooks(self, goal: str, max_steps: int, loop: asyncio.AbstractEventLoop):
        """Run dispatcher and emit step events by temporarily patching mcp.execute_tool."""
        import time

        path_map = {
            "S1_FAST":  "S1_FAST",
            "S2_SLOW":  "S2_SLOW",
            "TERMINAL": "TERMINAL",
        }

        # Wrap mcp.execute_tool to emit a step event after each real tool call
        original_execute = self.dispatcher.mcp.execute_tool
        step_counter = [0]

        def patched_execute(name: str, arguments: dict):
            t0 = time.perf_counter()
            result = original_execute(name, arguments)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            step_counter[0] += 1
            asyncio.run_coroutine_threadsafe(
                self._send_event({
                    "type": "step",
                    "step": step_counter[0],
                    "path": "S1_FAST",
                    "action": name,
                    "args": arguments,
                    "output": str(result.output if hasattr(result, "output") else result)[:400],
                    "latency_ms": latency_ms,
                    "tokens_used": 0,
                }),
                loop,
            )
            return result

        self.dispatcher.mcp.execute_tool = patched_execute
        try:
            return self.dispatcher.run(goal=goal, max_steps=max_steps)
        finally:
            self.dispatcher.mcp.execute_tool = original_execute

