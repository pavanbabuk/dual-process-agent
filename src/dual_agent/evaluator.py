"""Fast-path evaluation, guardrails, and termination checks using Jev (System 1)."""

from __future__ import annotations
import os
import logging
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from dual_agent.typesafe_client import JevSystemOneClient
from dual_agent.state import AgentState

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    is_valid: bool = True
    score: int = 5
    confidence: float = 1.0
    reason: str = ""
    latency_ms: float = 0.0


class JevEvaluator:
    """Evaluates agent outputs and state using Jev's Noul and Score primitives."""

    def __init__(self, client: Optional[JevSystemOneClient] = None):
        self.client = client or JevSystemOneClient()

    def verify_step_outcome(
        self,
        subgoal: str,
        action: str,
        action_input: Dict[str, Any],
        output: Any,
    ) -> Tuple[bool, str]:
        """Verify whether a tool action produced its intended effect and satisfied the subgoal."""
        if output is None:
            return False, f"Action '{action}' produced null output."

        out_str = str(output).strip()
        if not out_str and action not in ("run_shell_command", "finish_task"):
            return False, f"Action '{action}' produced empty output."

        # Explicit error strings returned by tools
        lower_out = out_str.lower()
        if out_str.startswith("Error:") or "[denied" in lower_out or "not registered in mcp" in lower_out:
            return False, f"Action '{action}' failed: {out_str[:300]}"

        # Filesystem checks
        if action in ("write_file", "patch_file"):
            target_path = action_input.get("path", "")
            if target_path and not os.path.exists(target_path):
                return False, f"Target file '{target_path}' does not exist after {action}."

            if target_path and target_path.endswith(".py") and os.path.exists(target_path):
                from dual_agent.mcp_host import validate_python_syntax
                try:
                    with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                        file_body = f.read()
                    syntax_err = validate_python_syntax(target_path, file_body)
                    if syntax_err:
                        return False, f"Python syntax check failed on '{target_path}': {syntax_err}"
                except Exception as e:
                    return False, f"Could not read '{target_path}' for syntax verification: {e}"

        elif action == "read_file":
            target_path = action_input.get("path", "")
            if target_path and not os.path.exists(target_path):
                return False, f"File '{target_path}' not found."

        elif action == "run_shell_command":
            if "error:" in lower_out or "command not found" in lower_out or "traceback (most recent call last)" in lower_out:
                return False, f"Shell command indicated error: {out_str[:300]}"

        elif action in ("mouse_click", "key_press", "mouse_move"):
            if "permission required" in lower_out or "kill switch" in lower_out or "out of bounds" in lower_out:
                return False, f"Screen actuation blocked: {out_str[:300]}"

        elif action == "screen_diff":
            try:
                import json
                diff_data = json.loads(out_str)
                if isinstance(diff_data, dict) and not diff_data.get("changed", True):
                    return False, f"Visual verification failed: {diff_data.get('details', 'No change detected.')}"
            except Exception:
                pass

        return True, f"Action '{action}' completed successfully."

    def check_task_completion(self, state: AgentState) -> bool:
        """Determines if the agent has fulfilled the user's primary goal."""
        state_summary = state.to_system_one_state()
        decision = self.client.evaluate_state_and_route(
            state_text=state_summary,
            tool_options={"status_check": "Check task status"},
        )
        return decision.is_terminal

    def score_output(self, content: str, rubric: str = "Accuracy and relevance to user request") -> int:
        """Assigns an ordinal quality score (1 to 5) to text/code output in ~10ms."""
        return self.client.evaluate_output_score(content, rubric)
