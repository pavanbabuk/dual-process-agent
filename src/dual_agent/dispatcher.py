"""Dual-Process Dispatcher and Execution Engine."""

from __future__ import annotations
import os
import time
import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from dual_agent.state import AgentState, StepRecord, StepType
from dual_agent.typesafe_client import JevSystemOneClient, JevDecision
from dual_agent.mcp_host import MCPHost, ToolExecutionResult
from dual_agent.system_two import SystemTwoProvider, get_system_two_provider
from dual_agent.memory import MemoryEngine, LearnedSkill

logger = logging.getLogger(__name__)


class DispatchResult(BaseModel):
    """Complete summary of a dual-process execution run."""
    goal: str
    is_completed: bool
    final_output: Optional[str]
    total_steps: int
    system_one_steps: int
    system_two_steps: int
    total_latency_ms: float
    system_one_latency_ms: float
    system_two_latency_ms: float
    tokens_used: int
    estimated_baseline_tokens: int
    estimated_token_savings_pct: float
    speedup_ratio: float


class DualProcessDispatcher:
    """The Dual-Process Orchestrator coordinating Jev (System 1) with MCP and System 2."""

    def __init__(
        self,
        system_one_client: Optional[JevSystemOneClient] = None,
        system_two_provider: Optional[SystemTwoProvider] = None,
        mcp_host: Optional[MCPHost] = None,
        memory_engine: Optional[MemoryEngine] = None,
        confidence_threshold: float = 0.85,
    ):
        self.s1 = system_one_client or JevSystemOneClient()
        self.s2 = system_two_provider or get_system_two_provider()
        self.mcp = mcp_host or MCPHost()
        self.memory = memory_engine or MemoryEngine()
        self.confidence_threshold = float(
            os.getenv("SYSTEM_ONE_CONFIDENCE_THRESHOLD", str(confidence_threshold))
        )

    def run(self, goal: str, max_steps: int = 15) -> DispatchResult:
        """Execute the agent loop for the given goal."""
        state = AgentState(goal=goal, max_steps=max_steps)
        tool_descriptions = self.mcp.get_tool_descriptions()

        # Check for cached learned skills
        cached_skill = self.memory.find_matching_skill(goal)
        if cached_skill:
            logger.info(
                f"[Learned Skill Replay] Found cached routine '{cached_skill.name}' "
                f"with tool sequence: {cached_skill.tool_sequence}"
            )

        s1_latency_total = 0.0
        s2_latency_total = 0.0

        for step_idx in range(1, max_steps + 1):
            if state.is_completed:
                break

            state_summary = state.to_system_one_state()

            # --- SYSTEM 1 REFLEX EVALUATION (~15ms) ---
            decision: JevDecision = self.s1.evaluate_state_and_route(
                state_text=state_summary,
                tool_options=tool_descriptions,
                allow_escalation=True,
            )
            s1_latency_total += decision.latency_ms

            # Check if Jev determined the task is already finished
            if decision.is_terminal or decision.selected_tool == "finish_task":
                state.is_completed = True
                state.final_output = state.final_output or "Goal satisfied successfully."
                state.history.append(
                    StepRecord(
                        step_index=step_idx,
                        step_type=StepType.TERMINATION,
                        action="finish_task",
                        output=state.final_output,
                        confidence=decision.confidence,
                        latency_ms=decision.latency_ms,
                        tokens_used=0,
                    )
                )
                break

            # --- ROUTING DECISION: Fast-path (System 1) vs Slow-path (System 2) ---
            can_use_fast_path = (
                decision.confidence >= self.confidence_threshold
                and not decision.needs_generation
                and decision.selected_tool in tool_descriptions
            )

            if can_use_fast_path:
                # FAST PATH: Execute MCP tool directly without waking System 2
                tool_name = decision.selected_tool
                default_args = self._infer_default_args(tool_name, state)
                exec_result: ToolExecutionResult = self.mcp.execute_tool(tool_name, default_args)
                step_latency = decision.latency_ms + exec_result.execution_time_ms
                s1_latency_total += exec_result.execution_time_ms

                output_val = exec_result.output if exec_result.success else exec_result.error
                state.history.append(
                    StepRecord(
                        step_index=step_idx,
                        step_type=StepType.SYSTEM_ONE_FAST_TOOL,
                        action=tool_name,
                        action_input=default_args,
                        output=output_val,
                        confidence=decision.confidence,
                        latency_ms=step_latency,
                        tokens_used=0,  # 0 LLM tokens burned for routing
                        metadata={"simulated": decision.simulated},
                    )
                )
            else:
                # SLOW PATH: Escalate to System 2 (Hermes, Grok, Claude)
                s2_prompt = state.to_system_two_prompt(
                    self.mcp.get_formatted_tool_list_for_system_two()
                )
                s2_response = self.s2.generate_step(s2_prompt)
                s2_latency_total += s2_response.latency_ms

                if s2_response.action == "finish_task":
                    state.is_completed = True
                    state.final_output = s2_response.generated_content or s2_response.thought
                    state.history.append(
                        StepRecord(
                            step_index=step_idx,
                            step_type=StepType.SYSTEM_TWO_GENERATION,
                            action="finish_task",
                            action_input=s2_response.args,
                            output=state.final_output,
                            latency_ms=s2_response.latency_ms,
                            tokens_used=s2_response.tokens_used,
                        )
                    )
                    break
                else:
                    exec_result = self.mcp.execute_tool(s2_response.action, s2_response.args)
                    output_val = exec_result.output if exec_result.success else exec_result.error
                    total_step_lat = s2_response.latency_ms + exec_result.execution_time_ms
                    state.history.append(
                        StepRecord(
                            step_index=step_idx,
                            step_type=StepType.SYSTEM_TWO_GENERATION,
                            action=s2_response.action,
                            action_input=s2_response.args,
                            output=output_val,
                            latency_ms=total_step_lat,
                            tokens_used=s2_response.tokens_used,
                        )
                    )

        # Baseline comparison: standard autoregressive agent uses ~1500 tokens & 1200ms per step
        total_steps = max(1, len(state.history))
        baseline_tokens = total_steps * 1500
        actual_tokens = state.total_tokens
        token_savings_pct = max(0.0, ((baseline_tokens - actual_tokens) / baseline_tokens) * 100)
        
        baseline_latency = total_steps * 1200.0
        actual_latency = state.total_latency_ms
        speedup = (baseline_latency / actual_latency) if actual_latency > 0 else 1.0

        # Persist session to SQLite memory
        try:
            self.memory.save_session(
                goal=goal,
                outcome=state.final_output,
                is_completed=state.is_completed,
                total_steps=total_steps,
                system_one_steps=state.system_one_steps,
                system_two_steps=state.system_two_steps,
                total_latency_ms=actual_latency,
                tokens_used=actual_tokens,
                token_savings_pct=round(token_savings_pct, 1),
                steps=[s.model_dump() for s in state.history],
            )

            # Auto-learn skill if completed with at least 1 tool step
            if state.is_completed:
                tool_seq = [s.action for s in state.history if s.action != "finish_task"]
                if tool_seq:
                    keywords = [w.lower() for w in goal.split() if len(w) > 3][:5]
                    clean_name = " ".join(keywords) if keywords else goal[:30]
                    self.memory.save_learned_skill(
                        name=clean_name,
                        intent_keywords=keywords or ["general"],
                        tool_sequence=tool_seq,
                    )
        except Exception as e:
            logger.warning(f"Error persisting session to memory: {e}")

        return DispatchResult(
            goal=goal,
            is_completed=state.is_completed,
            final_output=state.final_output,
            total_steps=total_steps,
            system_one_steps=state.system_one_steps,
            system_two_steps=state.system_two_steps,
            total_latency_ms=actual_latency,
            system_one_latency_ms=s1_latency_total,
            system_two_latency_ms=s2_latency_total,
            tokens_used=actual_tokens,
            estimated_baseline_tokens=baseline_tokens,
            estimated_token_savings_pct=round(token_savings_pct, 1),
            speedup_ratio=round(speedup, 2),
        )

    def _infer_default_args(self, tool_name: str, state: AgentState) -> Dict[str, Any]:
        """Infers deterministic arguments for common tool patterns."""
        if tool_name == "list_directory":
            return {"path": "."}
        elif tool_name == "read_file":
            return {"path": "pyproject.toml"}
        elif tool_name == "run_shell_command":
            return {"command": "ls -la"}
        return {}
