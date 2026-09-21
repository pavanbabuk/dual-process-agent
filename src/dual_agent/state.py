"""State models for the Dual-Process Agent Runtime."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import time


class StepType(str, Enum):
    SYSTEM_ONE_FAST_TOOL = "system_one_fast_tool"
    SYSTEM_TWO_GENERATION = "system_two_generation"
    SYSTEM_ONE_EVALUATION = "system_one_evaluation"
    TERMINATION = "termination"


class StepRecord(BaseModel):
    """Record of an individual step executed in the agent loop."""
    step_index: int
    step_type: StepType
    action: str
    action_input: Any = None
    output: Any = None
    confidence: Optional[float] = None
    latency_ms: float = 0.0
    tokens_used: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """Current state and execution history of the agent."""
    goal: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    history: List[StepRecord] = Field(default_factory=list)
    is_completed: bool = False
    final_output: Optional[str] = None
    max_steps: int = 20

    @property
    def step_count(self) -> int:
        return len(self.history)

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.history)

    @property
    def system_one_steps(self) -> int:
        return sum(
            1 for s in self.history 
            if s.step_type in (StepType.SYSTEM_ONE_FAST_TOOL, StepType.SYSTEM_ONE_EVALUATION)
        )

    @property
    def system_two_steps(self) -> int:
        return sum(1 for s in self.history if s.step_type == StepType.SYSTEM_TWO_GENERATION)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens_used for s in self.history)

    def to_system_one_state(self) -> str:
        """Compact string representation formatted for Jev state evaluation."""
        recent_actions = []
        for s in self.history[-5:]:
            recent_actions.append(f"Step {s.step_index}: {s.action} -> {str(s.output)[:120]}")
        history_summary = " | ".join(recent_actions) if recent_actions else "No prior actions."
        return (
            f"GOAL: {self.goal}\n"
            f"CURRENT STEP: {self.step_count + 1}/{self.max_steps}\n"
            f"RECENT ACTIONS: {history_summary}\n"
            f"VARIABLES: {list(self.variables.keys())}"
        )

    def to_system_two_prompt(self, available_tools_desc: str) -> str:
        """Detailed prompt context for System 2 LLM when creative generation is required."""
        history_lines = []
        for s in self.history:
            history_lines.append(
                f"- [{s.step_type.value}] Action: {s.action}\n"
                f"  Input: {s.action_input}\n"
                f"  Output: {s.output}"
            )
        history_text = "\n".join(history_lines) if history_lines else "None."

        return (
            f"You are the System 2 Reasoning Engine in a Dual-Process Agent architecture.\n"
            f"The fast reflex router (System 1) encountered a step requiring creative generation, code synthesis, or ambiguous reasoning.\n\n"
            f"GOAL: {self.goal}\n\n"
            f"EXECUTION HISTORY:\n{history_text}\n\n"
            f"AVAILABLE MCP TOOLS:\n{available_tools_desc}\n\n"
            f"Please provide your reasoning and the next concrete tool call or final synthesis in JSON format:\n"
            f"{{\"thought\": \"<reasoning>\", \"action\": \"<tool_name_or_finish>\", \"args\": {{...}}}}"
        )
