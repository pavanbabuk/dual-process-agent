"""State models for the Dual-Process Agent Runtime."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import time

# Bounds on the System 2 prompt.
#
# Every prior step's output is re-sent on every subsequent step, so a run that
# reads a few source files grows its prompt without limit. Measured on this
# codebase: 27.5K chars (~6.9K tokens) after 3 steps, ~130K chars (~32K tokens)
# by step 15, because read_file alone returns up to 10KB and nothing trimmed it.
# That is what made hosted providers time out inside the agent loop and then
# degrade to the mock provider — the run looked successful while generating
# nothing. Bounding the prompt is a correctness requirement, not an optimisation.
MAX_HISTORY_STEPS_IN_PROMPT = 6
MAX_OUTPUT_CHARS_PER_STEP = 1200


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


class PlanStep(BaseModel):
    """An individual step in a decomposed execution plan."""
    step_id: int
    description: str
    target_tool: Optional[str] = None
    completed: bool = False
    verified: bool = False
    verification_notes: Optional[str] = None


class AgentState(BaseModel):
    """Current state and execution history of the agent."""
    goal: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    history: List[StepRecord] = Field(default_factory=list)
    plan: List[PlanStep] = Field(default_factory=list)
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
        
        plan_summary = ""
        if self.plan:
            steps_desc = [
                f"[{'x' if p.completed else ' '}] {p.step_id}. {p.description}"
                for p in self.plan
            ]
            plan_summary = f"\nPLAN: {' | '.join(steps_desc)}"

        return (
            f"GOAL: {self.goal}\n"
            f"CURRENT STEP: {self.step_count + 1}/{self.max_steps}\n"
            f"RECENT ACTIONS: {history_summary}{plan_summary}\n"
            f"VARIABLES: {list(self.variables.keys())}"
        )

    def to_system_two_prompt(self, available_tools_desc: str) -> str:
        """Detailed prompt context for System 2 LLM when creative generation is required.

        Outputs are truncated and history is windowed (see MAX_HISTORY_STEPS_IN_PROMPT).
        Without those bounds the prompt grows with every step and eventually makes
        the provider time out, which degrades to the mock and produces nothing.
        """
        history_lines = []
        # Only the most recent steps are relevant to the next decision, and older
        # ones are what blow up the prompt.
        recent_steps = self.history[-MAX_HISTORY_STEPS_IN_PROMPT:]
        for idx, s in enumerate(recent_steps):
            output = str(s.output)
            # The most recent step's output is directly relevant to what the model
            # is about to decide/do (e.g. read_file or search results). Give it
            # up to 4000 chars so code sections aren't blinded, while older steps
            # are kept tight to MAX_OUTPUT_CHARS_PER_STEP.
            max_chars = 4000 if (idx == len(recent_steps) - 1) else MAX_OUTPUT_CHARS_PER_STEP
            if len(output) > max_chars:
                omitted = len(output) - max_chars
                output = (
                    output[:max_chars]
                    + f"\n...[truncated {omitted} chars]"
                )
            history_lines.append(
                f"- [{s.step_type.value}] Action: {s.action}\n"
                f"  Input: {s.action_input}\n"
                f"  Output: {output}"
            )
        history_text = "\n".join(history_lines) if history_lines else "None."
        if len(self.history) > MAX_HISTORY_STEPS_IN_PROMPT:
            history_text = (
                f"({len(self.history) - MAX_HISTORY_STEPS_IN_PROMPT} earlier steps omitted)\n"
                + history_text
            )

        plan_section = ""
        if self.plan:
            plan_lines = []
            for p in self.plan:
                status = "COMPLETED" if p.completed else ("VERIFIED" if p.verified else "PENDING")
                plan_lines.append(f"  {p.step_id}. [{status}] {p.description}")
            plan_section = f"EXECUTION PLAN:\n" + "\n".join(plan_lines) + "\n\n"

        remaining = max(0, self.max_steps - len(self.history))
        return (
            f"You are the System 2 Reasoning Engine in a Dual-Process Agent architecture.\n"
            f"The fast reflex router (System 1) encountered a step requiring deliberate tool execution or code synthesis.\n\n"
            f"GOAL: {self.goal}\n"
            f"STEP BUDGET: {remaining} step(s) remaining out of {self.max_steps}.\n\n"
            f"{plan_section}"
            f"EXECUTION HISTORY (most recent {MAX_HISTORY_STEPS_IN_PROMPT} steps):\n{history_text}\n\n"
            f"AVAILABLE MCP TOOLS:\n{available_tools_desc}\n\n"
            f"CRITICAL INSTRUCTIONS:\n"
            f"1. Keep 'thought' concise (1 to 2 sentences). State your immediate concrete action.\n"
            f"2. Use 'search_file' to locate target code/symbols (e.g. parser, imports, version) directly instead of re-reading large files.\n"
            f"3. Use 'patch_file' to apply surgical changes.\n"
            f"4. Verify your edits with 'run_shell_command' (e.g. python -m py_compile <file>).\n"
            f"5. Once verified, immediately conclude with action 'finish_task'.\n"
            f"6. Always output strictly valid JSON matching this schema:\n"
            f'{{"thought": "<concise reasoning>", "action": "<mcp_tool_name>", "args": {{...}}}}\n'
        )
