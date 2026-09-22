"""TypeSafe AI Jev (System 1) integration layer."""

from __future__ import annotations
import os
import time
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

try:
    from typesafe_sdk import TypeSafeClient, Choice, Noul, Score, RetryPolicy
    from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer
    HAS_TYPESAFE_SDK = True
except ImportError:
    HAS_TYPESAFE_SDK = False

logger = logging.getLogger(__name__)


class JevDecision(BaseModel):
    """Decision output from Jev System 1 model."""
    selected_tool: str
    confidence: float
    probabilities: Dict[str, float] = Field(default_factory=dict)
    is_terminal: bool = False
    needs_generation: bool = False
    evaluation_score: Optional[int] = None
    latency_ms: float = 0.0
    simulated: bool = False
    # Set whenever this decision did NOT come from the live Jev model — i.e. the
    # router ran a local heuristic, either because no credentials were present or
    # because a live call failed. Callers must surface this; a simulated decision
    # is not a Jev decision.
    fallback_reason: Optional[str] = None


class JevSystemOneClient:
    """Client wrapper for TypeSafe AI's Jev model (System 1).
    
    Operates in live mode if TYPESAFE_API_KEY is configured,
    or fallback simulation mode for testing and benchmarking.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 5.0,
        force_simulation: bool = False,
    ):
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        self.base_url = base_url or os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai")
        self.timeout = timeout

        # Why we are (or are not) talking to the live Jev model. This reason is
        # attached to every JevDecision so simulated routing can never be
        # silently reported as a real model call.
        self.simulation_reason: str = ""
        if force_simulation:
            self.simulation_reason = "force_simulation=True"
        elif not self.api_key:
            self.simulation_reason = "TYPESAFE_API_KEY is not set"
        elif not HAS_TYPESAFE_SDK:
            self.simulation_reason = "typesafe-sdk is not installed"

        self.force_simulation = bool(self.simulation_reason)
        self._sdk_client: Optional[Any] = None

        if not self.force_simulation:
            try:
                self._sdk_client = TypeSafeClient(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                    retry=RetryPolicy(max_retries=0),
                )
                logger.info("Initialized TypeSafeClient with live API.")
            except Exception as e:
                logger.warning(f"Failed to initialize TypeSafeClient ({e}). Falling back to simulation mode.")
                self.force_simulation = True
                self.simulation_reason = f"TypeSafeClient init failed: {type(e).__name__}: {e}"

    def evaluate_state_and_route(
        self,
        state_text: str,
        tool_options: Dict[str, str],
        allow_escalation: bool = True,
    ) -> JevDecision:
        """Route to next tool or escalate to System 2 using Jev Choice and Noul in parallel.
        
        Args:
            state_text: Compact serialized agent state.
            tool_options: Mapping of tool_name -> description.
            allow_escalation: Whether to include 'escalate_to_system_two' as a choice option.
        """
        start_time = time.perf_counter()
        
        choices = dict(tool_options)
        if allow_escalation and "escalate_to_system_two" not in choices:
            choices["escalate_to_system_two"] = "Escalate to System 2 LLM for creative generation, complex reasoning, or code synthesis."
        if "finish_task" not in choices:
            choices["finish_task"] = "Task goal is fully achieved. End loop and present final output."

        if not self.force_simulation and self._sdk_client:
            return self._call_live_jev(state_text, choices, start_time)
        else:
            return self._call_simulated_jev(state_text, choices, start_time)

    def _call_live_jev(
        self,
        state_text: str,
        choices: Dict[str, str],
        start_time: float,
    ) -> JevDecision:
        """Invoke live Jev model via typesafe-sdk."""
        try:
            criteria = {k: None for k in choices.keys()}
            questions = {
                "route": Choice(
                    instructions="Select the single most appropriate tool or action to progress the goal given current state.",
                    criteria=criteria,
                ),
                "is_finished": Noul(
                    instructions="Has the agent completely fulfilled the user's primary goal?",
                ),
                "needs_synthesis": Noul(
                    instructions="Does the current step require open-ended creative writing, complex script synthesis, or novel reasoning?",
                ),
            }
            
            response = self._sdk_client.system_one(
                state=state_text,
                questions=questions,
            )
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            choice_ans: ChoiceAnswer = response.choices["route"]
            noul_finish: NoulAnswer = response.nouls["is_finished"]
            noul_synth: NoulAnswer = response.nouls["needs_synthesis"]

            selected = choice_ans.choice
            confidence = choice_ans.confidence or 0.90
            probs = choice_ans.probabilities or {selected: confidence}
            
            is_terminal = (noul_finish.noul > 0.85) or (selected == "finish_task")
            needs_gen = (noul_synth.noul > 0.70) or (selected == "escalate_to_system_two")

            return JevDecision(
                selected_tool=selected,
                confidence=confidence,
                probabilities=probs,
                is_terminal=is_terminal,
                needs_generation=needs_gen,
                latency_ms=elapsed_ms,
                simulated=False,
            )
        except Exception as e:
            logger.warning(f"Live Jev API failed ({type(e).__name__}: {e}). Gracefully falling back to simulation mode.")
            self.force_simulation = True
            self.simulation_reason = f"live call failed: {type(e).__name__}: {e}"
            return self._call_simulated_jev(state_text, choices, start_time)

    def _call_simulated_jev(
        self,
        state_text: str,
        choices: Dict[str, str],
        start_time: float,
    ) -> JevDecision:
        """Local heuristic router used when the live Jev model is unavailable.

        IMPORTANT: this is a deterministic keyword-matching stub, NOT the Jev
        model. It sleeps ~12ms purely to emulate a network roundtrip so that
        local benchmarking has a realistic latency profile — the sleep is a
        simulation artifact and must never be reported as model latency.
        Every decision it returns carries `simulated=True` and a
        `fallback_reason` explaining why the live model was not used.
        """
        time.sleep(0.012)  # Simulation artifact: emulates a Jev roundtrip. Not real latency.
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        state_lower = state_text.lower()
        
        # Heuristic state inspection for simulation
        is_terminal = "goal achieved" in state_lower or "all steps completed" in state_lower
        needs_gen = (
            "synthesize" in state_lower 
            or "write a novel" in state_lower 
            or "creative" in state_lower
            or "complex refactor" in state_lower
        )

        selected = "finish_task" if is_terminal else "escalate_to_system_two" if needs_gen else None
        
        if not selected:
            # Pick the best matching tool from choices
            for tool_name in choices.keys():
                if tool_name in ("escalate_to_system_two", "finish_task"):
                    continue
                # Simple keyword relevance matching
                keyword = tool_name.replace("_", " ").split()[-1]
                if keyword in state_lower:
                    selected = tool_name
                    break
            
            if not selected:
                # Default to first non-special tool or escalate
                regular_tools = [k for k in choices.keys() if k not in ("escalate_to_system_two", "finish_task")]
                selected = regular_tools[0] if regular_tools else "escalate_to_system_two"

        confidence = 0.94 if not needs_gen else 0.45
        probs = {k: 0.05 for k in choices.keys()}
        probs[selected] = confidence

        return JevDecision(
            selected_tool=selected,
            confidence=confidence,
            probabilities=probs,
            is_terminal=is_terminal,
            needs_generation=needs_gen,
            latency_ms=elapsed_ms,
            simulated=True,
            fallback_reason=self.simulation_reason or "simulation mode",
        )

    def evaluate_output_score(self, output_text: str, rubric: str) -> int:
        """Score output using Jev Score (1 to 5)."""
        start_time = time.perf_counter()
        if not self.force_simulation and self._sdk_client:
            try:
                response = self._sdk_client.system_one(
                    state=output_text,
                    questions={
                        "quality": Score(
                            instructions=f"Score this result against the rubric: {rubric}",
                            criteria=["Poor", "Fair", "Good", "Very Good", "Excellent"],
                        )
                    },
                )
                score_ans: ScoreAnswer = response.scores["quality"]
                return max(1, min(5, round(float(score_ans.score))))
            except Exception as e:
                logger.warning(f"Error scoring with Jev: {e}. Defaulting to 4.")
                return 4

        # Offline heuristic: length-based, not a real quality judgement.
        time.sleep(0.008)
        return 5 if len(output_text.strip()) > 10 else 2
