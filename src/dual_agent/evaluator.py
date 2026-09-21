"""Fast-path evaluation, guardrails, and termination checks using Jev (System 1)."""

from __future__ import annotations
import logging
from typing import Optional
from pydantic import BaseModel
from dual_agent.typesafe_client import JevSystemOneClient
from dual_agent.state import AgentState

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    is_valid: bool
    score: int
    confidence: float
    reason: str
    latency_ms: float = 0.0


class JevEvaluator:
    """Evaluates agent outputs and state using Jev's Noul and Score primitives."""

    def __init__(self, client: Optional[JevSystemOneClient] = None):
        self.client = client or JevSystemOneClient()

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
