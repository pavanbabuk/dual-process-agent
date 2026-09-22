"""Integration Middleware for Nous Research Hermes Agent.

This module allows Hermes Agent instances (or any LangChain / CrewAI / ReAct loop)
to offload routine tool routing and loop-termination checks to TypeSafe AI's Jev model.

Routing a decision to Jev is billed per input token, and it is far cheaper per call
than a frontier LLM completion. How much that saves on a given workload is NOT
measured here — this repository contains no baseline harness, so no savings figure
is claimed. An earlier revision of this docstring asserted "up to 80% of outer-loop
token costs", which nothing in this project ever measured; it was removed rather
than left standing as an unverifiable number.
"""

from __future__ import annotations
import logging
from typing import Any, Callable, Dict, Optional
from dual_agent.typesafe_client import JevSystemOneClient, JevDecision

logger = logging.getLogger(__name__)


class HermesJevRoutingMiddleware:
    """Drop-in router interceptor for Hermes Agent tool dispatch."""

    def __init__(
        self,
        typesafe_client: Optional[JevSystemOneClient] = None,
        confidence_threshold: float = 0.85,
    ):
        self.jev = typesafe_client or JevSystemOneClient()
        self.confidence_threshold = confidence_threshold

    def intercept_tool_decision(
        self,
        current_state: str,
        available_tools: Dict[str, str],
        hermes_fallback_fn: Callable[[str], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Attempt fast System 1 routing via Jev.
        
        If confidence meets threshold, returns immediately (~15ms) without burning
        Hermes autoregressive tokens. Otherwise, falls back to Hermes generation.
        """
        decision: JevDecision = self.jev.evaluate_state_and_route(
            state_text=current_state,
            tool_options=available_tools,
        )

        if decision.confidence >= self.confidence_threshold and not decision.needs_generation:
            logger.info(f"[Fast-Path] Jev routed to '{decision.selected_tool}' ({decision.latency_ms:.1f}ms, conf={decision.confidence:.2f})")
            return {
                "source": "jev_system_one",
                "tool": decision.selected_tool,
                "confidence": decision.confidence,
                "latency_ms": decision.latency_ms,
                "tokens_burned": 0,
            }

        logger.info("[Slow-Path] Escalating to Hermes System 2 generation.")
        llm_result = hermes_fallback_fn(current_state)
        llm_result["source"] = "hermes_system_two"
        return llm_result
