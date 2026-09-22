"""Dual-Process Dispatcher and Execution Engine."""

from __future__ import annotations
import os
import re
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from dual_agent.state import AgentState, PlanStep, StepRecord, StepType
from dual_agent.typesafe_client import JevSystemOneClient, JevDecision
from dual_agent.mcp_host import MCPHost, ToolExecutionResult
from dual_agent.system_two import SystemTwoProvider, get_system_two_provider
from dual_agent.evaluator import JevEvaluator
from dual_agent.memory import MemoryEngine, LearnedSkill
from dual_agent.skills_manager import SkillsManager
from dual_agent.permission_broker import PermissionBroker, ApprovalDecision

logger = logging.getLogger(__name__)


def validate_tool_args(tool, args: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate inferred tool arguments against the tool's declared schema.

    Returns (ok, message). A missing tool, missing required property, or
    wrong-typed value all fail. This is the gate that keeps the fast path from
    executing a tool with arguments nobody verified.
    """
    if tool is None:
        return False, "tool is not registered"
    schema = tool.parameters_schema or {}
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for name in required:
        value = args.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            return False, f"required argument '{name}' is missing or empty"

    for name, value in args.items():
        if name not in properties:
            # Reject unknown keys to prevent executing tools with hallucinated/unverified parameters.
            return False, f"unknown argument '{name}' is not in tool schema"

        expected = properties[name].get("type")
        if expected == "string" and not isinstance(value, str):
            return False, f"argument '{name}' must be a string, got {type(value).__name__}"
        elif expected == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            return False, f"argument '{name}' must be an integer, got {type(value).__name__}"
        elif expected in ("number", "float") and (isinstance(value, bool) or not isinstance(value, (int, float))):
            return False, f"argument '{name}' must be a number, got {type(value).__name__}"
        elif expected == "boolean" and not isinstance(value, bool):
            return False, f"argument '{name}' must be a boolean, got {type(value).__name__}"
        elif expected in ("array", "list") and not isinstance(value, list):
            return False, f"argument '{name}' must be a list/array, got {type(value).__name__}"
        elif expected in ("object", "dict") and not isinstance(value, dict):
            return False, f"argument '{name}' must be a dict/object, got {type(value).__name__}"
    return True, ""


class DispatchResult(BaseModel):
    """Complete summary of a dual-process execution run.

    All latency/token fields are MEASURED. There is deliberately no
    "savings %" or "speedup x" field: those require a real baseline run of the
    same goal against a plain LLM agent, which this runtime does not perform.
    Modelling them from constants would be fabricating a benchmark result.
    """
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
    # True when System 1 routing was performed by the local heuristic stub rather
    # than the live Jev model. Any run where this is True cannot substantiate
    # claims about Jev latency or router quality.
    used_simulated_system_one: bool = False
    # Human-readable explanation of why simulation mode was active (None if live).
    system_one_fallback_reason: Optional[str] = None
    # Extra wall-clock cost attributable to simulation artifacts (the emulated
    # Jev roundtrip sleeps). Excluded from nothing — reported so the number
    # cannot be mistaken for real model latency.
    simulated_latency_ms: float = 0.0
    # True when System 2 fell back to the mock provider, so the "generated"
    # content is canned text rather than model output. Surfaced because a run
    # that reports success while generating nothing is the failure mode this
    # codebase has already been burned by (fabricated benchmark columns).
    system_two_is_mock: bool = False
    system_two_degraded_reason: Optional[str] = None
    plan: List[PlanStep] = Field(default_factory=list)


class DualProcessDispatcher:
    """The Dual-Process Orchestrator coordinating Jev (System 1) with MCP and System 2."""

    def __init__(
        self,
        system_one_client: Optional[JevSystemOneClient] = None,
        system_two_provider: Optional[SystemTwoProvider] = None,
        mcp_host: Optional[MCPHost] = None,
        memory_engine: Optional[MemoryEngine] = None,
        skills_manager: Optional[SkillsManager] = None,
        permission_broker: Optional[PermissionBroker] = None,
        evaluator: Optional[JevEvaluator] = None,
        confidence_threshold: float = 0.85,
        step_callback: Optional[Callable[[StepRecord], None]] = None,
        config: Optional[Any] = None,
    ):
        if config is not None:
            self.config = config
        else:
            try:
                from dual_agent.config import load_config
                self.config = load_config()
            except Exception:
                self.config = None
        self.s1 = system_one_client or JevSystemOneClient()
        self.s2 = system_two_provider or get_system_two_provider(config=self.config)
        self.mcp = mcp_host or MCPHost()
        self.evaluator = evaluator or JevEvaluator(self.s1)
        self.memory = memory_engine or MemoryEngine()
        self.skills = skills_manager or SkillsManager()
        self.step_callback = step_callback
        # Auto-allow permissions in CI/non-interactive environments
        auto_allow = os.getenv("DUAL_AGENT_AUTO_ALLOW_PERMISSIONS", "false").lower() == "true"
        self.broker = permission_broker or PermissionBroker(
            memory_engine=self.memory, auto_allow=auto_allow
        )
        self.confidence_threshold = float(
            os.getenv("SYSTEM_ONE_CONFIDENCE_THRESHOLD", str(confidence_threshold))
        )
        # The fast path executes a tool WITHOUT asking an LLM for its arguments,
        # so the router's choice has to be verified before it is trusted.
        # Opting out is explicit and noisy because it re-enables a path that can
        # only guess arguments.
        self.allow_unverified_fast_path = (
            os.getenv("DUAL_AGENT_ALLOW_UNVERIFIED_FAST_PATH", "false").lower() == "true"
        )
        self._current_goal: str = ""
        # Wall-clock nanoseconds-equivalent (ms) spent on verification router
        # calls, so they can be attributed to the step that triggered them.
        self._verification_clock_ms: float = 0.0
        self._pending_verification_latency_ms: float = 0.0

    def _scan_workspace(self, workspace_path: str) -> None:
        """Scan workspace for tech stack markers and persist to memory."""
        try:
            languages = []
            features = {}
            if os.path.exists(os.path.join(workspace_path, "pyproject.toml")) or os.path.exists(os.path.join(workspace_path, "setup.py")):
                languages.append("python")
                features["has_pyproject"] = True
            if os.path.exists(os.path.join(workspace_path, "package.json")):
                languages.append("javascript/typescript")
                features["has_package_json"] = True
            if os.path.exists(os.path.join(workspace_path, "Cargo.toml")):
                languages.append("rust")
                features["has_cargo"] = True
            if os.path.exists(os.path.join(workspace_path, "go.mod")):
                languages.append("go")
                features["has_gomod"] = True

            tech_stack = {"languages": languages, **features}
            user_prefs = {"confidence_threshold": self.confidence_threshold}
            self.memory.save_project_context(
                workspace_path=workspace_path,
                tech_stack=tech_stack,
                user_preferences=user_prefs,
            )
        except Exception as e:
            logger.debug(f"[Dispatcher] Workspace scan error: {e}")

    def _record_step(
        self,
        state: AgentState,
        record: StepRecord,
        callback: Optional[Callable[[StepRecord], None]] = None,
    ) -> None:
        """Append step record to state history and notify callback if registered."""
        state.history.append(record)
        cb = callback or self.step_callback
        if cb:
            try:
                cb(record)
            except Exception as e:
                logger.debug(f"[Dispatcher] step_callback error: {e}")

    def _create_plan(self, goal: str, available_tools_desc: str) -> List[PlanStep]:
        """Decompose a goal into concrete, sequential subgoals before execution."""
        subgoals: List[str] = []
        if self.s2 and not getattr(self.s2, "is_mock", False):
            prompt = (
                f"You are the Planning Engine in a Dual-Process Agent.\n"
                f"Decompose the user's goal into an ordered sequence of 2 or 3 concise, actionable subgoals (inspect, patch, verify).\n"
                f"GOAL: {goal}\n"
                f"AVAILABLE TOOLS:\n{available_tools_desc}\n\n"
                f"Respond with JSON only: {{\"thought\": \"planning\", \"action\": \"plan_subgoals\", \"args\": {{\"subgoals\": [\"<subgoal 1>\", \"<subgoal 2>\", ...]}}}}"
            )
            try:
                resp = self.s2.generate_step(prompt)
                if resp.action == "plan_subgoals" and "subgoals" in (resp.args or {}):
                    subgoals = resp.args["subgoals"]
                elif resp.generated_content:
                    import json
                    parsed = json.loads(resp.generated_content)
                    if "subgoals" in parsed and isinstance(parsed["subgoals"], list):
                        subgoals = parsed["subgoals"]
                    elif "args" in parsed and "subgoals" in parsed["args"]:
                        subgoals = parsed["args"]["subgoals"]
                if not subgoals and (resp.args or {}).get("subgoals"):
                    subgoals = resp.args["subgoals"]
            except Exception as e:
                logger.debug(f"[Dispatcher] System 2 planning fallback: {e}")

        if not subgoals:
            lower_g = goal.lower()
            if any(conj in lower_g for conj in (" and ", " then ", " also ", "add ", "update ", "modify ", "document ")):
                subgoals = [
                    f"Inspect workspace and examine target files for '{goal[:40]}'",
                    f"Perform requested modifications or patches for '{goal[:40]}'",
                    f"Verify changes parse, import, and satisfy '{goal[:40]}'",
                ]
            else:
                subgoals = [f"Execute task: {goal}"]

        return [
            PlanStep(step_id=idx + 1, description=desc)
            for idx, desc in enumerate(subgoals)
        ]

    def _update_plan_progress(
        self,
        state: AgentState,
        action: str,
        is_verified: bool,
        notes: str,
    ) -> None:
        """Update plan steps with execution outcome."""
        if not state.plan:
            return
        active_step = next((p for p in state.plan if not p.completed), None)
        if not active_step:
            return

        if not is_verified:
            active_step.verification_notes = f"Verification failed: {notes}"
            return

        active_step.verification_notes = notes
        desc = active_step.description.lower()
        if any(w in desc for w in ("inspect", "read", "examine", "locate", "check", "find")):
            if action in ("read_file", "list_directory", "run_shell_command"):
                active_step.completed = True
                active_step.verified = True
        elif any(w in desc for w in ("modify", "patch", "write", "add", "update", "edit", "implement")):
            if action in ("patch_file", "write_file", "run_shell_command"):
                active_step.completed = True
                active_step.verified = True
        elif any(w in desc for w in ("verify", "test", "validate", "confirm")):
            if action in ("run_shell_command", "read_file", "finish_task"):
                active_step.completed = True
                active_step.verified = True
        else:
            active_step.completed = True
            active_step.verified = True

    def _get_tool_confidence_threshold(self, tool_name: str) -> float:
        """Derive confidence threshold per tool risk level (vendor spec alignment).

        Destructive and file-modifying tools require higher confidence (>= 0.88 - 0.92)
        and verification, while read-only / non-destructive inspection tools have a
        lower threshold (~0.75).
        """
        tool_def = self.mcp.get_tool(tool_name)
        if not tool_def:
            return float(os.getenv("ACTION_CONFIDENCE_THRESHOLD_HIGH_RISK", "0.92"))

        risk = (tool_def.risk_level or "").lower()
        if risk == "high" or "destructive" in (tool_def.description or "").lower():
            return float(os.getenv("ACTION_CONFIDENCE_THRESHOLD_HIGH_RISK", "0.92"))
        elif risk == "medium" or tool_def.requires_approval or tool_name in ("write_file", "patch_file"):
            return float(os.getenv("ACTION_CONFIDENCE_THRESHOLD_MEDIUM_RISK", "0.88"))
        else:
            # Low risk (read-only inspection, e.g. list_directory, read_file)
            return float(os.getenv("ACTION_CONFIDENCE_THRESHOLD_LOW_RISK", "0.75"))

    def run(
        self,
        goal: str,
        max_steps: int = 15,
        step_callback: Optional[Callable[[StepRecord], None]] = None,
        enable_screen_loop: Optional[bool] = None,
    ) -> DispatchResult:
        """Execute the agent loop for the given goal."""
        self._current_goal = goal
        active_callback = step_callback or self.step_callback
        run_started_at = time.perf_counter()

        if enable_screen_loop is None:
            enable_screen_loop = any(
                kw in goal.lower()
                for kw in ("screen", "desktop", "click", "mouse", "window", "display", "gui", "screenshot", "type into")
            ) or (os.environ.get("DUAL_AGENT_SCREEN_LOOP", "0").lower() in ("1", "true"))

        # --- VISION PRECONDITION ---
        # A screen loop needs eyes, and sight is the part that is not free. If
        # the provider cannot see, say so now and refuse the run: capturing
        # screenshots for an agent that will never look at them burns steps, and
        # the failure would otherwise surface late as a generic System 2 abort.
        #
        # Only a real implementation counts. `getattr` on a MagicMock returns a
        # truthy mock for *any* attribute, so a bare getattr would make every
        # mocked test provider look blind and short-circuit its run — the check
        # must be for a concrete method on the class, not for attribute presence.
        screen_preflight = None
        s2_has_preflight = getattr(type(self.s2), "preflight", None)
        if callable(s2_has_preflight):
            screen_preflight = getattr(self.s2, "preflight")
        if enable_screen_loop and callable(screen_preflight):
            blocking = screen_preflight()
            if blocking:
                message = (
                    f"Screen control was requested but the System 2 provider cannot see: "
                    f"{blocking}"
                )
                logger.warning(f"[Dispatcher] {message}")
                return DispatchResult(
                    goal=goal,
                    is_completed=False,
                    final_output=f"System 2 failure: {message}",
                    total_steps=0,
                    system_one_steps=0,
                    system_two_steps=0,
                    total_latency_ms=(time.perf_counter() - run_started_at) * 1000,
                    system_one_latency_ms=0.0,
                    system_two_latency_ms=0.0,
                    tokens_used=0,
                    used_simulated_system_one=bool(getattr(self.s1, "force_simulation", False)),
                    system_one_fallback_reason=getattr(self.s1, "simulation_reason", None) or None,
                    simulated_latency_ms=0.0,
                    # Reported as a degraded System 2 so no caller reads a
                    # zero-step run as a completed goal.
                    system_two_is_mock=True,
                    system_two_degraded_reason=message,
                    plan=[],
                )
        # Verification router calls are billed here as they happen; anything the
        # per-step records miss is reconciled against the wall clock at the end.
        self._verification_clock_ms = 0.0
        self._pending_verification_latency_ms = 0.0
        state = AgentState(goal=goal, max_steps=max_steps)
        tool_descriptions = self.mcp.get_tool_descriptions()

        # --- WORKSPACE SCAN: discover project context if enabled ---
        if getattr(self.config, "auto_scan_workspace", False):
            self._scan_workspace(os.getcwd())

        # --- GOAL DECOMPOSITION: create structured plan ---
        state.plan = self._create_plan(goal, self.mcp.get_formatted_tool_list_for_system_two())
        verification_failures: Dict[int, int] = {}
        correction_guidance = ""

        # --- RECALL CONTEXT: inject past sessions matching this goal ---
        recall_ctx = self.memory.build_recall_context(goal, limit=3)
        if recall_ctx:
            logger.debug(f"[Dispatcher] Injecting recall context ({len(recall_ctx)} chars)")

        # --- SKILL CONTEXT: inject relevant .SKILL.md procedures ---
        skill_ctx = self.skills.build_skill_context(goal)
        if skill_ctx:
            logger.debug(f"[Dispatcher] Injecting skill context ({len(skill_ctx)} chars)")

        # Note: self.memory.find_matching_skill is reserved for memory lookup;
        # inert replay logging without actual execution skipping has been removed.

        s1_latency_total = 0.0
        s2_latency_total = 0.0
        simulated_s1_latency = 0.0
        s2_is_mock = False
        s2_degraded_reason: Optional[str] = None

        # Stall detection. A goal the router cannot terminate on (e.g. one whose
        # text never satisfies the terminal check) used to spin until max_steps,
        # repeating the same tool with the same arguments and the same output.
        # In live mode each of those steps is a paid Jev call, so a stuck loop
        # costs real money to produce nothing. Three consecutive no-progress
        # steps end the run instead.
        last_signature = None
        repeat_count = 0
        stalled = False

        # Surface simulation mode up front rather than only in logs — a run whose
        # "System 1" is the local stub must never be reported as a Jev run.
        if getattr(self.s1, "force_simulation", False):
            reason = getattr(self.s1, "simulation_reason", "simulation mode")
            logger.warning(
                f"[Dispatcher] System 1 is SIMULATED (not the Jev model): {reason}. "
                "Measured latencies reflect the local stub, not Jev."
            )

        for step_idx in range(1, max_steps + 1):
            if state.is_completed:
                break

            # --- SCREEN PERCEPTION (when screen loop enabled) ---
            step_screenshot_path: Optional[str] = None
            step_grid_path: Optional[str] = None
            if enable_screen_loop and self.mcp.get_tool("screenshot"):
                try:
                    ss_res = self.mcp.execute_tool("screenshot", {})
                    if ss_res.success and ss_res.output:
                        import json as _json
                        ss_data = _json.loads(ss_res.output)
                        step_screenshot_path = ss_data.get("path")
                        if step_screenshot_path and self.mcp.get_tool("grid_overlay"):
                            g_res = self.mcp.execute_tool("grid_overlay", {"image_path": step_screenshot_path})
                            if g_res.success and g_res.output:
                                g_data = _json.loads(g_res.output)
                                step_grid_path = g_data.get("overlay_path")
                except Exception as ss_e:
                    logger.debug(f"[Dispatcher] Screen observation error: {ss_e}")

            active_plan_step = next((p for p in state.plan if not p.completed), None)
            current_subgoal = active_plan_step.description if active_plan_step else goal
            state_summary = state.to_system_one_state()

            # --- SYSTEM 1 REFLEX EVALUATION (~15ms) ---
            decision: JevDecision = self.s1.evaluate_state_and_route(
                state_text=state_summary,
                tool_options=tool_descriptions,
                allow_escalation=True,
            )
            s1_latency_total += decision.latency_ms
            # Verification router calls were real model calls; count them.
            s1_latency_total += self._pending_verification_latency_ms
            if decision.simulated:
                simulated_s1_latency += decision.latency_ms
                simulated_s1_latency += self._pending_verification_latency_ms

            # Check if Jev determined the task is already finished
            if decision.is_terminal or decision.selected_tool == "finish_task":
                step_fail = verification_failures.get(active_plan_step.step_id if active_plan_step else 0, 0)
                if step_fail == 0 and self.evaluator.check_task_completion(state):
                    state.is_completed = True
                    state.final_output = state.final_output or "Goal satisfied successfully."
                    for p in state.plan:
                        if not p.completed:
                            p.completed = True
                            p.verified = True
                    self._record_step(
                        state,
                        StepRecord(
                            step_index=step_idx,
                            step_type=StepType.TERMINATION,
                            action="finish_task",
                            output=state.final_output,
                            confidence=decision.confidence,
                            latency_ms=decision.latency_ms,
                            tokens_used=0,
                        ),
                        active_callback,
                    )
                    break
                else:
                    # Incomplete or failed verification; terminal reflex was premature
                    decision.needs_generation = True

            # --- ROUTING DECISION: Fast-path (System 1) vs Slow-path (System 2) ---
            # Assigned only on the fast path; pre-declared so a rejected fast path
            # can never leave them unbound.
            tool_name: str = ""
            default_args: Dict[str, Any] = {}
            fast_path_confidence = self._compute_fast_path_confidence(decision, state)

            tool_threshold = self._get_tool_confidence_threshold(decision.selected_tool)
            can_use_fast_path = (
                fast_path_confidence >= tool_threshold
                and not decision.needs_generation
                and decision.selected_tool in tool_descriptions
            )

            if can_use_fast_path:
                tool_name = decision.selected_tool
                default_args = self._infer_default_args(tool_name, state)

                # --- SCHEMA VALIDATION: verify args against the tool's schema ---
                # The router only chose a tool NAME; nothing has yet confirmed
                # that the inferred arguments are the right ones. Sending
                # plausible-but-wrong args (e.g. reading pyproject.toml for an
                # unrelated goal) looks like success and is worse than failing.
                arg_ok, arg_errors = validate_tool_args(
                    self.mcp.get_tool(tool_name), default_args
                )

                if not arg_ok:
                    # Refuse the fast path rather than guessing. Fall through to
                    # System 2, which can read the goal and supply real args.
                    logger.info(
                        f"[Dispatcher] Fast path refused for '{tool_name}': "
                        f"{arg_errors} — escalating to System 2 for arguments."
                    )
                    if self.allow_unverified_fast_path:
                        logger.warning(
                            "[Dispatcher] DUAL_AGENT_ALLOW_UNVERIFIED_FAST_PATH=true — "
                            "executing with unverified arguments anyway."
                        )
                    else:
                        decision.needs_generation = True
                        can_use_fast_path = False

            if can_use_fast_path:
                # --- PERMISSION BROKER: gate risky tools ---
                tool_def = self.mcp.get_tool(tool_name)
                if tool_def and tool_def.requires_approval:
                    approval, default_args = self.broker.request_approval(
                        tool_name=tool_name,
                        args=default_args,
                        risk_level=tool_def.risk_level,
                    )
                    if approval == ApprovalDecision.DENY:
                        self._record_step(
                            state,
                            StepRecord(
                                step_index=step_idx,
                                step_type=StepType.SYSTEM_ONE_FAST_TOOL,
                                action=tool_name,
                                action_input=default_args,
                                output="[DENIED by user]",
                                confidence=decision.confidence,
                                latency_ms=decision.latency_ms,
                                tokens_used=0,
                            ),
                            active_callback,
                        )
                        continue

                exec_result: ToolExecutionResult = self.mcp.execute_tool(tool_name, default_args)
                step_latency = decision.latency_ms + exec_result.execution_time_ms
                s1_latency_total += exec_result.execution_time_ms

                output_val = exec_result.output if exec_result.success else exec_result.error
                self._record_step(
                    state,
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
                    ),
                    active_callback,
                )

                # --- OUTCOME VERIFICATION (JevEvaluator) ---
                is_verified, v_notes = self.evaluator.verify_step_outcome(
                    current_subgoal, tool_name, default_args, output_val
                )
                self._update_plan_progress(state, tool_name, is_verified, v_notes)
                if not is_verified:
                    step_id = active_plan_step.step_id if active_plan_step else 0
                    verification_failures[step_id] = verification_failures.get(step_id, 0) + 1
                    correction_guidance = (
                        f"[VERIFICATION FAILED]: Step '{tool_name}' failed verification: {v_notes}. "
                        f"Subgoal was: '{current_subgoal}'. Correct your parameters or approach.\n"
                    )
                    if verification_failures[step_id] >= 3:
                        state.is_completed = False
                        state.final_output = (
                            f"Could not complete task: subgoal '{current_subgoal}' "
                            f"failed verification repeatedly: {v_notes}"
                        )
                        break
                else:
                    correction_guidance = ""

                signature = (tool_name, repr(default_args), str(output_val)[:200])
                repeat_count = repeat_count + 1 if signature == last_signature else 1
                last_signature = signature
                if repeat_count >= 3:
                    # Stalled runs halted early without accomplishing the goal.
                    stalled = True
                    state.is_completed = True
                    state.final_output = (
                        f"Stopped after {repeat_count} identical "
                        f"'{tool_name}' steps with no change in output — the goal "
                        "is not progressing. Nothing further is being attempted."
                    )
                    logger.warning(
                        f"[Dispatcher] Stalled on repeated '{tool_name}' "
                        f"({repeat_count}x identical output); ending run early."
                    )
                    break
            else:
                # SLOW PATH: Escalate to System 2 LLM (DeepSeek, Grok, OpenAI, Custom)
                # Recompute recall context per step so earlier actions and findings in this run
                # are incorporated into search terms rather than keeping a stale initial query.
                step_terms = " ".join([s.action for s in state.history] + ([decision.selected_tool] if decision.selected_tool else []))
                step_query = f"{goal} {step_terms}".strip() if step_terms else goal
                step_recall_ctx = self.memory.build_recall_context(step_query, limit=3) or recall_ctx

                context_prefix = ""
                if correction_guidance:
                    context_prefix += correction_guidance + "\n"
                if step_recall_ctx:
                    context_prefix += step_recall_ctx + "\n"
                if skill_ctx:
                    context_prefix += skill_ctx + "\n"

                last_step = state.history[-1] if state.history else None
                if last_step and any(err_kw in str(last_step.output).lower() for err_kw in ("error", "failed", "denied", "exception", "not found")):
                    context_prefix += (
                        f"[AUTO-CORRECTION GUIDANCE]: The previous action '{last_step.action}' resulted in: "
                        f"{str(last_step.output)[:300]}. Analyze this feedback, adjust your parameters or strategy, and proceed.\n"
                    )

                s2_prompt = context_prefix + state.to_system_two_prompt(
                    self.mcp.get_formatted_tool_list_for_system_two()
                )
                s2_images = (
                    [step_grid_path or step_screenshot_path]
                    if (step_grid_path or step_screenshot_path)
                    else None
                )
                if s2_images:
                    try:
                        s2_response = self.s2.generate_step(s2_prompt, images=s2_images)
                    except TypeError:
                        s2_response = self.s2.generate_step(s2_prompt)
                else:
                    s2_response = self.s2.generate_step(s2_prompt)
                s2_latency_total += s2_response.latency_ms
                if s2_response.is_mock:
                    s2_is_mock = True
                    s2_degraded_reason = s2_response.degraded_reason
                    # If this was not intentionally the mock provider, fail loudly rather than faking completion
                    if s2_degraded_reason and "mock provider — no real model" not in s2_degraded_reason:
                        state.is_completed = False
                        state.final_output = f"System 2 failure: {s2_degraded_reason}"
                        self._record_step(
                            state,
                            StepRecord(
                                step_index=step_idx,
                                step_type=StepType.SYSTEM_TWO_GENERATION,
                                action="system_two_failure",
                                action_input={"reason": s2_degraded_reason},
                                output=state.final_output,
                                latency_ms=s2_response.latency_ms,
                                tokens_used=0,
                            ),
                            active_callback,
                        )
                        break

                if s2_response.action == "system_two_incomplete":
                    self._record_step(
                        state,
                        StepRecord(
                            step_index=step_idx,
                            step_type=StepType.SYSTEM_TWO_GENERATION,
                            action="system_two_incomplete",
                            action_input=s2_response.args,
                            output=(
                                "Model provided thought but did not specify an 'action' or 'args'. "
                                "You must invoke a concrete MCP tool (e.g. patch_file, write_file, search_file) to make progress."
                            ),
                            latency_ms=s2_response.latency_ms,
                            tokens_used=s2_response.tokens_used,
                        ),
                        active_callback,
                    )
                    continue

                if s2_response.action == "finish_task":
                    state.is_completed = True
                    state.final_output = s2_response.generated_content or s2_response.thought
                    for p in state.plan:
                        if not p.completed:
                            p.completed = True
                            p.verified = True
                    self._record_step(
                        state,
                        StepRecord(
                            step_index=step_idx,
                            step_type=StepType.SYSTEM_TWO_GENERATION,
                            action="finish_task",
                            action_input=s2_response.args,
                            output=state.final_output,
                            latency_ms=s2_response.latency_ms,
                            tokens_used=s2_response.tokens_used,
                        ),
                        active_callback,
                    )
                    break
                else:
                    s2_action = s2_response.action
                    s2_args = s2_response.args
                    tool_def = self.mcp.get_tool(s2_action)

                    # --- SCHEMA VALIDATION: validate System 2 arguments too ---
                    # Wrong-typed or unknown arguments from System 2 must be rejected
                    # rather than executed, allowing the auto-correction guidance loop to kick in.
                    s2_arg_ok, s2_arg_err = validate_tool_args(tool_def, s2_args)
                    if not s2_arg_ok:
                        val_err_msg = f"Argument validation failed for '{s2_action}': {s2_arg_err}"
                        logger.warning(f"[Dispatcher] {val_err_msg}")
                        self._record_step(
                            state,
                            StepRecord(
                                step_index=step_idx,
                                step_type=StepType.SYSTEM_TWO_GENERATION,
                                action=s2_action,
                                action_input=s2_args,
                                output=val_err_msg,
                                latency_ms=s2_response.latency_ms,
                                tokens_used=s2_response.tokens_used,
                            ),
                            active_callback,
                        )
                        continue

                    if tool_def and tool_def.requires_approval:
                        approval, s2_args = self.broker.request_approval(
                            tool_name=s2_action,
                            args=s2_args,
                            risk_level=tool_def.risk_level,
                        )
                        if approval == ApprovalDecision.DENY:
                            self._record_step(
                                state,
                                StepRecord(
                                    step_index=step_idx,
                                    step_type=StepType.SYSTEM_TWO_GENERATION,
                                    action=s2_action,
                                    action_input=s2_args,
                                    output="[DENIED by user]",
                                    latency_ms=s2_response.latency_ms,
                                    tokens_used=s2_response.tokens_used,
                                    ),
                                active_callback,
                            )
                            continue

                    exec_result = self.mcp.execute_tool(s2_action, s2_args)
                    output_val = exec_result.output if exec_result.success else exec_result.error
                    total_step_lat = s2_response.latency_ms + exec_result.execution_time_ms
                    self._record_step(
                        state,
                        StepRecord(
                            step_index=step_idx,
                            step_type=StepType.SYSTEM_TWO_GENERATION,
                            action=s2_action,
                            action_input=s2_args,
                            output=output_val,
                            latency_ms=total_step_lat,
                            tokens_used=s2_response.tokens_used,
                        ),
                        active_callback,
                    )

                    # --- OUTCOME VERIFICATION (JevEvaluator) ---
                    is_verified, v_notes = self.evaluator.verify_step_outcome(
                        current_subgoal, s2_action, s2_args, output_val
                    )

                    # Physical visual verification via screen_diff
                    if (
                        is_verified
                        and s2_action in ("mouse_click", "key_press", "mouse_move")
                        and step_screenshot_path
                        and self.mcp.get_tool("screen_diff")
                    ):
                        try:
                            time.sleep(0.05)
                            post_ss = self.mcp.execute_tool("screenshot", {})
                            if post_ss.success and post_ss.output:
                                import json as _json
                                post_meta = _json.loads(post_ss.output)
                                post_path = post_meta.get("path")
                                if post_path:
                                    diff_res = self.mcp.execute_tool(
                                        "screen_diff",
                                        {
                                            "image_path_1": step_screenshot_path,
                                            "image_path_2": post_path,
                                        },
                                    )
                                    if diff_res.success and diff_res.output:
                                        diff_data = _json.loads(diff_res.output)
                                        if not diff_data.get("changed", True):
                                            is_verified = False
                                            v_notes = (
                                                f"Visual verification failed: screen diff detected no change "
                                                f"({diff_data.get('diff_percentage', '0.00%')}) after '{s2_action}'. "
                                                "The action did not alter the screen."
                                            )
                        except Exception as diff_err:
                            logger.debug(f"[Dispatcher] Visual screen_diff error: {diff_err}")

                    self._update_plan_progress(state, s2_action, is_verified, v_notes)
                    if not is_verified:
                        step_id = active_plan_step.step_id if active_plan_step else 0
                        verification_failures[step_id] = verification_failures.get(step_id, 0) + 1
                        correction_guidance = (
                            f"[VERIFICATION FAILED]: Step '{s2_action}' failed verification: {v_notes}. "
                            f"Subgoal was: '{current_subgoal}'. Correct your parameters or approach.\n"
                        )
                        if verification_failures[step_id] >= 3:
                            state.is_completed = False
                            state.final_output = (
                                f"Could not complete task: subgoal '{current_subgoal}' "
                                f"failed verification repeatedly: {v_notes}"
                            )
                            break
                    else:
                        correction_guidance = ""

                    if s2_action in ("mouse_click", "key_press"):
                        screen_sig = (s2_action, repr(s2_args), is_verified)
                        if screen_sig == last_signature and not is_verified:
                            repeat_count += 1
                        else:
                            repeat_count = 1
                        last_signature = screen_sig
                        if repeat_count >= 3:
                            stalled = True
                            state.is_completed = True
                            state.final_output = (
                                f"Stopped after {repeat_count} consecutive failed '{s2_action}' steps "
                                "with no visual screen change — the goal is not progressing."
                            )
                            logger.warning(
                                f"[Dispatcher] Stalled on repeated '{s2_action}' without screen change; ending run early."
                            )
                            break

        if not state.is_completed and not state.final_output:
            incomplete = [p.description for p in state.plan if not p.completed]
            if incomplete:
                state.final_output = (
                    f"Could not complete task within {max_steps} steps. "
                    f"Unfinished subgoals: {', '.join(incomplete)}"
                )
            else:
                state.final_output = f"Could not complete task within step budget ({max_steps} steps)."

        # Attribute any router time not yet billed to a step record.
        #
        # The fall-through to System 2 at step N computes router verification
        # latency, and that step can then end the run via `break` without a
        # matching record — so the cost never reaches `total_latency_ms`. That
        # made System 1 latency able to EXCEED the reported total run latency,
        # which is impossible: a subsystem cannot cost more time than the whole
        # run took.
        if self._pending_verification_latency_ms:
            if state.history:
                # Fold the remainder into the last record so totals reconcile.
                last = state.history[-1]
                last.latency_ms = round(
                    last.latency_ms + self._pending_verification_latency_ms, 3
                )
            else:
                self._record_step(
                    state,
                    StepRecord(
                        step_index=1,
                        step_type=StepType.SYSTEM_ONE_EVALUATION,
                        action="route_verification",
                        output=None,
                        latency_ms=round(self._pending_verification_latency_ms, 3),
                        tokens_used=0,
                    ),
                    active_callback,
                )
            self._pending_verification_latency_ms = 0.0

        # Reconcile the step sum against measured wall-clock time.
        #
        # Hand-maintained latency sums drift: a code path that forgets to add its
        # own elapsed time makes the run look faster than it was, which is the
        # same class of error as the hardcoded baselines this file used to print.
        # The wall clock is the one number that cannot be mislaid, so it wins.
        wall_clock_ms = (time.perf_counter() - run_started_at) * 1000
        step_sum_ms = state.total_latency_ms
        if wall_clock_ms > step_sum_ms and state.history:
            unaccounted = wall_clock_ms - step_sum_ms
            last = state.history[-1]
            last.latency_ms = round(last.latency_ms + unaccounted, 3)
            last.metadata["latency_reconciled_ms"] = round(unaccounted, 3)
            logger.debug(
                f"[Dispatcher] Reconciled {unaccounted:.1f}ms of unrecorded time "
                "into the final step so total latency reflects the wall clock."
            )


        # MEASURED TOTALS ONLY.
        #
        # Earlier revisions of this file multiplied the step count by hardcoded
        # constants (1500 tokens / 1200 ms per step) and reported the difference
        # as "token savings" and "speedup". That is not a measurement — it is a
        # number this file chose for itself, and it made every run look like a
        # 70-90% win regardless of what actually happened. Doing a real
        # comparison requires running the identical goal through a plain
        # single-model agent loop and diffing the results; until that exists,
        # this runtime reports only what it observed.
        total_steps = max(1, len(state.history))
        actual_tokens = state.total_tokens
        actual_latency = state.total_latency_ms

        baseline_tokens = None
        baseline_latency = None

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
                # No savings figure exists without a real baseline run.
                token_savings_pct=None,
                steps=[s.model_dump() for s in state.history],
            )

            # Auto-learn skill only if run completed cleanly without stalling
            # and auto_learn_skills is enabled in configuration.
            # Stalled runs and runs that exhausted max steps did not achieve the
            # goal, so synthesizing a skill would preserve a non-working sequence.
            should_learn = getattr(self.config, "auto_learn_skills", True)
            if state.is_completed and not stalled and should_learn:
                tool_seq = [s.action for s in state.history if s.action != "finish_task"]
                if tool_seq:
                    keywords = [w.lower() for w in goal.split() if len(w) > 3][:5]
                    clean_name = " ".join(keywords) if keywords else goal[:30]
                    self.memory.save_learned_skill(
                        name=clean_name,
                        intent_keywords=keywords or ["general"],
                        tool_sequence=tool_seq,
                    )
                    # Also synthesize a portable .SKILL.md file (Hermes-style)
                    try:
                        self.skills.synthesize_skill(
                            goal=goal,
                            tool_sequence=tool_seq,
                            outcome=state.final_output or "",
                        )
                    except Exception as se:
                        logger.warning(f"[Dispatcher] Could not synthesize skill file: {se}")

            # Persist durable facts to USER.md (cross-session user profile)
            if state.is_completed and not stalled:
                try:
                    tool_seq = [s.action for s in state.history if s.action != "finish_task"]
                    tool_desc = f" using {', '.join(sorted(set(tool_seq)))}" if tool_seq else ""
                    fact = f"Completed goal: '{goal}'{tool_desc}."
                    self.memory.update_user_profile(fact)
                except Exception as ue:
                    logger.debug(f"[Dispatcher] Could not update USER.md: {ue}")
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
            # Both values rounded from the same source so the documented
            # invariant "simulated_latency <= system_one_latency" holds exactly
            # rather than failing on sub-microsecond rounding noise.
            system_one_latency_ms=round(s1_latency_total, 3),
            system_two_latency_ms=s2_latency_total,
            tokens_used=actual_tokens,
            used_simulated_system_one=bool(getattr(self.s1, "force_simulation", False)),
            system_one_fallback_reason=getattr(self.s1, "simulation_reason", None) or None,
            simulated_latency_ms=round(simulated_s1_latency, 3),
            system_two_is_mock=s2_is_mock,
            system_two_degraded_reason=s2_degraded_reason,
            plan=state.plan,
        )

    def _check_fast_path_confidence(
        self, tool_name: str, descriptions: Dict[str, str]
    ) -> float:
        """Sibling-tool confidence used by `_compute_fast_path_confidence`.

        Ignore any sibling score that is not a real number: YAML's `yes`/`no`
        parse to bool, and `float(True)` is 1.0, which would silently become a
        perfect-confidence signal.
        """
        # Accumulate true router wall-clock time so the caller can bill this
        # verification call to System 1 latency instead of hiding it.
        started = time.perf_counter()
        try:
            decision = self.s1.evaluate_state_and_route(
                state_text=(
                    f"GOAL: {self._current_goal}\n"
                    f"CANDIDATE TOOL: {tool_name}\n"
                    "Decide which single tool to use next."
                ),
                tool_options=descriptions,
            )
        except Exception as e:
            logger.warning(f"[Dispatcher] Confidence vote for '{tool_name}' failed: {e}")
            self._verification_clock_ms = getattr(self, "_verification_clock_ms", 0.0) + (
                time.perf_counter() - started
            ) * 1000
            return 0.0

        self._verification_clock_ms = getattr(self, "_verification_clock_ms", 0.0) + (
            time.perf_counter() - started
        ) * 1000

        value = decision.confidence
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0.0
        return float(value)

    def _compute_fast_path_confidence(self, decision, state: AgentState) -> float:
        """Confidence that `decision.selected_tool` is the correct next tool.

        The confidence returned belongs to the WEAKEST link, not to the router's
        self-report. A tool is only trusted when it beats both controls:

        1. the decision's own confidence, and
        2. the confidence of the strongest rival declaration re-checked directly.

        Gate 2 exists because the aggregate is self-reported AND because tool
        descriptions differ in specificity: a narrowly-worded declaration such as
        "read a specific file" loses to "list a directory" on vague goals and
        gets outvoted. Re-checking the runner-up in isolation cancels that bias.
        """
        # The verification below costs real router calls; bill them to the step
        # that needed the check so System 1 latency reflects every model call.
        self._pending_verification_latency_ms = 0.0

        confidences = [float(decision.confidence or 0.0)]

        rival_tool = None
        rival_prob = 0.0
        rivals = []
        for name, prob in (decision.probabilities or {}).items():
            if name in (decision.selected_tool, "escalate_to_system_two", "finish_task"):
                continue
            if isinstance(prob, bool) or not isinstance(prob, (int, float)):
                continue
            rivals.append((float(prob), name))
        rivals.sort(key=lambda pair: (-pair[0], pair[1]))
        if rivals:
            rival_prob, rival_tool = rivals[0]

        if rival_tool and rival_prob > 0.0:
            before = getattr(self, "_verification_clock_ms", 0.0)
            score = self._check_fast_path_confidence(rival_tool, {rival_tool: ""})
            confidences.append(score)
            self._pending_verification_latency_ms = max(
                0.0, getattr(self, "_verification_clock_ms", 0.0) - before
            )

        return min(confidences)

    def _resolve_fast_path_tool(
        self, tool_name: str, tool_descriptions: Dict[str, str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve a router choice that names only a command, not a subcommand.

        The router cannot express `git add` or `git push` distinctly — a short
        hex subcommand has no English word to match on, so the choice comes back
        as bare `git` and every `git ...` tool is registered under that key.
        Rather than guessing (which would make routing depend on dict ordering),
        ask System 2 to disambiguate; if it cannot, execute nothing.
        """
        if tool_name in tool_descriptions:
            return tool_name, None

        candidates = [k for k in tool_descriptions if k.split() and k.split()[0] == tool_name]
        if not candidates:
            return None, f"Tool '{tool_name}' is not registered in MCP host."

        if len(candidates) == 1:
            return candidates[0], None

        chosen, error = self._select_argv0_tool(candidates, tool_descriptions)
        return chosen, error

    def _infer_default_args(self, tool_name: str, state: AgentState) -> Dict[str, Any]:
        """Derive tool arguments from the actual goal.

        NOTE: when the confidence gate is disabled via
        DUAL_AGENT_ALLOW_UNVERIFIED_FAST_PATH, this function is the only thing
        choosing a tool's arguments, and it CANNOT do so correctly in general —
        the values below are extracted from goal text by regex, which is a
        guess. The default path therefore refuses the fast path, because silently
        running `ls -la` or reading `pyproject.toml` for every goal produces
        plausible-looking success while never doing what the user asked.
        """
        if tool_name == "list_directory":
            path = self._extract_path(state.goal, default=".")
            return {"path": path}
        elif tool_name == "read_file":
            return {"path": self._extract_path(state.goal, default="")}
        elif tool_name == "run_shell_command":
            return {"command": self._extract_shell_command(state.goal)}
        return {}

    @staticmethod
    def _extract_path(goal: str, default: str) -> str:
        """Best-effort path extraction from goal text (regex, not understanding)."""
        for token in goal.split():
            cleaned = token.strip("'\"`,;()[]")
            if cleaned.startswith(("./", "/", "~/")) or "." in cleaned[1:]:
                if os.path.exists(os.path.expanduser(cleaned)):
                    return cleaned
        return default

    @staticmethod
    def _extract_shell_command(goal: str) -> str:
        """Best-effort shell command extraction from goal text.

        Only returns a command when the goal quotes one explicitly. Never
        invents one — an unrequested shell command executed on the user's
        machine is worse than admitting the goal is ambiguous.
        """
        match = re.search(r"[`\"']([^`\"']{2,})[`\"']", goal)
        return match.group(1) if match else ""

    def _select_argv0_tool(
        self, candidates: List[str], tool_descriptions: Dict[str, str]
    ) -> tuple:
        """Ask System 2 to choose between rival tools from the same argv[0].

        Choosing arbitrarily here would make routing depend on dict iteration
        order — the same goal could route to `git_add` on one run and
        `git_push` on the next. Ambiguity is escalated instead.
        """
        prompt = (
            "Multiple tools share the command name 'git'. Exactly one step is being\n"
            f"requested by this goal: {self._current_goal!r}\n\n"
            "Candidate tools:\n"
            + "\n".join(f"- {name}: {tool_descriptions[name]}" for name in candidates)
            + '\n\nReply with JSON only: {"tool": "<exact tool name>"}\n'
        )
        try:
            response = self.s2.generate_step(prompt)
            chosen = (response.args or {}).get("tool")
            if chosen in candidates:
                return chosen, None
            # Tolerate a provider that answers in plain text.
            for name in candidates:
                if name in (response.thought or "") or name in (response.generated_content or ""):
                    return name, None
        except Exception as e:
            logger.warning(f"[Dispatcher] argv[0] disambiguation failed: {e}")
        return None, (
            f"Ambiguous command '{candidates[0].split()[0]}' matched "
            f"{len(candidates)} tools ({', '.join(candidates)}) and no unambiguous "
            "choice could be made. No tool was executed."
        )
