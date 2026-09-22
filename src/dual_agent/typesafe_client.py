"""TypeSafe AI Jev (System 1) integration layer."""

from __future__ import annotations
import math
import os
import re
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

    # ------------------------------------------------------------------
    # Offline router: evidence vocabularies.
    #
    # WHY this exists at all: without an API key the router is the only thing
    # deciding what the agent does, so a stub that matches a single word of the
    # tool name makes every offline run useless — `list_directory` wins any goal
    # containing "directory" and the loop never terminates. These tables let the
    # router weigh ALL candidates on evidence actually present in the state.
    # It is still a HEURISTIC, not a model: it reasons over token overlap and
    # argument shapes, it does not understand the goal.
    # ------------------------------------------------------------------

    # Intent verbs grouped into action families. A tool whose name/description
    # belongs to the same family as the verb used in the goal is a better fit
    # than one that merely shares a noun with it.
    _SIM_INTENT_VERBS: Dict[str, tuple] = {
        "list": ("list", "ls", "dir", "directory", "enumerate", "inventory", "browse", "inspect"),
        "read": ("read", "cat", "open", "show", "display", "print", "view", "load", "contents"),
        "write": ("write", "save", "create", "append", "store", "output", "generate", "make"),
        "run": ("run", "execute", "invoke", "shell", "command", "terminal", "bash", "script"),
        "find": ("find", "search", "grep", "locate", "lookup", "query", "match"),
    }

    # Nouns and verbs that mark work no registered tool performs: the answer has
    # to be written, not produced by a tool call. Both forms are listed because
    # a goal says "summarize the file" or "write a summary" and either is the
    # same request.
    _SIM_SYNTHESIS_VERBS: tuple = (
        "summarize", "summarise", "summarization", "summarise", "summary", "summaries",
        "synthesize", "synthesise", "synthesis",
        "draft", "explain", "explanation", "describe", "description", "narrate",
        "refactor", "rewrite", "rephrase", "translate", "translation",
        "brainstorm", "design", "redesign", "compose", "critique", "review",
        "analyze", "analyse", "analysis", "reason", "plan", "outline", "poem",
        "story", "essay", "report on", "blog", "article", "prose",
    )

    # Phrases that require content generation regardless of the verb used.
    _SIM_GENERATION_NOUNS: tuple = (
        "summary", "summaries", "explanation", "poem", "essay", "story",
        "design doc", "documentation", "readme", "blog post", "write-up", "report",
    )

    # Action families each tool name maps to. Matching is by token, so
    # `list_directory` scores for "list" but not for an unrelated verb.
    _SIM_TOOL_FAMILIES: Dict[str, frozenset] = {
        "list_directory": frozenset({"list"}),
        "read_file": frozenset({"read"}),
        "write_file": frozenset({"write"}),
        "run_shell_command": frozenset({"run"}),
        "search_files": frozenset({"find"}),
    }

    # Extensions that name a FILE. A goal mentioning one of these is a file
    # operation; a directory-oriented tool should lose to a file-oriented one.
    _SIM_FILE_EXTENSIONS: frozenset = frozenset({
        "py", "txt", "md", "json", "yaml", "yml", "toml", "ini", "cfg", "csv", "tsv",
        "js", "ts", "jsx", "tsx", "html", "css", "sh", "bash", "zsh", "sql", "xml",
        "log", "rst", "env", "lock", "go", "rs", "java", "rb", "c", "h", "cpp", "hpp",
    })

    # Paths that name a DIRECTORY even when they contain no extension.
    _SIM_DIRECTORY_TOKENS: tuple = (
        "directory", "directories", "folder", "folders", "dir", "workspace",
        "listing", "contents of", "ls -la", "tree",
    )

    # Words that mark a requirement the agent cannot satisfy with a tool call,
    # so the slow path (System 2) has to be woken.
    _SIM_REQUIRES_GENERATION_PHRASES: tuple = (
        "write a", "write me", "write up", "compose a", "draft a", "generate a",
        "generate the", "create a script", "produce a", "creative",
    )

    # Recent-action output that reports success rather than an error or a
    # refusal. Used for the termination decision, never to claim a model verdict.
    _SIM_SUCCESS_TOKENS: tuple = (
        "success", "completed", "created", "wrote", "written", "updated",
        "entries", "exitcode: 0", "found", "read", "listed", "saved",
    )

    # Log lines the router can use even when there is no tool output yet.
    _SIM_COMPLETION_MARKERS: tuple = (
        "goal achieved", "all steps completed", "task complete", "task completed",
        "goal satisfied", "no further action",
    )

    @classmethod
    def _sim_route_offline(
        cls,
        state_text: str,
        choices: Dict[str, str],
    ) -> Dict[str, Any]:
        """Deterministic evidence-based routing over the parsed state.

        HEURISTIC, NOT A MODEL. Every score below is an additive weight derived
        from string evidence in `state_text`, the candidate tool's own
        description, and the argument names it requires. Newer candidate tools
        may be declared with schemas _SIM_TOOL_FAMILIES does not know about, so
        the family lookup falls back to matching the tool's own name tokens
        against the shared vocabulary instead of guessing a family from dict
        position.

        Returns the selected option, a probability distribution that sums to
        ~1.0, is_terminal and needs_generation.
        """
        state_lower = state_text.lower()
        goal, recent_actions, variables = cls._sim_parse_state(state_text)

        # -- Parse phase: pull signals out of the state text --------------
        words = set(re.findall(r"[a-z0-9_]+", goal.lower()))

        # A path/extension token in the goal is the strongest single signal
        # available offline: it distinguishes "read config.yaml" from "list the
        # directory" without any language understanding.
        file_like_tokens: List[str] = []
        directory_like_tokens: List[str] = []
        for raw in re.split(r"[\s,;'\"]+", goal):
            token = raw.strip("`'\"()[]<>")
            if not token or token in (".", "..", "/"):
                continue
            if cls._sim_names_directory(token):
                directory_like_tokens.append(token)
                continue
            extension = token.rsplit(".", 1)[-1].lower() if "." in token[1:] else ""
            if extension and extension in cls._SIM_FILE_EXTENSIONS:
                file_like_tokens.append(token)
            elif token.startswith(("./", "/", "~/")) or "/" in token:
                # A path with no extension is ambiguous on purpose: it is not
                # counted as a file, so it cannot push the router toward a file
                # reader on its own.
                directory_like_tokens.append(token)

        goal_names_directory = bool(directory_like_tokens) or any(
            tok in goal.lower() for tok in cls._SIM_DIRECTORY_TOKENS
        )

        goal_verbs = {
            verb for verb, synonyms in cls._SIM_INTENT_VERBS.items()
            if any(word in words for word in synonyms)
        }

        # Which verbs the agent has ALREADY exercised, and which tools it has
        # already run to a successful output. Either is evidence that the
        # remaining work is something ELSE: re-issuing a completed action is the
        # behaviour that makes an offline run loop until max_steps.
        completed_verbs: set = set()
        completed_tools: set = set()
        for _action, output in recent_actions:
            if not cls._sim_output_is_productive(output):
                continue
            completed_tools.add(_action.strip().lower())
            action_tokens = set(re.findall(r"[a-z0-9_]+", _action.lower()))
            for verb, synonyms in cls._SIM_INTENT_VERBS.items():
                if any(word in action_tokens for word in synonyms):
                    completed_verbs.add(verb)

        # -- Scoring phase: score every candidate tool --------------------
        # Scores are additive weights, deliberately small and few so the ordering
        # is auditable by hand. Ties are broken by name (below), never by dict
        # order, so the result does not depend on insertion order.
        regular_tools = [
            name for name in choices
            if name not in ("escalate_to_system_two", "finish_task")
        ]
        raw_scores: Dict[str, float] = {}
        for tool_name in regular_tools:
            description = str(choices.get(tool_name, "") or "")
            families = cls._sim_tool_families(tool_name, description)
            name_tokens = set(re.findall(r"[a-z0-9_]+", tool_name.lower()))
            desc_words = set(re.findall(r"[a-z0-9_]+", description.lower()))
            required_args = cls._sim_required_args(description)

            score = 0.0

            # (a) Verb agreement between the goal and the tool's action family.
            for verb in goal_verbs:
                if verb in families:
                    score += 2.5
                elif verb in completed_verbs:
                    pass  # already done elsewhere; do not reward the repetition
                else:
                    score -= 0.4

            # (b) Tool name evidence from the goal text (word or whole name).
            if tool_name.lower() in goal.lower():
                score += 3.0
            if name_tokens & words:
                score += 1.5

            # (c) Description evidence from the goal text.
            overlap = desc_words & words
            if overlap:
                score += min(1.5, 0.5 * len(overlap))

            # (d) Argument-shape evidence. A tool that REQUIRES `path` is a fit
            # for anything naming a file or path — but only when the goal agreed
            # that a path was in play. Paying every `path` tool equally on a
            # bare directory goal would hand the win to whichever name sorted
            # first, which is the dict-order bug in a new costume.
            if "path" in required_args and (file_like_tokens or goal_names_directory):
                score += 1.5
            if "command" in required_args and "run" in goal_verbs:
                score += 1.5
            elif "command" in required_args and not goal_verbs:
                score -= 0.25
            if "content" in required_args and "write" in goal_verbs:
                score += 1.5

            # (e) File vs directory discrimination. Without this, a tool whose
            # name merely contains the goal's noun wins on a goal that asked for
            # a different operation entirely.
            if file_like_tokens and not goal_names_directory:
                if "read" in families:
                    score += 1.5
                if "list" in families:
                    score -= 1.5
            if goal_names_directory:
                if "list" in families:
                    score += 1.5
                if "read" in families and not file_like_tokens:
                    score -= 1.0
                if "run" in families:
                    score -= 0.5

            # (f) The verb has already been satisfied — the goal is not "do it
            # again", so stop paying for the tool that just ran successfully.
            if families & completed_verbs:
                score -= 1.0

            # (g) Hard anti-repetition rule. A tool that already produced a
            # successful output cannot be re-selected: the work it does exists.
            # Without this the router re-runs `list_directory` forever whenever
            # the file listing it returned removes the goal's remaining evidence
            # (its description words no longer match), and offline runs only end
            # via the dispatcher's stall detector.
            if tool_name.strip().lower() in completed_tools:
                score -= 6.0

            raw_scores[tool_name] = score

        # -- Control-option scoring ---------------------------------------
        # escalate_to_system_two is the standing fallback for work no tool can
        # do; finish_task wins only when the evidence says the work is done.
        escalate_score = 0.4  # baseline: escalation is always admissible
        if any(word in words for word in cls._SIM_SYNTHESIS_VERBS):
            escalate_score += 2.0
        if not regular_tools:
            escalate_score += 2.0

        # One more file-vs-directory discriminator, now that the goals are
        # known: with a file named in the goal the reader is at 2.5 and the
        # lister at 1.5, which is only a 1-point gap. Punishing the lister
        # directly makes the two orderings distinguishable rather than a coin
        # flip dressed up as a probability.
        if file_like_tokens and not goal_names_directory:
            for name in regular_tools:
                if "list" in cls._sim_tool_families(name, str(choices.get(name, "") or "")):
                    raw_scores[name] -= 0.5

        is_terminal = cls._sim_is_terminal(
            state_text=state_text,
            goal=goal,
            goal_verbs=goal_verbs,
            completed_verbs=completed_verbs,
            recent_actions=recent_actions,
            has_regular_tools=bool(regular_tools),
        )
        finish_score = 3.0 if is_terminal else -2.0
        if is_terminal:
            # A finished goal must WIN, not merely be plausible: any tool left
            # near it in probability can be picked by a one-point score shift
            # and re-run a step that already happened. So the finished branch
            # additionally caps every other option below finish_task.
            escalate_score = min(escalate_score, 0.3)
            for name in regular_tools:
                raw_scores[name] = min(raw_scores[name], 1.5)

        raw_scores["escalate_to_system_two"] = escalate_score
        raw_scores["finish_task"] = finish_score

        # -- Softmax over the raw weights ---------------------------------
        # Deterministic: plain arithmetic, fixed iteration order (the caller
        # passes tool_options in a stable order), no sampling anywhere.
        # Sort by (-score, name): the name is the tie-break so the outcome never
        # depends on the order tool_options happened to be built in.
        selected_tool = sorted(raw_scores, key=lambda name: (-raw_scores[name], name))[0]
        max_score = raw_scores[selected_tool]
        exponentials = {
            name: math.exp(int(round(score - max_score)) * cls._SIM_SCORE_TEMPERATURE)
            for name, score in raw_scores.items()
        }
        total = sum(exponentials.values()) or 1.0
        probabilities = {name: value / total for name, value in exponentials.items()}

        if is_terminal:
            # A finished goal is not a preference between options, it is a
            # statement that the loop is over. Reporting 0.56 here would hand a
            # caller a distribution in which a tool still looks plausible, so
            # the finished branch reports near-certainty instead.
            probabilities = {
                name: (cls._SIM_TERMINAL_CONFIDENCE if name == "finish_task"
                       else (1.0 - cls._SIM_TERMINAL_CONFIDENCE) / max(1, len(probabilities) - 1))
                for name in probabilities
            }

        # -- Generation need ----------------------------------------------
        # System 2 is required when the goal asks for text/code content that has
        # to be WRITTEN. A registered tool can still perform the write; what no
        # tool can do is produce the content. Matching is on whole words or
        # multi-word phrases — substring matching would fire on "describe" inside
        # an unrelated word and on file names.
        lowered_goal = goal.lower()
        needs_generation = any(word in words for word in cls._SIM_SYNTHESIS_VERBS) or any(
            noun in lowered_goal for noun in cls._SIM_GENERATION_NOUNS
        )
        if not needs_generation:
            needs_generation = bool(goal_verbs & {"write", "run"}) and any(
                phrase in lowered_goal for phrase in cls._SIM_REQUIRES_GENERATION_PHRASES
            )
        # A goal whose work is already done does not need generation.
        if is_terminal and selected_tool == "finish_task":
            needs_generation = False

        if not is_terminal and selected_tool == "finish_task":
            # finish_task can only be selected by the terminal branch above;
            # this guards against a future weight change silently terminating
            # a live goal.
            is_terminal = True

        return {
            "selected_tool": selected_tool,
            "probabilities": probabilities,
            "is_terminal": is_terminal,
            "needs_generation": needs_generation,
            "variables": variables,
        }

    # Temperature is applied to the ROUNDED score gap, not the raw difference:
    # raw gaps are exact integers, and rounding first means float noise in the
    # arithmetic can never reorder two candidates that differ by a whole point.
    _SIM_SCORE_TEMPERATURE = 0.8

    # Probability mass the finished branch gives `finish_task`. Confidence is a
    # gated quantity downstream, so a terminal decision is reported as the
    # near-certainty it is rather than as one option among several.
    _SIM_TERMINAL_CONFIDENCE = 0.95

    @staticmethod
    def _sim_parse_state(state_text: str) -> tuple:
        """Split the compact state blob into (goal, recent_actions, variables).

        The dispatcher's `AgentState.to_system_one_state` emits:
            GOAL: <goal>
            CURRENT STEP: n/max
            RECENT ACTIONS: Step i: <action> -> <truncated output> | Step j: ...
            VARIABLES: ['a', 'b']

        The dispatcher is deliberately NOT imported here — importing it would be
        a circular import, and the router must keep working against any caller
        that produces this shape, including tests and the Hermes middleware.
        """
        goal = ""
        actions_blob = ""
        variables: List[str] = []
        for line in (state_text or "").splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("GOAL:"):
                goal = stripped[len("GOAL:"):].strip()
            elif upper.startswith("RECENT ACTIONS:"):
                actions_blob = stripped[len("RECENT ACTIONS:"):].strip()
            elif upper.startswith("VARIABLES:"):
                variables = re.findall(r"[\w.\-/]+", stripped[len("VARIABLES:"):])
        if not goal:
            # No labelled GOAL line (e.g. the middleware passes free text).
            # Treat the whole blob as the goal rather than routing on nothing.
            goal = state_text or ""

        recent_actions: List[tuple] = []
        if actions_blob and actions_blob.lower() != "no prior actions.":
            for chunk in actions_blob.split(" | "):
                # "Step 3: read_file -> <output>"; the arrow separates the action
                # from its truncated output.
                step_body = re.sub(r"^\s*step\s*\d+\s*:\s*", "", chunk, flags=re.IGNORECASE)
                if "->" in step_body:
                    action, output = step_body.split("->", 1)
                else:
                    action, output = step_body, ""
                recent_actions.append((action.strip(), output.strip()))
        return goal, recent_actions, variables

    @classmethod
    def _sim_required_args(cls, description: str) -> set:
        """Argument names a tool requires, recovered from its description.

        `get_tool_descriptions()` collapses an MCP tool to a single description
        string, so the schema is not available as a structure. The built-in
        descriptions name their arguments ("Read contents of a text file",
        "Shell command to execute"), so match on those names as words instead of
        inventing a schema parser for a format that does not exist here.
        """
        arg_words = set(re.findall(r"[a-z_]+", description.lower()))
        return arg_words & {"path", "command", "content", "args", "text", "query", "url"}

    @classmethod
    def _sim_tool_families(cls, tool_name: str, description: str) -> set:
        """Which action families a candidate tool belongs to."""
        if tool_name in cls._SIM_TOOL_FAMILIES:
            return set(cls._SIM_TOOL_FAMILIES[tool_name])
        families: set = set()
        tokens = set(re.findall(r"[a-z0-9_]+", f"{tool_name} {description}".lower()))
        for verb, synonyms in cls._SIM_INTENT_VERBS.items():
            if tokens & set(synonyms):
                families.add(verb)
        return families

    @staticmethod
    def _sim_names_directory(token: str) -> bool:
        """True when a path-looking token is a directory rather than a file.

        A trailing separator is decisive; otherwise a bare name such as
        "notes" is treated as a file-like target only when it carries an
        extension. Ambiguity is resolved toward "not a file" so a goal that
        mentions a directory is never routed to a file reader.
        """
        cleaned = token.strip().strip("`'\"()[]<>")
        if not cleaned:
            return False
        if cleaned.endswith(("/", "\\")):
            return True
        if "." in cleaned[1:]:
            return False
        return True

    @staticmethod
    def _sim_output_is_productive(output: str) -> bool:
        """True when a step's output shows real work rather than an error."""
        if not output:
            return False
        lowered = output.lower()
        if any(marker in lowered for marker in ("error", "denied", "does not exist", "not found")):
            return False
        return any(token in lowered for token in JevSystemOneClient._SIM_SUCCESS_TOKENS)

    @classmethod
    def _sim_is_terminal(
        cls,
        state_text: str,
        goal: str,
        goal_verbs: set,
        completed_verbs: set,
        recent_actions: List[tuple],
        has_regular_tools: bool,
    ) -> bool:
        """Decide whether the run is finished, from the state text alone.

        HEURISTIC. Three independent signals, in rough order of strength:

        1. The state says so in words ("goal achieved", an explicit completion
           log line).
        2. Every verb the goal asks for has already been performed successfully
           — the work exists, so repeating it is the loop-until-max_steps bug.
        3. The last two steps repeated the same action AND the same output:
           no progress is being made, which is a reason to stop, not to retry.
        """
        if not recent_actions:
            return False

        state_lower = (state_text or "").lower()
        if any(marker in state_lower for marker in cls._SIM_COMPLETION_MARKERS):
            return True

        productive = [pair for pair in recent_actions if cls._sim_output_is_productive(pair[1])]
        if not productive:
            return False

        # Signal 3: the same action produced the same output twice — stalled.
        for (prev_action, prev_output), (action, output) in zip(productive, productive[1:]):
            if prev_action == action and prev_output == output and output:
                return True

        if not has_regular_tools:
            # Nothing left to call but the goal has already produced usable
            # output: escalate is the only remaining action, and it has been
            # reached because the tools ran out, not because work is pending.
            return True

        if goal_verbs and goal_verbs.issubset(completed_verbs):
            return True

        return False

    def _call_simulated_jev(
        self,
        state_text: str,
        choices: Dict[str, str],
        start_time: float,
    ) -> JevDecision:
        """Local heuristic router used when the live Jev model is unavailable.

        IMPORTANT: this is a deterministic evidence-scoring heuristic, NOT the
        Jev model. It reads the state text, scores every candidate tool against
        the goal's verbs, file/path tokens, the tool's description and the
        arguments it needs, and returns a softmax over those scores. It does not
        understand language, and its confidence is an artefact of the score gap
        rather than a calibrated probability of correctness.

        The sleep below is a SIMULATION ARTIFACT emulating a network roundtrip so
        local benchmarks have a realistic latency profile. It is not model
        latency and must never be reported as such. Every decision this returns
        carries `simulated=True` and a `fallback_reason` saying why the live
        model was not used.
        """
        time.sleep(0.012)  # Simulation artifact: emulates a Jev roundtrip. Not real latency.
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        route = self._sim_route_offline(state_text, choices)
        selected = route["selected_tool"]
        probabilities = route["probabilities"]
        confidence = float(probabilities.get(selected, 0.0))

        return JevDecision(
            selected_tool=selected,
            confidence=confidence,
            probabilities=probabilities,
            is_terminal=route["is_terminal"],
            needs_generation=route["needs_generation"],
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
