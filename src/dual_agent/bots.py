"""Multi-bot identity layer.

A *bot* here is an identity, not a config bundle: a name, a role, a model
preference, a persona fragment, and an explicit allow-list of the MCP tools
that identity may call. `team_manifest.py` exports servers and skills; it does
not model who is answering. This module models who is answering.

Everything is additive and opt-in. With no `bots.json` on disk the registry is
empty, `registry.enabled` is False, `wrap_mcp_host()` returns the host it was
given unchanged, and every existing entry point behaves exactly as it did
before this module existed.

Config conventions are the same ones `config.py` already uses:
  - the file lives under `memory.get_default_data_dir()` (honours DUAL_AGENT_HOME),
  - it is written 0600,
  - the file is the source of truth; an environment variable only overrides it
    when it is deliberately set.

Two bounds keep a bot-to-bot conversation from running forever: `max_hops`
per hand-off chain and `max_depth` for nesting. Both are hard-stopped and the
stop is reported as a stop, never silently swallowed.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator

from dual_agent.memory import get_default_data_dir
from dual_agent.permission_broker import ApprovalDecision, PermissionBroker

logger = logging.getLogger(__name__)

# Hand-off bounds. A bot that asks another bot which asks another bot ... must
# terminate. 8 hops is generous for a real org chart and cheap to hit.
DEFAULT_MAX_HOPS = 8
DEFAULT_MAX_DEPTH = 4

BOTS_DIR_NAME = "bots"
BOTS_FILE_NAME = "bots.json"


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


class BotProfile(BaseModel):
    """Declarative identity for one bot.

    `allowed_tools` is an ALLOW-LIST, deliberately not a deny-list: a tool that
    is not named here is not callable by this bot, including tools registered
    after the profile was loaded.
    """

    id: str
    display_name: str
    role: str
    model: Optional[str] = None
    provider: Optional[str] = None
    persona: str = ""
    allowed_tools: List[str] = Field(default_factory=list)
    # Free-text keywords folded into routing. Kept separate from `role` so the
    # human-readable title stays human-readable.
    routing_keywords: List[str] = Field(default_factory=list)
    # A bot with no persona of its own falls back to the caller-supplied one.
    inherit_default_persona: bool = True

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", v or ""):
            raise ValueError(
                f"bot id {v!r} must be lowercase alphanumeric with - or _ "
                "(it is used as a lookup key and appears in refusal messages)"
            )
        return v

    def allows_tool(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@dataclass
class RoutingDecision:
    """Which bot answers, and the honest reason it was picked.

    `method` is never "ml" or "model". The current implementation is a
    deterministic keyword scorer; the field exists so a future scorer can say
    what it is without this one pretending to be it.
    """

    bot_id: Optional[str]
    reason: str
    score: float = 0.0
    method: str = "keyword_heuristic"
    candidates: List[Tuple[str, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "reason": self.reason,
            "score": self.score,
            "method": self.method,
            "candidates": [{"bot_id": b, "score": s} for b, s in self.candidates],
        }


def _tokenize(text: str) -> List[str]:
    """Lowercase word/digit tokens. No stemming — approximate is fine here."""
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t]


def _keyword_score(profile: BotProfile, text: str) -> Tuple[float, List[str]]:
    """Score one profile against a message.

    Deliberately simple and deterministic: exact token matches on the bot's
    role words and routing keywords, plus substring matches on multi-word
    routing phrases. Weights are arbitrary constants chosen to make "a keyword
    the operator wrote on purpose" outrank "a word that happened to be in the
    title". They are not calibrated against anything and are not a measure of
    correctness.
    """
    tokens = set(_tokenize(text))
    haystack = (text or "").lower()
    hits: List[str] = []
    score = 0.0

    for kw in profile.routing_keywords:
        k = str(kw).strip().lower()
        if not k:
            continue
        if " " in k:
            if k in haystack:
                score += 3.0
                hits.append(k)
        elif k in tokens:
            score += 2.0
            hits.append(k)

    role_words = set()
    for word in _tokenize(profile.role):
        # Generic org-chart filler carries no signal and would make every bot
        # match every message that mentions "agent" or "specialist".
        if len(word) > 2 and word not in _ROLE_STOPWORDS:
            role_words.add(word)
    for word in sorted(role_words & tokens):
        score += 1.0
        hits.append(word)

    return score, hits


_ROLE_STOPWORDS = {
    "and", "the", "for", "with", "agent", "specialist", "developer", "engineer",
    "lead", "manager", "expert", "assistant", "bot", "team", "senior", "junior",
    "operations", "operator",
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class BotRegistry:
    """Loads bot profiles from one file under the data dir.

    Empty registry == feature off. That is the backward-compatibility contract:
    `only_allow_listed_tools` is False and `wrap_mcp_host` is a no-op until
    someone actually adds bots.
    """

    def __init__(
        self,
        profiles: Optional[Sequence[BotProfile]] = None,
        source_path: Optional[str] = None,
        default_bot_id: Optional[str] = None,
        max_hops: int = DEFAULT_MAX_HOPS,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ):
        self._profiles: Dict[str, BotProfile] = {}
        for p in profiles or []:
            self.add(p)
        self.source_path = source_path
        self.default_bot_id = default_bot_id
        self.max_hops = int(max_hops)
        self.max_depth = int(max_depth)

    # -- construction ---------------------------------------------------

    @classmethod
    def load(cls, path: Optional[str] = None) -> "BotRegistry":
        """Read bots.json. A missing or unparseable file yields an empty registry.

        Missing is normal (feature unused). Unreadable or malformed is NOT
        silently swallowed — it is logged as an error at the point of failure,
        because a typo in bots.json silently disabling every tool allow-list
        would be the worst kind of fallback.
        """
        path = path or get_bot_config_path()
        if not os.path.isfile(path):
            return cls(source_path=None)

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logger.error(
                f"[BotRegistry] Could not parse {path}: {e}. No bot profiles "
                "loaded, so per-bot tool allow-lists are NOT in force."
            )
            return cls(source_path=None)

        profiles_raw = raw.get("bots", raw) if isinstance(raw, dict) else raw
        if not isinstance(profiles_raw, list):
            logger.error(
                f"[BotRegistry] {path} must contain a list under 'bots' or be a "
                "list of profiles. No bot profiles loaded."
            )
            return cls(source_path=None)

        profiles: List[BotProfile] = []
        for entry in profiles_raw:
            try:
                profiles.append(BotProfile(**entry))
            except Exception as e:
                logger.error(f"[BotRegistry] Skipping invalid profile {entry!r}: {e}")

        settings = raw if isinstance(raw, dict) else {}
        # `default_bot` is the routing fallback when no role matches.
        default_bot = settings.get("default_bot") if isinstance(settings, dict) else None
        if default_bot and default_bot not in {p.id for p in profiles}:
            logger.warning(
                f"[BotRegistry] default_bot={default_bot!r} in {path} is not a "
                "known bot id; ignoring it."
            )
            default_bot = None

        # The file is the source of truth; the env var only wins when the
        # operator set it on purpose (same rule as config.py's _resolve).
        env_default = os.environ.get("DUAL_AGENT_DEFAULT_BOT")
        if env_default:
            if env_default in {p.id for p in profiles}:
                default_bot = env_default
            else:
                logger.warning(
                    f"[BotRegistry] DUAL_AGENT_DEFAULT_BOT={env_default!r} is not "
                    "a known bot id; ignoring it."
                )

        return cls(
            profiles=profiles,
            source_path=path,
            default_bot_id=default_bot,
            max_hops=int(settings.get("max_hops", DEFAULT_MAX_HOPS)),
            max_depth=int(settings.get("max_depth", DEFAULT_MAX_DEPTH)),
        )

    # -- access ---------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True only when the operator actually registered bots."""
        return bool(self._profiles)

    @property
    def only_allow_listed_tools(self) -> bool:
        """Whether tool calls must be inside the answering bot's allow-list."""
        return self.enabled

    def add(self, profile: BotProfile) -> None:
        if profile.id in self._profiles:
            raise ValueError(f"duplicate bot id {profile.id!r}")
        self._profiles[profile.id] = profile

    def get(self, bot_id: str) -> Optional[BotProfile]:
        return self._profiles.get(bot_id)

    def all(self) -> List[BotProfile]:
        return list(self._profiles.values())

    def team_map(self) -> List[Dict[str, Any]]:
        """Renderable view of the team: name, role, model, granted tools."""
        return [
            {
                "id": p.id,
                "display_name": p.display_name,
                "role": p.role,
                "provider": p.provider,
                "model": p.model,
                "tools": list(p.allowed_tools),
            }
            for p in self._profiles.values()
        ]

    # -- routing --------------------------------------------------------

    def route(self, message: str) -> RoutingDecision:
        """Pick the bot whose declared role/keywords best match `message`.

        HEURISTIC, by design and in name: a deterministic keyword/role-token
        scorer. It is not trained, not a model, and makes no claim to accuracy.
        It exists so routing is auditable and reproducible rather than a black
        box; the returned RoutingDecision carries the input text, the winner
        and the reason so a caller can disagree with it.
        """
        if not self._profiles:
            return RoutingDecision(
                bot_id=None,
                reason="no bots configured; single default agent handles this message",
                method="none",
            )

        scored = sorted(
            ((p.id, *_keyword_score(p, message)) for p in self._profiles.values()),
            key=lambda t: (-t[1], t[0]),
        )
        candidates = [(bid, sc) for bid, sc, _ in scored]
        top_id, top_score, hits = scored[0]

        if top_score <= 0:
            fallback = self.default_bot_id or top_id
            return RoutingDecision(
                bot_id=fallback,
                reason=(
                    f"heuristic found no role keyword in the message; "
                    f"{'configured default_bot' if self.default_bot_id else 'lowest-id bot'} "
                    f"{fallback!r} selected"
                ),
                score=0.0,
                candidates=candidates,
            )

        profile = self._profiles[top_id]
        return RoutingDecision(
            bot_id=top_id,
            reason=(
                f"role {profile.role!r} matched message keywords "
                f"{sorted(set(hits))} (score {top_score:g})"
            ),
            score=top_score,
            candidates=candidates,
        )

    # -- permissions ----------------------------------------------------

    def check_tool(self, bot_id: Optional[str], tool_name: str) -> Tuple[bool, Optional[str]]:
        """Is `tool_name` inside `bot_id`'s allow-list?

        Returns (allowed, refusal_message). The refusal names both the bot and
        the tool — a bare "denied" leaves the operator guessing which grant is
        missing.
        """
        if not self.only_allow_listed_tools:
            return True, None
        profile = self._profiles.get(bot_id) if bot_id else None
        if profile is None:
            known = ", ".join(sorted(self._profiles)) or "(none)"
            return False, (
                f"Refused: tool '{tool_name}' was not called by a known bot "
                f"(unknown bot id {bot_id!r}). Registered bots: {known}."
            )
        if profile.allows_tool(tool_name):
            return True, None
        granted = ", ".join(sorted(profile.allowed_tools)) or "(none)"
        owners = sorted(
            p.id for p in self._profiles.values() if p.allows_tool(tool_name)
        )
        owners_note = (
            f" Bots that do hold '{tool_name}': {', '.join(owners)}."
            if owners
            else f" No bot in this registry holds '{tool_name}'."
        )
        return False, (
            f"Refused: bot '{profile.id}' ({profile.display_name}, "
            f"{profile.role}) may not call tool '{tool_name}'. "
            f"Its allow-list is: {granted}.{owners_note} "
            f"Add '{tool_name}' to bot '{profile.id}' in bots.json to grant it."
        )

    def wrap_mcp_host(self, mcp_host: Any, bot_id: str = "") -> Any:
        """Return a host whose tool surface is filtered to one bot's grants.

        The wrapper is per-bot, so it must be built for the bot that is
        answering. `bot_id` is required: without it there is no allow-list to
        enforce and every call would be refused, which looks like a broken tool
        host rather than a missing grant.

        With no bots registered this returns the original host object itself
        (identity, not a copy), which is what makes the no-bots path
        byte-for-byte the old behaviour.
        """
        if not self.only_allow_listed_tools:
            return mcp_host
        if not bot_id:
            raise ValueError(
                "wrap_mcp_host() needs the answering bot's id; without it there "
                "is no allow-list to enforce. Use apply_bot_profile() on a "
                "dispatcher, or pass bot_id=<BotProfile.id>."
            )
        return BotScopedMCPHost(mcp_host, self, bot_id=bot_id)


# ---------------------------------------------------------------------------
# Tool-surface wrapper
# ---------------------------------------------------------------------------


class BotScopedMCPHost:
    """A view of an MCPHost restricted to one bot's allow-list.

    Duck-types the MCPHost surface the dispatcher uses (`get_tool`,
    `list_tools`, `get_tool_descriptions`, `get_formatted_tool_list_for_system_two`,
    `execute_tool`, `register_tool`) and delegates to the real host.

    Enforcement is TWO-SIDED on purpose:
      1. Filtered listings mean the router and System 2 are never offered a tool
         this bot may not call, so the disallowed tool is not even reachable.
      2. `execute_tool` for a tool outside the allow-list returns a failing
         ToolExecutionResult naming the bot and the tool rather than an
         exception, so one refused call cannot abort a run mid-step.
    Neither side can be picked independently: side 2 is what makes "refused"
    true even if a caller bypasses the listings, which is exactly the case the
    test suite exercises.
    """

    def __init__(
        self,
        host: Any,
        registry: BotRegistry,
        bot_id: str = "",
        broker: Optional[PermissionBroker] = None,
    ):
        self._host = host
        self._registry = registry
        self._bot_id = bot_id
        # The shared broker. Not a second broker — the same instance, so a
        # session-level ALLOW can never widen a bot's allow-list.
        self._broker = broker

    @property
    def bot_id(self) -> str:
        return self._bot_id

    def _allowed(self, tool_name: str) -> bool:
        ok, _ = self._registry.check_tool(self._bot_id, tool_name)
        return ok

    # -- read surface ---------------------------------------------------

    def get_tool(self, name: str):
        if not self._allowed(name):
            return None
        return self._host.get_tool(name)

    def list_tools(self) -> List[Any]:
        return [t for t in self._host.list_tools() if self._allowed(t.name)]

    def get_tool_descriptions(self) -> Dict[str, str]:
        return {
            name: desc
            for name, desc in self._host.get_tool_descriptions().items()
            if self._allowed(name)
        }

    def get_formatted_tool_list_for_system_two(self) -> str:
        return "\n".join(
            f"### Tool: `{t.name}`\n{t.description}\n"
            f"Parameters:\n{json.dumps(t.parameters_schema.get('properties', {}), indent=2)}\n"
            for t in self.list_tools()
        )

    # -- write surface --------------------------------------------------

    def register_tool(self, tool: Any) -> None:
        # Registration does not grant access; the allow-list still decides.
        self._host.register_tool(tool)

    def execute_tool(self, name: str, arguments: Dict[str, Any]):
        from dual_agent.mcp_host import ToolExecutionResult

        allowed, refusal = self._registry.check_tool(self._bot_id, name)
        if not allowed:
            logger.warning(f"[BotScopedMCPHost] {refusal}")
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                output=None,
                error=refusal,
            )

        # A tool this bot holds may still be risky. Route that through the same
        # PermissionBroker instance the host path uses, so a bot grant is never
        # a way around an approval card.
        tool_def = self._host.get_tool(name)
        if self._broker is not None and tool_def is not None and getattr(
            tool_def, "requires_approval", False
        ):
            decision, args = self._broker.request_approval(
                tool_name=name,
                args=arguments,
                risk_level=getattr(tool_def, "risk_level", "high"),
            )
            if decision == ApprovalDecision.DENY:
                return ToolExecutionResult(
                    tool_name=name,
                    success=False,
                    output=None,
                    error=(
                        f"Refused: tool '{name}' was not approved by the "
                        f"PermissionBroker for bot '{self._bot_id}'. The bot "
                        "holds the grant; approval was denied."
                    ),
                )
            arguments = args

        return self._host.execute_tool(name, arguments)

    def __getattr__(self, item: str) -> Any:
        # Anything else (future host API) passes through unchanged.
        return getattr(self._host, item)


# ---------------------------------------------------------------------------
# Group conversation
# ---------------------------------------------------------------------------


@dataclass
class Handoff:
    """One bot-to-bot hand-off, recorded so the chain is auditable."""

    hop_index: int
    depth: int
    sender: str
    recipient: str
    task: str
    outcome: str
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hop_index": self.hop_index,
            "depth": self.depth,
            "from": self.sender,
            "to": self.recipient,
            "task": self.task,
            "outcome": self.outcome,
            "detail": self.detail,
        }


@dataclass
class GroupTaskResult:
    """Result of a task run across a group of bots."""

    final_output: str
    is_completed: bool
    stopped_reason: Optional[str]
    hops: List[Handoff] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)

    @property
    def hop_count(self) -> int:
        return sum(1 for h in self.hops if h.outcome == "completed")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "final_output": self.final_output,
            "is_completed": self.is_completed,
            "stopped_reason": self.stopped_reason,
            "trace": list(self.trace),
            "hops": [h.as_dict() for h in self.hops],
        }


class MultiBotGroup:
    """A bounded group of bots where one can hand a sub-task to another.

    Bounds (both enforced, never silently skipped):
      - `max_hops`: total completed hand-offs in one chain. Exceeding it stops
        the chain with `stopped_reason='hop_limit_reached'`.
      - `max_depth`: how deeply hand-offs may nest. Exceeding it stops with
        `stopped_reason='depth_limit_reached'`.

    `run_fn(bot_id, task, depth)` performs the actual work for one bot. It is
    injected so tests can drive a real registry and real allow-list checks
    without needing a live model.
    """

    def __init__(
        self,
        registry: BotRegistry,
        run_fn: Any,
        max_hops: Optional[int] = None,
        max_depth: Optional[int] = None,
    ):
        self.registry = registry
        self.run_fn = run_fn
        self.max_hops = int(max_hops if max_hops is not None else registry.max_hops)
        self.max_depth = int(max_depth if max_depth is not None else registry.max_depth)

    def run(
        self,
        task: str,
        originator: str,
        handoff: Optional[Tuple[str, Sequence[str]]] = None,
    ) -> GroupTaskResult:
        """Run `task` as `originator`, optionally handing sub-tasks onward.

        `handoff` is (recipient_bot_id, sub_tasks). For each sub-task the
        recipient runs it at depth+1 and the result is returned to the
        originator, whose `final_output` records that it received them. This is
        the minimal shape of a group conversation: one hop out, result back.
        """
        if originator not in {p.id for p in self.registry.all()}:
            raise ValueError(
                f"unknown bot {originator!r}; registered: "
                f"{[p.id for p in self.registry.all()]}"
            )

        hops: List[Handoff] = []
        trace: List[str] = []
        stopped_reason: Optional[str] = None

        def _run_one(bot_id: str, job: str, depth: int) -> GroupTaskResult:
            """Recursive worker. Every exit path names its reason."""
            nonlocal stopped_reason
            sub_hops: List[Handoff] = []
            sub_trace: List[str] = [f"depth={depth} bot={bot_id} task={job!r}"]
            sub_stopped: Optional[str] = None

            try:
                output = self.run_fn(bot_id, job, depth)
            except Exception as e:
                return GroupTaskResult(
                    final_output=f"bot {bot_id} failed: {e}",
                    is_completed=False,
                    stopped_reason=f"bot_error: {type(e).__name__}: {e}",
                    hops=sub_hops,
                    trace=sub_trace + [f"error={type(e).__name__}: {e}"],
                )

            sub_trace.append(f"depth={depth} bot={bot_id} output={str(output)[:160]!r}")

            return GroupTaskResult(
                final_output=str(output),
                is_completed=True,
                stopped_reason=sub_stopped,
                hops=sub_hops,
                trace=sub_trace,
            )

        primary = _run_one(originator, task, depth=0)
        hops.extend(primary.hops)
        trace.extend(primary.trace)
        if primary.stopped_reason:
            stopped_reason = primary.stopped_reason

        if handoff and stopped_reason is None:
            recipient, sub_tasks = handoff
            if recipient not in {p.id for p in self.registry.all()}:
                raise ValueError(f"unknown hand-off recipient {recipient!r}")

            for i, sub_task in enumerate(sub_tasks, start=1):
                if len(hops) >= self.max_hops:
                    hops.append(
                        Handoff(
                            hop_index=len(hops) + 1,
                            depth=1,
                            sender=originator,
                            recipient=recipient,
                            task=sub_task,
                            outcome="stopped",
                            detail=f"hop_limit_reached (max_hops={self.max_hops})",
                        )
                    )
                    trace.append(
                        f"depth=1 STOPPED before running {sub_task!r}: "
                        f"hop limit {self.max_hops} reached"
                    )
                    stopped_reason = "hop_limit_reached"
                    break

                if 1 > self.max_depth:
                    hops.append(
                        Handoff(
                            hop_index=len(hops) + 1,
                            depth=1,
                            sender=originator,
                            recipient=recipient,
                            task=sub_task,
                            outcome="stopped",
                            detail=f"depth_limit_reached (max_depth={self.max_depth})",
                        )
                    )
                    trace.append(
                        f"depth=1 STOPPED before running {sub_task!r}: "
                        f"depth limit {self.max_depth} reached"
                    )
                    stopped_reason = "depth_limit_reached"
                    break

                sub = _run_one(recipient, sub_task, depth=1)
                if sub.stopped_reason:
                    hops.append(
                        Handoff(
                            hop_index=len(hops) + 1,
                            depth=1,
                            sender=originator,
                            recipient=recipient,
                            task=sub_task,
                            outcome="failed",
                            detail=sub.stopped_reason,
                        )
                    )
                    trace.extend(sub.trace)
                    stopped_reason = sub.stopped_reason
                    break

                hops.append(
                    Handoff(
                        hop_index=len(hops) + 1,
                        depth=1,
                        sender=originator,
                        recipient=recipient,
                        task=sub_task,
                        outcome="completed",
                        detail=sub.final_output[:300],
                    )
                )
                trace.extend(sub.trace)
                trace.append(
                    f"depth=0 bot={originator} received result from {recipient}: "
                    f"{sub.final_output[:160]!r}"
                )

        returned = [h for h in hops if h.sender == originator and h.outcome == "completed"]
        if returned:
            summary = (
                f"bot '{originator}' completed its own task and received "
                f"{len(returned)} result(s) back from '{returned[0].recipient}': "
                + "; ".join(h.detail for h in returned)
            )
        else:
            summary = primary.final_output

        return GroupTaskResult(
            final_output=summary,
            is_completed=stopped_reason is None and primary.is_completed,
            stopped_reason=stopped_reason,
            hops=hops,
            trace=trace,
        )


# ---------------------------------------------------------------------------
# Attaching to the dispatcher
# ---------------------------------------------------------------------------


def apply_bot_profile(
    dispatcher: Any,
    profile: Optional[BotProfile],
    registry: Optional[BotRegistry] = None,
) -> None:
    """Stamp a bot identity onto a live dispatcher, in place.

    Called after construction so no existing entry point changes: a dispatcher
    built without a profile is exactly what it was before this module existed.

    What "stamped" means — no field is decorative:
      - `dispatcher.bot_profile` / `dispatcher.bot_registry`, for auditing.
      - the MCP host becomes bot-scoped, so the router and System 2 see only
        this bot's tools and `execute_tool` refuses everything else (side 2 of
        the enforcement, using the dispatcher's EXISTING broker).
      - `model` / `provider` are copied onto the System 2 provider object and
        reported by `applied_identity()`. This does NOT switch endpoints: the
        provider is constructed by the caller and this layer does not know how
        to re-target it. The fields say what was asked for; the caller is
        responsible for the provider actually honouring it.
    """
    if profile is None or registry is None or not registry.only_allow_listed_tools:
        return

    dispatcher.bot_profile = profile
    dispatcher.bot_registry = registry

    host = getattr(dispatcher, "mcp", None)
    if host is not None and not isinstance(host, BotScopedMCPHost):
        dispatcher.mcp = BotScopedMCPHost(
            host, registry, bot_id=profile.id, broker=getattr(dispatcher, "broker", None)
        )

    if profile.model:
        setattr(dispatcher.s2, "model", profile.model)
    if profile.provider:
        setattr(dispatcher.s2, "provider", profile.provider)

    logger.info(
        f"[Bots] dispatcher stamped as bot '{profile.id}' ({profile.role}); "
        f"{len(profile.allowed_tools)} tool(s) allowed"
    )


def applied_identity(dispatcher: Any) -> Dict[str, Any]:
    """Audit view of which bot a dispatcher is answering as."""
    profile: Optional[BotProfile] = getattr(dispatcher, "bot_profile", None)
    if profile is None:
        return {
            "bot_id": None,
            "display_name": None,
            "role": None,
            "model": None,
            "provider": None,
            "enforced": False,
            "note": "no bot profile applied; single default agent",
        }
    s2 = getattr(dispatcher, "s2", None)
    return {
        "bot_id": profile.id,
        "display_name": profile.display_name,
        "role": profile.role,
        "model": getattr(s2, "model", None) or profile.model,
        "provider": getattr(s2, "provider", None) or profile.provider,
        "enforced": isinstance(getattr(dispatcher, "mcp", None), BotScopedMCPHost),
        "note": "bot profile applied",
    }


def compose_persona(profile: Optional[BotProfile], base_persona: str = "") -> str:
    """Fold a bot's persona fragment into a caller-supplied base persona.

    Returns `base_persona` unchanged when there is no profile, or when the
    profile opts out of inheriting it.
    """
    if profile is None:
        return base_persona
    if not profile.persona:
        return base_persona if profile.inherit_default_persona else ""
    header = f"You are {profile.display_name}, {profile.role}."
    if profile.inherit_default_persona and base_persona:
        return f"{header}\n{profile.persona}\n\n{base_persona}"
    return f"{header}\n{profile.persona}"


# ---------------------------------------------------------------------------
# Disk helpers (config.py conventions: data dir, 0600, explicit precedence)
# ---------------------------------------------------------------------------


class BotConfigError(ValueError):
    """Raised only by the explicit save/create path, never by load()."""


def get_bot_config_path() -> str:
    """Path to the single bot profile file.

    Precedence matches the rest of the project: an explicitly set environment
    variable wins, then the profile's own data dir. `DUAL_AGENT_HOME` is
    handled inside `get_default_data_dir()`, so tests that redirect it also
    redirect this file.
    """
    explicit = os.environ.get("DUAL_AGENT_BOTS_FILE")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    return os.path.join(get_default_data_dir(), BOTS_FILE_NAME)


def save_bots(
    profiles: Sequence[BotProfile],
    path: Optional[str] = None,
    default_bot: Optional[str] = None,
    max_hops: int = DEFAULT_MAX_HOPS,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> str:
    """Write profiles to bots.json with 0600 permissions. Returns the path."""
    path = path or get_bot_config_path()
    known = {p.id for p in profiles}
    if default_bot and default_bot not in known:
        raise BotConfigError(
            f"default_bot {default_bot!r} is not among {sorted(known)}"
        )
    payload: Dict[str, Any] = {
        "bots": [p.model_dump() for p in profiles],
        "max_hops": int(max_hops),
        "max_depth": int(max_depth),
    }
    if default_bot:
        payload["default_bot"] = default_bot

    os.makedirs(os.path.dirname(path) or ".", mode=0o700, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass
    return path
