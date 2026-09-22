"""Multi-bot identity layer — profiles, per-bot tool grants, routing, hand-offs.

Every assertion here is against a real outcome: a real BotRegistry, real
BotProfile objects, the real MCPHost tool surface, the real PermissionBroker,
and the real DualProcessDispatcher. Nothing is asserted against a stand-in that
merely echoes what the test asked for.

`run_fn` in the group tests is the harness that decides *what work a bot
performs*, which is genuinely caller-supplied in production too; the
enforcement it exercises (allow-list checks, hop/depth bounds, the audit trail)
is all real.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from dual_agent.bots import (
    BotProfile,
    BotRegistry,
    BotScopedMCPHost,
    MultiBotGroup,
    applied_identity,
    apply_bot_profile,
    compose_persona,
    get_bot_config_path,
    save_bots,
)
from dual_agent.dispatcher import DualProcessDispatcher
from dual_agent.mcp_host import MCPHost
from dual_agent.memory import MemoryEngine, get_default_data_dir
from dual_agent.permission_broker import ApprovalDecision, PermissionBroker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEVELOPER_TOOLS = ["read_file", "write_file", "patch_file", "search_file", "run_shell_command"]
OUTREACH_TOOLS = ["read_file", "send_email"]
ANALYST_TOOLS = ["read_file", "search_file"]


def _three_bots():
    """Three bots with deliberately DISJOINT tool grants.

    `send_email` exists only on outreach, `write_file`/`patch_file`/
    `run_shell_command` only on developer. That disjointness is what makes the
    refusal test meaningful.
    """
    return [
        BotProfile(
            id="dev",
            display_name="Ada",
            role="Web & AI Agent Developer",
            provider="deepseek",
            model="deepseek-chat",
            persona="You ship code. Prefer small patches and run the tests you write.",
            routing_keywords=["code", "python", "repo", "pytest", "implement", "refactor", "bug"],
            allowed_tools=DEVELOPER_TOOLS,
        ),
        BotProfile(
            id="outreach",
            display_name="Miles",
            role="Outreach & Sales Specialist",
            provider="openai",
            model="gpt-4o",
            persona="You write short, specific cold email. No hype.",
            routing_keywords=["outreach", "cold", "email", "lead", "prospect", "sales", "pitch"],
            allowed_tools=OUTREACH_TOOLS,
        ),
        BotProfile(
            id="analyst",
            display_name="Grace",
            role="Research Analyst",
            provider="deepseek",
            model="deepseek-reasoner",
            persona="You cite what you read and say when you could not verify it.",
            routing_keywords=["research", "compare", "analyse", "analyze", "survey", "review"],
            allowed_tools=ANALYST_TOOLS,
        ),
    ]


@pytest.fixture
def registry():
    return BotRegistry(profiles=_three_bots(), default_bot_id="analyst")


@pytest.fixture
def bots_file(tmp_path, monkeypatch):
    """A bots.json on disk, inside an isolated data dir."""
    home = tmp_path / "dual_agent_home"
    home.mkdir()
    monkeypatch.setenv("DUAL_AGENT_HOME", str(home))
    monkeypatch.delenv("DUAL_AGENT_BOTS_FILE", raising=False)
    monkeypatch.delenv("DUAL_AGENT_DEFAULT_BOT", raising=False)
    path = str(home / "bots.json")
    save_bots(_three_bots(), path=path, default_bot="analyst")
    return path


@pytest.fixture
def dispatcher():
    """A real dispatcher on a temp memory DB. Never the developer's DB."""
    from dual_agent.memory import MemoryEngine as ME
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="bots-test-")
    memory = ME(db_path=os.path.join(tmpdir, "memory.db"))
    d = DualProcessDispatcher(memory_engine=memory)
    yield d
    try:
        d.memory.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# (a) Routing selects the role-appropriate bot and records WHY
# ---------------------------------------------------------------------------


def test_route_picks_role_match_and_records_reason(registry):
    """A clearly developer-flavoured goal must land on the developer bot."""
    decision = registry.route("implement a pytest for the bot registry and fix the bug")

    assert decision.bot_id == "dev", decision.as_dict()
    assert decision.method == "keyword_heuristic", (
        "the router must describe itself honestly; it is not a model"
    )
    assert decision.reason, "a routing decision with no reason is not auditable"
    assert "Web & AI Agent Developer" in decision.reason
    assert decision.score > 0
    # The reason names the actual matched keywords, not a generic sentence.
    assert any(kw in decision.reason for kw in ("pytest", "bug", "implement"))
    # Every candidate and its score is retained, so the choice can be disputed.
    assert {c[0] for c in decision.candidates} == {"dev", "outreach", "analyst"}
    assert dict(decision.candidates)["dev"] == max(s for _, s in decision.candidates)


def test_route_picks_other_bots_for_their_own_domains(registry):
    """The same router must not just always answer 'dev'."""
    email = registry.route("draft a cold outreach email to a prospect lead")
    assert email.bot_id == "outreach", email.as_dict()
    assert "Outreach & Sales Specialist" in email.reason

    research = registry.route("analyse and compare what the survey papers claim")
    assert research.bot_id == "analyst", research.as_dict()
    assert "Research Analyst" in research.reason

    # Three distinct goals, three distinct winners: the router discriminates.
    winners = {
        registry.route("write python code for the repo").bot_id,
        registry.route("send a sales pitch email").bot_id,
        registry.route("research survey review").bot_id,
    }
    assert winners == {"dev", "outreach", "analyst"}


def test_route_with_no_match_falls_back_and_says_so(registry):
    """An unmatched message must not claim a keyword match it did not get."""
    decision = registry.route("zzzz qqqq")
    assert decision.bot_id == "analyst"  # configured default_bot
    assert decision.score == 0.0
    assert "no role keyword" in decision.reason
    assert "default_bot" in decision.reason


def test_route_on_empty_registry_reports_single_agent():
    decision = BotRegistry().route("anything at all")
    assert decision.bot_id is None
    assert decision.method == "none"
    assert "single default agent" in decision.reason


# ---------------------------------------------------------------------------
# (b) Per-bot refusal, naming both the bot and the tool
# ---------------------------------------------------------------------------


def test_wrap_is_identity_when_no_bots_configured():
    """The no-bots path must return the SAME object, not a filtered copy."""
    host = MCPHost()
    empty = BotRegistry()
    assert empty.enabled is False
    assert empty.only_allow_listed_tools is False
    assert empty.wrap_mcp_host(host) is host
    # Even the required-bot-id guard is skipped, because nothing is enforced.
    assert empty.wrap_mcp_host(host, bot_id="anything") is host


def test_wrap_refuses_to_build_a_scoped_host_without_a_bot_id(registry):
    """A scoped host with no bot is a host that refuses everything. Reject it."""
    with pytest.raises(ValueError) as exc:
        registry.wrap_mcp_host(MCPHost())
    assert "bot's id" in str(exc.value)


def test_bot_a_refused_tool_held_by_bot_b(registry):
    """dev may not call send_email, which is outreach's tool. Refusal names both."""
    host = MCPHost()
    assert "send_email" in registry.get("outreach").allowed_tools
    assert "send_email" not in registry.get("dev").allowed_tools

    # The tool really exists on the host, and really works when called
    # directly — so this is a permission refusal, not a missing tool.
    host.register_tool(
        type(host.list_tools()[0])(
            name="send_email",
            description="Send an email to a lead.",
            parameters_schema={"type": "object", "properties": {"to": {"type": "string"}}},
            handler=lambda args: f"sent to {args.get('to')}",
        )
    )
    real = host.execute_tool("send_email", {"to": "lead@example.com"})
    assert real.success is True and real.output == "sent to lead@example.com"

    scoped = BotScopedMCPHost(host, registry, bot_id="dev")
    result = scoped.execute_tool("send_email", {"to": "lead@example.com"})

    assert result.success is False, "dev must not be able to send email"
    assert "dev" in result.error, f"refusal must name the bot: {result.error!r}"
    assert "send_email" in result.error, f"refusal must name the tool: {result.error!r}"
    assert "outreach" in result.error, (
        f"refusal should name the bot that does hold the grant: {result.error!r}"
    )
    assert "sent to" not in str(result.output)


def test_allow_list_is_two_sided_not_just_a_listing_filter(registry):
    """A tool outside the allow-list must be invisible AND unreachable."""
    host = MCPHost()
    scoped = BotScopedMCPHost(host, registry, bot_id="analyst")

    # Side 1: not offered to the router or to System 2.
    offered = scoped.get_tool_descriptions()
    assert "read_file" in offered and "search_file" in offered
    assert "write_file" not in offered
    assert "run_shell_command" not in offered
    assert scoped.get_tool("write_file") is None
    assert "write_file" not in scoped.get_formatted_tool_list_for_system_two()

    # Side 2: even calling it directly by name is refused, not silently executed.
    refused = scoped.execute_tool("write_file", {"path": "/tmp/should-not-exist.txt", "content": "x"})
    assert refused.success is False
    assert "analyst" in refused.error and "write_file" in refused.error
    assert not os.path.exists("/tmp/should-not-exist.txt")


def test_granted_tool_still_executes_for_the_right_bot(registry):
    """The allow-list must not be a blanket refusal.

    `list_directory` is a real host tool that is on NO bot's allow-list here,
    so it is refused even for `dev` — see
    `test_list_directory_is_refused_because_no_bot_grants_it`. What is granted
    actually runs.
    """
    host = MCPHost()
    scoped = BotScopedMCPHost(host, registry, bot_id="dev")

    ok, reason = registry.check_tool("dev", "read_file")
    assert ok and reason is None
    read = scoped.execute_tool("read_file", {"path": "pyproject.toml"})
    assert read.success is True
    assert "dual-agent" in str(read.output).lower() or "version" in str(read.output).lower()

    # search_file is granted too, and finds a string that is genuinely there.
    found = scoped.execute_tool(
        "search_file", {"path": "src/dual_agent/bots.py", "query": "class BotProfile"}
    )
    assert found.success is True
    assert "class BotProfile" in str(found.output)


def test_list_directory_is_refused_because_no_bot_grants_it(registry):
    """An allow-list means exactly what it says: unlisted is denied.

    `list_directory` exists on the host and works — calling it on a bare host
    succeeds — but no bot in this registry lists it, so every bot is refused
    it. That is the intended semantics; accidentally permissive would be worse.
    """
    host = MCPHost()
    assert host.execute_tool("list_directory", {"path": "."}).success is True

    for bot_id in ("dev", "outreach", "analyst"):
        result = BotScopedMCPHost(host, registry, bot_id=bot_id).execute_tool(
            "list_directory", {"path": "."}
        )
        assert result.success is False, f"{bot_id} must not reach an unlisted tool"
        assert bot_id in result.error and "list_directory" in result.error


def test_registry_refuses_unknown_bot_by_name():
    reg = BotRegistry(profiles=_three_bots())
    ok, reason = reg.check_tool("nobody", "read_file")
    assert ok is False
    assert "nobody" in reason and "read_file" in reason
    assert "analyst" in reason  # lists who IS registered


def test_bot_grant_does_not_bypass_the_existing_permission_broker(registry):
    """A risky tool on the allow-list still needs broker approval.

    This is the integration point with the EXISTING broker: the same class, the
    same semantics. An allow-list grants *reachability*, not *consent*.
    """
    host = MCPHost()
    assert host.get_tool("write_file").requires_approval is True

    # A bot that holds write_file.
    dev = registry.get("dev")
    assert dev.allows_tool("write_file")

    # auto_allow=False, non_interactive=True is the gateway posture: an
    # unanswerable approval is a denial, not a hang.
    broker = PermissionBroker(auto_allow=False, non_interactive=True)
    scoped = BotScopedMCPHost(host, registry, bot_id="dev", broker=broker)
    result = scoped.execute_tool("write_file", {"path": "/tmp/x.txt", "content": "x"})
    assert result.success is False
    assert "not approved" in result.error and "write_file" in result.error
    assert not os.path.exists("/tmp/x.txt")

    # With an auto-allow broker (the CI posture) the same call goes through,
    # so the refusal above came from the broker and not from the allow-list.
    permissive = BotScopedMCPHost(
        host, registry, bot_id="dev", broker=PermissionBroker(auto_allow=True)
    )
    allowed = permissive.execute_tool("write_file", {"path": "/tmp/bots-allow-test.txt", "content": "ok"})
    assert allowed.success is True
    os.remove("/tmp/bots-allow-test.txt")


# ---------------------------------------------------------------------------
# Registry loading / config conventions / 0600
# ---------------------------------------------------------------------------


def test_loads_from_disk_and_honours_data_dir(bots_file):
    reg = BotRegistry.load(bots_file)
    assert reg.enabled is True
    assert {p.id for p in reg.all()} == {"dev", "outreach", "analyst"}
    assert reg.default_bot_id == "analyst"
    assert reg.get("dev").model == "deepseek-chat"
    assert reg.get("dev").provider == "deepseek"


def test_bots_file_is_0600(bots_file):
    mode = stat.S_IMODE(os.stat(bots_file).st_mode)
    assert mode == 0o600, f"bots.json has mode {oct(mode)}"


def test_bot_config_path_follows_data_dir(bots_file, monkeypatch):
    assert get_bot_config_path() == bots_file
    monkeypatch.setenv("DUAL_AGENT_BOTS_FILE", "/tmp/elsewhere.json")
    assert get_bot_config_path() == "/tmp/elsewhere.json"


def test_missing_file_is_silently_off(tmp_path):
    """Absent bots.json is the normal case: feature off, no error, no noise."""
    reg = BotRegistry.load(str(tmp_path / "nope" / "bots.json"))
    assert reg.enabled is False
    assert reg.source_path is None
    assert reg.all() == []


def test_malformed_file_disables_rights_and_logs_error(tmp_path, caplog):
    """A broken bots.json must not silently look like 'no permissions in force'."""
    path = tmp_path / "bots.json"
    path.write_text("{not json at all", encoding="utf-8")
    with caplog.at_level("ERROR"):
        reg = BotRegistry.load(str(path))
    assert reg.enabled is False
    assert any("bots.json" in r.message or "Could not parse" in r.message for r in caplog.records), (
        "a malformed bots.json must say so out loud"
    )


def test_env_default_bot_only_overrides_when_deliberately_set(bots_file, monkeypatch):
    monkeypatch.setenv("DUAL_AGENT_DEFAULT_BOT", "dev")
    assert BotRegistry.load(bots_file).default_bot_id == "dev"
    monkeypatch.setenv("DUAL_AGENT_DEFAULT_BOT", "ghost")
    # Unknown id is ignored, and the file's default still wins.
    assert BotRegistry.load(bots_file).default_bot_id == "analyst"


def test_save_rejects_unknown_default_bot(tmp_path):
    from dual_agent.bots import BotConfigError

    with pytest.raises(BotConfigError):
        save_bots(_three_bots(), path=str(tmp_path / "b.json"), default_bot="ghost")


def test_invalid_bot_id_rejected():
    with pytest.raises(Exception):
        BotProfile(id="Not A Bot", display_name="x", role="y")


def test_team_map_reports_role_model_and_grants(registry):
    tm = {row["id"]: row for row in registry.team_map()}
    assert tm["dev"]["role"] == "Web & AI Agent Developer"
    assert tm["dev"]["model"] == "deepseek-chat"
    assert tm["dev"]["tools"] == DEVELOPER_TOOLS
    assert "send_email" in tm["outreach"]["tools"]
    assert "send_email" not in tm["dev"]["tools"]


def test_duplicate_bot_id_rejected(registry):
    with pytest.raises(ValueError):
        registry.add(BotProfile(id="dev", display_name="Copy", role="dupe"))


# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------


def test_apply_profile_scopes_the_live_dispatcher(dispatcher, registry):
    """The dispatcher's real host is replaced by a real bot-scoped view."""
    assert isinstance(dispatcher.mcp, MCPHost)
    assert getattr(dispatcher, "bot_profile", None) is None

    apply_bot_profile(dispatcher, registry.get("outreach"), registry)

    assert isinstance(dispatcher.mcp, BotScopedMCPHost)
    ident = applied_identity(dispatcher)
    assert ident["bot_id"] == "outreach"
    assert ident["role"] == "Outreach & Sales Specialist"
    assert ident["enforced"] is True
    assert ident["model"] == "gpt-4o" and ident["provider"] == "openai"

    # Through the live dispatcher's host, the refusal is real.
    result = dispatcher.mcp.execute_tool("run_shell_command", {"command": "echo hi"})
    assert result.success is False
    assert "run_shell_command" in result.error and "outreach" in result.error
    # ...and its own tool surface is intact.
    assert dispatcher.mcp.get_tool("read_file") is not None


def test_apply_profile_is_a_noop_without_bots(dispatcher):
    """Stamping must do nothing when there are no bots — the compat guarantee."""
    original_host = dispatcher.mcp
    apply_bot_profile(dispatcher, None, BotRegistry())
    apply_bot_profile(dispatcher, None, None)
    assert dispatcher.mcp is original_host
    assert getattr(dispatcher, "bot_profile", None) is None
    assert applied_identity(dispatcher)["enforced"] is False
    assert isinstance(dispatcher.mcp, MCPHost) and not isinstance(dispatcher.mcp, BotScopedMCPHost)


def test_model_preference_is_recorded_but_does_not_switch_endpoints(dispatcher, registry):
    """A KNOWN LIMITATION, pinned so nobody assumes more than is true.

    The profile's `model`/`provider` are copied onto the System 2 object and
    reported by `applied_identity()`, but the provider CLASS is unchanged —
    stamping a bot does not re-target the endpoint. Claiming "each bot runs its
    own model" would be false until the caller builds a provider per bot.
    """
    before_class = type(dispatcher.s2).__name__
    apply_bot_profile(dispatcher, registry.get("outreach"), registry)

    ident = applied_identity(dispatcher)
    assert ident["model"] == "gpt-4o" and ident["provider"] == "openai"
    assert type(dispatcher.s2).__name__ == before_class, (
        "if this ever changes, the docstring on apply_bot_profile must change "
        "too — the per-bot model does not switch the provider today"
    )


def test_end_to_end_route_then_stamp_then_refuse(dispatcher, registry):
    """The full path: a message is routed, the winner is stamped, grants hold."""
    decision = registry.route("send a cold outreach email to this prospect")
    assert decision.bot_id == "outreach"

    apply_bot_profile(dispatcher, registry.get(decision.bot_id), registry)

    # outreach may read_file ...
    assert dispatcher.mcp.get_tool("read_file") is not None
    # ... and may NOT patch_file, which the routed-away bot holds.
    refused = dispatcher.mcp.execute_tool(
        "patch_file", {"path": "/tmp/nope.py", "old_string": "a", "new_string": "b"}
    )
    assert refused.success is False
    assert "outreach" in refused.error and "patch_file" in refused.error
    assert "dev" in refused.error  # names the bot that does hold it
    assert decision.reason and decision.method == "keyword_heuristic"


def test_persona_composition(registry):
    dev = registry.get("dev")
    composed = compose_persona(dev, "You are a careful agent.")
    assert composed.startswith("You are Ada, Web & AI Agent Developer.")
    assert "small patches" in composed
    assert "careful agent" in composed

    base = "Base persona only."
    solo = BotProfile(id="solo", display_name="S", role="R", persona="", allowed_tools=[])
    assert compose_persona(solo, base) == base
    assert compose_persona(None, base) == base
    # A profile that opts out does not inherit.
    independent = BotProfile(
        id="solo2", display_name="S2", role="R", persona="Mine.", inherit_default_persona=False
    )
    assert "Base persona" not in compose_persona(independent, base)


# ---------------------------------------------------------------------------
# (c) Hand-off completes, results return, bounds stop a runaway chain
# ---------------------------------------------------------------------------


def _recording_run_fn(log, refuse=()):
    """A group run_fn that does real work and records the calls it received."""

    def run_fn(bot_id, task, depth):
        log.append((bot_id, task, depth))
        if bot_id in refuse:
            raise RuntimeError(f"bot {bot_id} refuses this task")
        return f"{bot_id} did: {task}"

    return run_fn


def test_handoff_completes_and_result_returns_to_originator(registry):
    log = []
    group = MultiBotGroup(registry, _recording_run_fn(log), max_hops=8, max_depth=4)

    result = group.run(
        task="write the landing page copy",
        originator="dev",
        handoff=("outreach", ["email the beta list about the launch"]),
    )

    assert result.is_completed is True
    assert result.stopped_reason is None
    assert ("dev", "write the landing page copy", 0) in log
    assert ("outreach", "email the beta list about the launch", 1) in log
    assert result.hop_count == 1

    hop = result.hops[0]
    assert hop.sender == "dev" and hop.recipient == "outreach"
    assert hop.outcome == "completed"
    assert "outreach did: email the beta list" in hop.detail
    # The result came BACK: it is in the originator's final output.
    assert "dev" in result.final_output
    assert "outreach" in result.final_output
    assert "outreach did:" in result.final_output
    # And the chain is auditable.
    assert any("received result from outreach" in line for line in result.trace)


def test_hop_limit_stops_a_runaway_chain(registry):
    """Ten sub-tasks with max_hops=3 must run three and stop, naming the stop."""
    log = []
    group = MultiBotGroup(registry, _recording_run_fn(log), max_hops=3, max_depth=10)

    result = group.run(
        task="kick off the whole launch",
        originator="dev",
        handoff=("outreach", [f"sub-task {i}" for i in range(1, 11)]),
    )

    assert result.stopped_reason == "hop_limit_reached"
    assert result.is_completed is False
    assert result.hop_count == 3, f"expected exactly 3 completed hops, got {result.hop_count}"
    assert [t for b, t, d in log if b == "outreach"] == ["sub-task 1", "sub-task 2", "sub-task 3"]
    stopped = [h for h in result.hops if h.outcome == "stopped"]
    assert len(stopped) == 1 and "hop_limit_reached" in stopped[0].detail
    assert any("hop limit 3 reached" in line for line in result.trace)


def test_depth_limit_stops_nesting(registry):
    """max_depth=0 means no nesting is permitted at all — and says so."""
    log = []
    group = MultiBotGroup(registry, _recording_run_fn(log), max_hops=100, max_depth=0)

    result = group.run(
        task="delegate everything",
        originator="dev",
        handoff=("outreach", ["nested job"]),
    )

    assert result.stopped_reason == "depth_limit_reached"
    assert result.is_completed is False
    assert log == [("dev", "delegate everything", 0)], (
        "the nested task must not have run at all"
    )
    assert result.hops[0].outcome == "stopped"
    assert "depth_limit_reached" in result.hops[0].detail
    assert any("depth limit 0 reached" in line for line in result.trace)


def test_self_handoff_chain_terminates(registry):
    """A bot handing work to itself is exactly the loop that must not spin.

    Each hop costs one of the budget, so the chain ends; it does not recurse
    forever.
    """
    log = []
    group = MultiBotGroup(registry, _recording_run_fn(log), max_hops=4, max_depth=4)

    result = group.run(
        task="review the plan",
        originator="analyst",
        handoff=("analyst", [f"re-review {i}" for i in range(50)]),
    )

    assert result.stopped_reason == "hop_limit_reached"
    assert result.hop_count == 4
    assert len(log) == 5  # 1 originator run + 4 hops
    assert result.is_completed is False


def test_handoff_failure_returns_to_originator_and_names_the_cause(registry):
    log = []
    group = MultiBotGroup(registry, _recording_run_fn(log, refuse={"outreach"}), max_hops=5)

    result = group.run(
        task="write copy",
        originator="dev",
        handoff=("outreach", ["email the list"]),
    )

    assert result.is_completed is False
    assert "bot_error" in (result.stopped_reason or "")
    assert "RuntimeError" in result.stopped_reason
    # The originator's own work still happened and its output is reported.
    assert ("dev", "write copy", 0) in log
    assert result.hops[0].outcome == "failed"
    assert "bot_error" in result.hops[0].detail


def test_group_rejects_unknown_bots(registry):
    group = MultiBotGroup(registry, _recording_run_fn([]))
    with pytest.raises(ValueError):
        group.run(task="x", originator="ghost")
    with pytest.raises(ValueError):
        group.run(task="x", originator="dev", handoff=("ghost", ["y"]))


def test_group_defaults_come_from_the_registry():
    reg = BotRegistry(profiles=_three_bots(), max_hops=2, max_depth=1)
    group = MultiBotGroup(reg, _recording_run_fn([]))
    assert group.max_hops == 2 and group.max_depth == 1


def test_two_bot_group_uses_real_allow_lists(registry):
    """Handing a task to a bot uses THAT bot's grants, checked for real."""
    host = MCPHost()
    scoped = {b: BotScopedMCPHost(host, registry, bot_id=b) for b in ("dev", "outreach")}

    def run_fn(bot_id, task, depth):
        # The work a bot is allowed to do, decided by its real allow-list.
        if "email" in task:
            return scoped[bot_id].execute_tool("send_email", {"to": "x@y.z"})
        return scoped[bot_id].execute_tool("read_file", {"path": "missing-file"})

    group = MultiBotGroup(registry, run_fn, max_hops=2)
    result = group.run(task="read_file the brief", originator="dev", handoff=("outreach", ["email the lead"]))

    assert result.is_completed is True
    assert result.hop_count == 1
    # outreach holds send_email and was allowed to reach it: the refusal it got
    # was "not registered", proving it passed the allow-list gate.
    assert "not registered" in result.hops[0].detail
    assert "may not call" not in result.hops[0].detail

    # Reverse it: dev does not hold send_email, so the same call is refused by
    # the allow-list before it ever reaches the host.
    reverse = MultiBotGroup(
        BotRegistry(profiles=_three_bots()), run_fn, max_hops=2
    ).run(task="email the lead", originator="outreach", handoff=("dev", ["email the lead"]))
    assert reverse.hops[0].outcome == "completed"
    assert "may not call tool 'send_email'" in reverse.hops[0].detail
    assert "bot 'dev'" in reverse.hops[0].detail


# ---------------------------------------------------------------------------
# (d) Backward compatibility: no bots.json == today's single-agent behaviour
# ---------------------------------------------------------------------------


def test_no_bots_file_means_no_registry(monkeypatch):
    """The real path resolution, with no bots.json anywhere."""
    monkeypatch.delenv("DUAL_AGENT_BOTS_FILE", raising=False)
    reg = BotRegistry.load()
    assert reg.enabled is False
    assert reg.only_allow_listed_tools is False
    assert reg.source_path is None
    # The file it looked for lives under the (test-redirected) data dir.
    assert get_bot_config_path() == os.path.join(get_default_data_dir(), "bots.json")
    assert not os.path.exists(get_bot_config_path())


def test_no_bots_file_full_tool_surface_is_unchanged(monkeypatch):
    """Every tool the host had before is still there, via the registry path."""
    monkeypatch.delenv("DUAL_AGENT_BOTS_FILE", raising=False)
    host = MCPHost()
    before = sorted(host.get_tool_descriptions())
    reg = BotRegistry.load()
    wrapped = reg.wrap_mcp_host(host)
    assert wrapped is host
    assert sorted(wrapped.get_tool_descriptions()) == before
    assert wrapped.get_tool("run_shell_command") is not None
    assert wrapped.get_formatted_tool_list_for_system_two() == host.get_formatted_tool_list_for_system_two()


def test_a_run_that_used_an_unlisted_tool_now_refuses(monkeypatch, tmp_path, dispatcher):
    """Proves the previous test is not vacuous.

    Same dispatcher, same tool, same arguments: with no bots.json it executes;
    with a bots.json that omits the tool from the bot's allow-list it refuses.
    So 'unchanged' above is a real difference in state, not a no-op assertion.
    """
    monkeypatch.delenv("DUAL_AGENT_BOTS_FILE", raising=False)

    # 1. No bots.json -> the tool runs, as today.
    baseline = dispatcher.mcp.execute_tool("run_shell_command", {"command": "echo baseline"})
    assert baseline.success is True, baseline.error
    assert "baseline" in str(baseline.output)

    # 2. bots.json present, analyst does not hold run_shell_command -> refused.
    path = tmp_path / "bots.json"
    save_bots(_three_bots(), path=str(path))
    monkeypatch.setenv("DUAL_AGENT_BOTS_FILE", str(path))
    reg = BotRegistry.load()
    assert reg.enabled is True

    apply_bot_profile(dispatcher, reg.get("analyst"), reg)
    after = dispatcher.mcp.execute_tool("run_shell_command", {"command": "echo should-not-run"})
    assert after.success is False
    assert "analyst" in after.error and "run_shell_command" in after.error


def test_dispatcher_run_is_unaffected_without_bots(dispatcher):
    """A real run() on a real dispatcher with no bot stamping still completes."""
    from dual_agent.typesafe_client import JevSystemOneClient

    dispatcher.s1 = JevSystemOneClient(force_simulation=True)
    result = dispatcher.run(goal="list the files in this directory", max_steps=2)
    assert result.goal == "list the files in this directory"
    # Nothing about the run says it had a bot identity, because it did not.
    assert getattr(dispatcher, "bot_profile", None) is None
    assert getattr(result, "bot_id", None) is None
    assert result.total_steps >= 1
    assert result.used_simulated_system_one is True


def test_session_router_untouched_by_bot_layer(monkeypatch):
    """SessionRouter still returns one dispatcher per chat_id, unchanged."""
    from dual_agent.gateway.session_router import SessionRouter

    built = []

    class FakeDispatcher:
        def __init__(self, session_id):
            self.session_id = session_id

    def factory(**kwargs):
        d = FakeDispatcher(kwargs["session_id"])
        built.append(d)
        return d

    router = SessionRouter(dispatcher_factory=factory)
    a1 = router.get_or_create("chat-1")
    a2 = router.get_or_create("chat-1")
    b = router.get_or_create("chat-2")

    assert a1 is a2
    assert a1 is not b
    assert router.active_sessions == 2
    assert len(built) == 2
    # No bot identity is invented for a chat that never asked for one.
    assert not hasattr(a1, "bot_profile")


def test_all_default_tools_remain_reachable_for_a_bot_that_holds_them(registry):
    """A full-grant bot sees exactly the host's full surface — no accidental loss.

    The allow-list is built from the host's own tool list, so any tool the
    filter dropped would show up as a diff here.
    """
    host = MCPHost()
    expected = sorted(host.get_tool_descriptions())
    assert len(expected) >= 12, f"expected the host's built-in tools, got {expected}"

    everything = BotProfile(
        id="all",
        display_name="All",
        role="Generalist",
        allowed_tools=host.get_tool_descriptions().keys(),
    )
    reg = BotRegistry(profiles=[everything])
    wrapped = reg.wrap_mcp_host(host, bot_id="all")
    assert isinstance(wrapped, BotScopedMCPHost)

    assert sorted(wrapped.get_tool_descriptions()) == expected
    for name in expected:
        assert wrapped.get_tool(name) is not None, f"{name} was dropped by the filter"
    assert len(wrapped.list_tools()) == len(host.list_tools())
    # And a granted risky tool really runs through to the host.
    assert wrapped.get_tool("run_shell_command") is not None


def test_allow_list_accepts_any_key_iterable():
    """`allowed_tools` must survive a dict-keys view, not just a list.

    The full-grant test above passes `dict_keys`; if the model coerced that
    wrongly, that test would pass for the wrong reason.
    """
    host = MCPHost()
    profile = BotProfile(
        id="all2", display_name="All2", role="R", allowed_tools=host.get_tool_descriptions().keys()
    )
    assert sorted(profile.allowed_tools) == sorted(host.get_tool_descriptions())
    reg = BotRegistry(profiles=[profile])
    assert reg.check_tool("all2", "write_file") == (True, None)
