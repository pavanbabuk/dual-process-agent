"""Focused tests for the offline (simulation-mode) router in JevSystemOneClient.

These tests exist because the offline router used to pick a tool by checking
whether the LAST WORD of its name appeared anywhere in the state text, and it
declared the run finished only on two literal phrases. That made offline mode
useless as a router: `list_directory` beat `read_file` on any goal containing
the word "directory", ties went to whichever tool came first in dict order, and
real runs looped until max_steps.

Every client below is built with force_simulation=True. That is not cosmetic:
the machine this test suite runs on may hold a live TypeSafe Jev API key, and a
client constructed without force_simulation would issue real, billed calls.
"""

import pytest

from dual_agent.typesafe_client import JevDecision, JevSystemOneClient

CONTROL_OPTIONS = ("escalate_to_system_two", "finish_task")


def _client() -> JevSystemOneClient:
    """Simulation-only client. Never live, so no request is ever billed."""
    return JevSystemOneClient(force_simulation=True)


def _tools() -> dict:
    """The offline tool set, with the descriptions the dispatcher really passes.

    Descriptions mirror `MCPHost.get_tool_descriptions()`, which is what the
    router scores against in production.
    """
    return {
        "list_directory": "List contents of a local directory or inspect workspace files.",
        "read_file": "Read contents of a text file from workspace.",
        "write_file": "Create or overwrite a file in the workspace.",
        "run_shell_command": "Execute a shell command in the local environment and return stdout/stderr.",
    }


def _state(goal: str, actions: str = "No prior actions.", step: str = "1/6") -> str:
    """Build a state blob shaped exactly like AgentState.to_system_one_state()."""
    return (
        f"GOAL: {goal}\n"
        f"CURRENT STEP: {step}\n"
        f"RECENT ACTIONS: {actions}\n"
        f"VARIABLES: []"
    )


# --------------------------------------------------------------------------
# (a) A goal naming a file must route to read_file, not list_directory.
# --------------------------------------------------------------------------

def test_file_goal_routes_to_read_file_not_list_directory():
    decision = _client().evaluate_state_and_route(
        state_text=_state("read the file pyproject.toml and report its dependencies"),
        tool_options=_tools(),
    )

    assert decision.selected_tool == "read_file"
    assert decision.probabilities["read_file"] > decision.probabilities["list_directory"]


@pytest.mark.parametrize(
    "goal",
    [
        "read config.yaml",
        "open notes.md and tell me what is in it",
        "show me the contents of ./src/dual_agent/state.py",
    ],
)
def test_various_file_goals_route_to_the_file_reader(goal):
    decision = _client().evaluate_state_and_route(
        state_text=_state(goal),
        tool_options=_tools(),
    )
    assert decision.selected_tool == "read_file"


# --------------------------------------------------------------------------
# (b) A goal about listing a directory must route to list_directory.
# --------------------------------------------------------------------------

def test_directory_goal_routes_to_list_directory():
    decision = _client().evaluate_state_and_route(
        state_text=_state("list the files in the workspace directory"),
        tool_options=_tools(),
    )

    assert decision.selected_tool == "list_directory"
    assert decision.probabilities["list_directory"] > decision.probabilities["read_file"]


def test_directory_goal_without_a_file_token_still_routes_to_listing():
    decision = _client().evaluate_state_and_route(
        state_text=_state("what is in ./build/ right now?"),
        tool_options=_tools(),
    )
    assert decision.selected_tool == "list_directory"


# --------------------------------------------------------------------------
# (c) A real probability distribution over all options plus the two controls.
# --------------------------------------------------------------------------

def test_probabilities_cover_every_option_and_sum_to_one():
    tools = _tools()
    decision = _client().evaluate_state_and_route(
        state_text=_state("read pyproject.toml"),
        tool_options=tools,
    )

    assert isinstance(decision.probabilities, dict)
    for name in list(tools) + list(CONTROL_OPTIONS):
        assert name in decision.probabilities, f"missing probability for {name}"

    assert set(decision.probabilities) == set(tools) | set(CONTROL_OPTIONS)
    assert sum(decision.probabilities.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 <= p <= 1.0 for p in decision.probabilities.values())

    # confidence is the winner's probability, not an independent constant.
    assert decision.confidence == pytest.approx(decision.probabilities[decision.selected_tool])


def test_control_options_are_present_even_with_no_tools_registered():
    decision = _client().evaluate_state_and_route(
        state_text=_state("do something interesting"),
        tool_options={},
    )

    assert set(decision.probabilities) == set(CONTROL_OPTIONS)
    assert sum(decision.probabilities.values()) == pytest.approx(1.0, abs=1e-9)
    assert decision.selected_tool == "escalate_to_system_two"


def test_decision_marks_itself_simulated_with_a_reason():
    decision = _client().evaluate_state_and_route(
        state_text=_state("read pyproject.toml"),
        tool_options=_tools(),
    )

    assert isinstance(decision, JevDecision)
    assert decision.simulated is True
    assert decision.fallback_reason == "force_simulation=True"
    assert decision.latency_ms > 0.0


# --------------------------------------------------------------------------
# (d) Synthesis/writing goals no registered tool can satisfy.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "goal",
    [
        "summarize the repository for a new contributor",
        "write a short poem about the build system",
        "draft a design doc for the router",
        "explain how the two-process loop works",
        "refactor the dispatcher into smaller functions",
    ],
)
def test_synthesis_goals_set_needs_generation(goal):
    decision = _client().evaluate_state_and_route(
        state_text=_state(goal),
        tool_options=_tools(),
    )
    assert decision.needs_generation is True


def test_plain_tool_goal_does_not_set_needs_generation():
    decision = _client().evaluate_state_and_route(
        state_text=_state("list the files in the workspace directory"),
        tool_options=_tools(),
    )
    assert decision.needs_generation is False


# --------------------------------------------------------------------------
# (e) Termination: the goal's work is already done.
# --------------------------------------------------------------------------

def test_completed_recent_actions_make_the_decision_terminal():
    """The goal asked to read a file and the history shows it was read."""
    state_text = (
        "GOAL: read the file pyproject.toml\n"
        "CURRENT STEP: 3/6\n"
        "RECENT ACTIONS: Step 1: read_file -> Success: read 412 characters from "
        "pyproject.toml | Step 2: read_file -> Success: read 412 characters from "
        "pyproject.toml\n"
        "VARIABLES: []"
    )

    decision = _client().evaluate_state_and_route(state_text=state_text, tool_options=_tools())

    assert decision.is_terminal is True
    assert decision.selected_tool == "finish_task"
    # A finished goal has nothing left to generate.
    assert decision.needs_generation is False


def test_terminal_when_last_outputs_repeat_with_no_progress():
    """Identical action AND identical output twice: stalled, so stop."""
    state_text = (
        "GOAL: inspect the workspace\n"
        "CURRENT STEP: 4/6\n"
        "RECENT ACTIONS: Step 1: list_directory -> entries: [] total: 0 | "
        "Step 2: list_directory -> entries: [] total: 0\n"
        "VARIABLES: []"
    )

    decision = _client().evaluate_state_and_route(state_text=state_text, tool_options=_tools())

    assert decision.is_terminal is True


def test_fresh_goal_with_no_useful_history_is_not_terminal():
    decision = _client().evaluate_state_and_route(
        state_text=_state("read the file pyproject.toml"),
        tool_options=_tools(),
    )
    assert decision.is_terminal is False
    assert decision.selected_tool != "finish_task"


def test_failed_step_does_not_count_as_completed_work():
    """An error in the history is not evidence the goal was satisfied."""
    state_text = (
        "GOAL: read the file missing.py\n"
        "CURRENT STEP: 2/6\n"
        "RECENT ACTIONS: Step 1: read_file -> Error: File 'missing.py' does not exist.\n"
        "VARIABLES: []"
    )

    decision = _client().evaluate_state_and_route(state_text=state_text, tool_options=_tools())

    assert decision.is_terminal is False


# --------------------------------------------------------------------------
# (f) Determinism.
# --------------------------------------------------------------------------

def test_same_state_yields_an_identical_decision_twice():
    state_text = _state(
        "read pyproject.toml and list the workspace directory",
        actions="Step 1: list_directory -> entries: ['a.py'] total: 1",
        step="2/6",
    )

    first = _client().evaluate_state_and_route(state_text=state_text, tool_options=_tools())
    second = _client().evaluate_state_and_route(state_text=state_text, tool_options=_tools())

    assert first.selected_tool == second.selected_tool
    assert first.confidence == second.confidence
    assert first.probabilities == second.probabilities
    assert first.is_terminal == second.is_terminal
    assert first.needs_generation == second.needs_generation


def test_candidate_ordering_does_not_change_the_route():
    """Ties must not fall to whichever candidate came first in the dict."""
    state_text = _state("list the files in the workspace directory")

    forward = _client().evaluate_state_and_route(state_text=state_text, tool_options=_tools())
    reversed_tools = dict(reversed(list(_tools().items())))
    backward = _client().evaluate_state_and_route(state_text=state_text, tool_options=reversed_tools)

    assert forward.selected_tool == backward.selected_tool == "list_directory"
    assert forward.probabilities == backward.probabilities
