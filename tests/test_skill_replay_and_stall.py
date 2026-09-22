"""Regression test for Fix 4: stall/exhaustion must not synthesize skills, and inert skill replay is removed."""

import logging
from unittest.mock import MagicMock
from dual_agent.dispatcher import DualProcessDispatcher
from dual_agent.typesafe_client import JevDecision
from dual_agent.state import StepRecord, StepType
from dual_agent.skills_manager import SkillsManager
from dual_agent.memory import MemoryEngine
from dual_agent.mcp_host import MCPHost, ToolExecutionResult


def test_stalled_run_does_not_synthesize_skill_and_is_not_completed(tmp_path, caplog):
    skills_dir = tmp_path / "skills"
    skills = SkillsManager(skills_dir=str(skills_dir))
    memory = MemoryEngine(db_path=str(tmp_path / "mem.db"))
    mcp = MCPHost()

    # S1 returns the same tool every time with confidence 0.95
    s1 = MagicMock()
    s1.evaluate_state_and_route.return_value = JevDecision(
        is_terminal=False,
        selected_tool="list_directory",
        confidence=0.95,
        latency_ms=10.0,
        needs_generation=False,
    )
    s1.force_simulation = False
    s1.simulation_reason = None

    dispatcher = DualProcessDispatcher(
        system_one_client=s1,
        mcp_host=mcp,
        memory_engine=memory,
        skills_manager=skills,
    )

    with caplog.at_level(logging.INFO):
        res = dispatcher.run(goal="inspect directory listing", max_steps=5)

    # 1. Stalled run terminates early with stall message
    assert "not progressing" in (res.final_output or "")
    assert res.total_steps < 5
    
    # 2. Must not have synthesized any skill files or stored learned skills
    created_skills = list(skills_dir.glob("*.SKILL.md"))
    assert len(created_skills) == 0, f"Stalled run must not synthesize skill files, found: {created_skills}"
    assert len(memory.get_all_skills()) == 0, "Stalled run must not save learned skill to memory"


def test_exhausted_max_steps_does_not_synthesize_skill(tmp_path):
    skills_dir = tmp_path / "skills"
    skills = SkillsManager(skills_dir=str(skills_dir))
    memory = MemoryEngine(db_path=str(tmp_path / "mem.db"))
    mcp = MCPHost()

    # Return different tools to avoid stall detection, but never terminate
    tools = ["list_directory", "get_system_metrics", "list_directory", "get_system_metrics"]
    tool_iter = iter(tools)
    s1 = MagicMock()
    s1.evaluate_state_and_route.side_effect = lambda *args, **kwargs: JevDecision(
        is_terminal=False,
        selected_tool=next(tool_iter, "list_directory"),
        confidence=0.95,
        latency_ms=10.0,
        needs_generation=False,
    )
    s1.force_simulation = False
    s1.simulation_reason = None

    s2 = MagicMock()
    from dual_agent.system_two import SystemTwoResponse
    s2.generate_step.return_value = SystemTwoResponse(
        action="list_directory",
        arguments={"path": "."},
        thought="inspecting",
        latency_ms=10.0,
        tokens_used=10,
        is_mock=False,
    )

    dispatcher = DualProcessDispatcher(
        system_one_client=s1,
        system_two_provider=s2,
        mcp_host=mcp,
        memory_engine=memory,
        skills_manager=skills,
    )

    res = dispatcher.run(goal="inspect files and metrics", max_steps=3)
    assert res.is_completed is False
    assert len(list(skills_dir.glob("*.SKILL.md"))) == 0
    assert len(memory.get_all_skills()) == 0


def test_no_inert_learned_skill_replay_log(tmp_path, caplog):
    skills_dir = tmp_path / "skills"
    skills = SkillsManager(skills_dir=str(skills_dir))
    memory = MemoryEngine(db_path=str(tmp_path / "mem.db"))
    
    # Pre-populate a matching skill in memory
    memory.save_learned_skill(
        name="inspect files",
        intent_keywords=["inspect", "files"],
        tool_sequence=["list_directory"],
    )

    s1 = MagicMock()
    s1.evaluate_state_and_route.return_value = JevDecision(
        is_terminal=True,
        selected_tool="finish_task",
        confidence=0.95,
        latency_ms=10.0,
    )
    s1.force_simulation = False
    s1.simulation_reason = None

    dispatcher = DualProcessDispatcher(
        system_one_client=s1,
        memory_engine=memory,
        skills_manager=skills,
    )

    with caplog.at_level(logging.INFO):
        dispatcher.run(goal="inspect files")

    # There should NOT be an inert log line claiming replay happened
    assert "[Learned Skill Replay]" not in caplog.text
