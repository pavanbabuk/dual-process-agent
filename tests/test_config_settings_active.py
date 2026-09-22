"""Regression test for Fix 9: AgentConfig.auto_learn_skills and auto_scan_workspace must be active, not dead."""

import os
from unittest.mock import MagicMock
from dual_agent.config import AgentConfig
from dual_agent.dispatcher import DualProcessDispatcher
from dual_agent.typesafe_client import JevDecision
from dual_agent.memory import MemoryEngine
from dual_agent.skills_manager import SkillsManager


def test_auto_learn_skills_false_suppresses_skill_synthesis(tmp_path):
    skills_dir = tmp_path / "skills"
    skills = SkillsManager(skills_dir=str(skills_dir))
    memory = MemoryEngine(db_path=str(tmp_path / "mem.db"))

    # Config with auto_learn_skills = False
    config = AgentConfig(auto_learn_skills=False)

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
        config=config,
    )

    res = dispatcher.run(goal="inspect some codebase structure")
    assert res.is_completed is True
    # Must NOT synthesize skills or save learned skills when auto_learn_skills is False
    assert len(list(skills_dir.glob("*.SKILL.md"))) == 0
    assert len(memory.get_all_skills()) == 0


def test_auto_scan_workspace_populates_project_context(tmp_path):
    memory = MemoryEngine(db_path=str(tmp_path / "mem.db"))
    config = AgentConfig(auto_scan_workspace=True)

    # Create dummy pyproject.toml in the workspace
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test-pkg'\n")

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
        config=config,
    )

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        res = dispatcher.run(goal="inspect the workspace")
        assert res.is_completed is True

        # Project context must have been scanned and saved
        ctx = memory.get_project_context(str(tmp_path))
        assert ctx is not None, "auto_scan_workspace=True must record project context in memory"
        assert "python" in ctx.get("tech_stack", {}).get("languages", []) or ctx.get("tech_stack", {}).get("has_pyproject") is True
    finally:
        os.chdir(cwd)


def test_unimplemented_anthropic_fails_loudly():
    import pytest
    from dual_agent.system_two import get_system_two_provider

    with pytest.raises(ValueError, match="not implemented"):
        get_system_two_provider("anthropic")

