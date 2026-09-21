"""Tests for MemoryEngine SQLite persistence."""

import os
from dual_agent.memory import MemoryEngine, LearnedSkill


def test_memory_engine_sessions_and_stats(tmp_path):
    db_path = str(tmp_path / "test_memory.db")
    memory = MemoryEngine(db_path=db_path)

    # Initially empty
    stats = memory.get_aggregate_stats()
    assert stats["total_sessions"] == 0

    # Save a session
    row_id = memory.save_session(
        goal="Test goal",
        outcome="Test outcome",
        is_completed=True,
        total_steps=5,
        system_one_steps=4,
        system_two_steps=1,
        total_latency_ms=120.0,
        tokens_used=400,
        token_savings_pct=80.0,
        steps=[{"step": 1, "action": "list_directory"}],
    )
    assert row_id > 0

    # Verify session retrieval
    recent = memory.get_recent_sessions(limit=5)
    assert len(recent) == 1
    assert recent[0]["goal"] == "Test goal"
    assert recent[0]["system_one_steps"] == 4

    # Verify aggregate stats
    updated_stats = memory.get_aggregate_stats()
    assert updated_stats["total_sessions"] == 1
    assert updated_stats["total_steps"] == 5
    assert updated_stats["total_s1_steps"] == 4
    assert updated_stats["avg_token_savings_pct"] == 80.0


def test_memory_engine_skills(tmp_path):
    db_path = str(tmp_path / "skills.db")
    memory = MemoryEngine(db_path=db_path)

    # Save skill
    memory.save_learned_skill(
        name="inspect repository",
        intent_keywords=["inspect", "repository", "files"],
        tool_sequence=["list_directory", "read_file"],
    )

    all_skills = memory.get_all_skills()
    assert len(all_skills) == 1
    assert all_skills[0].name == "inspect repository"
    assert all_skills[0].tool_sequence == ["list_directory", "read_file"]

    # Match skill
    matched = memory.find_matching_skill("Please inspect repository files now")
    assert matched is not None
    assert matched.name == "inspect repository"

    # Increment count
    memory.save_learned_skill(
        name="inspect repository",
        intent_keywords=["inspect", "repository", "files"],
        tool_sequence=["list_directory", "read_file"],
    )
    reloaded = memory.get_all_skills()
    assert reloaded[0].success_count == 2


def test_memory_engine_project_context(tmp_path):
    db_path = str(tmp_path / "context.db")
    memory = MemoryEngine(db_path=db_path)

    workspace = "/path/to/project"
    tech = {"language": "python", "framework": "pytest"}
    prefs = {"style": "concise"}

    memory.save_project_context(workspace, tech, prefs)
    retrieved = memory.get_project_context(workspace)

    assert retrieved is not None
    assert retrieved["tech_stack"]["language"] == "python"
    assert retrieved["user_preferences"]["style"] == "concise"
