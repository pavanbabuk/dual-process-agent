"""Tests for SkillsManager (.SKILL.md synthesis and retrieval)."""

import os
import tempfile
import pytest
from dual_agent.skills_manager import SkillsManager, Skill


@pytest.fixture
def tmp_skills_dir(tmp_path):
    return str(tmp_path / "skills")


def test_synthesize_creates_skill_file(tmp_skills_dir):
    sm = SkillsManager(skills_dir=tmp_skills_dir)
    path = sm.synthesize_skill(
        goal="inspect pyproject.toml and find dependencies",
        tool_sequence=["list_directory", "read_file"],
        outcome="Found 5 dependencies.",
    )
    assert path is not None
    assert os.path.exists(path)
    assert path.endswith(".SKILL.md")


def test_synthesize_increments_count(tmp_skills_dir):
    sm = SkillsManager(skills_dir=tmp_skills_dir)
    path1 = sm.synthesize_skill("inspect pyproject", ["read_file"], "done")
    path2 = sm.synthesize_skill("inspect pyproject", ["read_file"], "done again")
    assert path1 == path2  # same file updated

    skill = sm._parse_file(path1)
    assert skill is not None
    assert skill.success_count == 2


def test_load_all_skills_returns_parsed(tmp_skills_dir):
    sm = SkillsManager(skills_dir=tmp_skills_dir)
    sm.synthesize_skill("run tests with pytest", ["run_shell_command"], "passed")
    sm.synthesize_skill("read config file content", ["read_file"], "done")

    skills = sm.load_all_skills()
    assert len(skills) == 2
    assert all(isinstance(s, Skill) for s in skills)


def test_find_relevant_skills_by_keywords(tmp_skills_dir):
    sm = SkillsManager(skills_dir=tmp_skills_dir)
    sm.synthesize_skill("inspect python dependencies from pyproject", ["read_file", "list_directory"], "ok")
    sm.synthesize_skill("run pytest test suite", ["run_shell_command"], "ok")

    relevant = sm.find_relevant_skills("list python dependencies", top_k=3)
    assert len(relevant) >= 1
    # The "inspect python dependencies" skill should rank highest
    assert "read_file" in relevant[0].tool_sequence or "list_directory" in relevant[0].tool_sequence


def test_build_skill_context_returns_string(tmp_skills_dir):
    sm = SkillsManager(skills_dir=tmp_skills_dir)
    sm.synthesize_skill("inspect workspace files", ["list_directory"], "listed")
    ctx = sm.build_skill_context("inspect workspace")
    assert "[RELEVANT SKILLS FROM MEMORY]" in ctx
    assert "list_directory" in ctx


def test_build_skill_context_empty_when_no_match(tmp_skills_dir):
    sm = SkillsManager(skills_dir=tmp_skills_dir)
    ctx = sm.build_skill_context("some totally unrelated query xyz")
    assert ctx == ""


def test_slugify_produces_safe_filename(tmp_skills_dir):
    sm = SkillsManager(skills_dir=tmp_skills_dir)
    slug = sm._slugify("Inspect pyproject.toml & find all MISSING deps!")
    assert " " not in slug
    assert "&" not in slug
    assert "!" not in slug
