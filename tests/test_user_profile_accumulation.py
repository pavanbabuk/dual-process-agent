"""Regression test for Fix 5: completed runs must write durable facts to USER.md."""

from unittest.mock import MagicMock
from io import StringIO
from rich.console import Console
from dual_agent.dispatcher import DualProcessDispatcher
from dual_agent.typesafe_client import JevDecision
from dual_agent.memory import MemoryEngine
from dual_agent.mcp_host import MCPHost


def test_completed_run_updates_user_profile(tmp_path):
    mem_dir = tmp_path / "agent_data"
    mem_dir.mkdir()
    memory = MemoryEngine(db_path=str(mem_dir / "memory.db"))

    # Fresh install: no profile
    assert memory.get_user_profile() == ""

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
    )

    result = dispatcher.run(goal="inspect repository layout and structure")
    assert result.is_completed is True

    # USER.md must have been created and contain a durable fact
    profile = memory.get_user_profile()
    assert profile != "", "USER.md was not written on successful task completion"
    assert "inspect repository layout" in profile
    user_md_file = mem_dir / "USER.md"
    assert user_md_file.exists()


def test_whoami_displays_accumulated_profile(tmp_path, monkeypatch):
    from dual_agent.shell import InteractiveShell
    mem_dir = tmp_path / "agent_data"
    mem_dir.mkdir()
    memory = MemoryEngine(db_path=str(mem_dir / "memory.db"))

    shell = InteractiveShell(memory=memory)
    
    # Capture output before any run
    buf1 = StringIO()
    monkeypatch.setattr("dual_agent.shell.console", Console(file=buf1, color_system=None))
    shell._show_user_profile()
    assert "No USER.md profile yet" in buf1.getvalue()

    # Write a fact via update_user_profile (as dispatcher does)
    memory.update_user_profile("User prefers pytest over unittest")

    buf2 = StringIO()
    monkeypatch.setattr("dual_agent.shell.console", Console(file=buf2, color_system=None))
    shell._show_user_profile()
    out = buf2.getvalue()
    assert "No USER.md profile yet" not in out
    assert "User prefers pytest over unittest" in out

