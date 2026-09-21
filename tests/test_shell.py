"""Tests for InteractiveShell and slash commands."""

from dual_agent.shell import InteractiveShell
from dual_agent.memory import MemoryEngine


def test_shell_slash_commands(tmp_path, capsys):
    test_db = str(tmp_path / "test_shell.db")
    memory = MemoryEngine(db_path=test_db)
    shell = InteractiveShell(memory=memory, force_simulation=True)

    # /help
    res = shell.handle_command("/help")
    assert res is True
    out = capsys.readouterr().out
    assert "Available Commands" in out

    # /tools
    res = shell.handle_command("/tools")
    assert res is True
    out = capsys.readouterr().out
    assert "Registered MCP Tools" in out
    assert "list_directory" in out

    # /memory
    res = shell.handle_command("/memory")
    assert res is True
    out = capsys.readouterr().out
    assert "Memory & Savings Statistics" in out

    # /exit
    res = shell.handle_command("/exit")
    assert res is False


def test_shell_task_execution(tmp_path, capsys):
    test_db = str(tmp_path / "test_shell_task.db")
    memory = MemoryEngine(db_path=test_db)
    shell = InteractiveShell(memory=memory, force_simulation=True)

    res = shell.handle_command("Inspect directory")
    assert res is True
    out = capsys.readouterr().out
    assert "Executing Task:" in out
    assert "Stats:" in out
