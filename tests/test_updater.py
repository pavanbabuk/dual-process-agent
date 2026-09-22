"""Tests for updater module."""

from dual_agent.updater import perform_update, get_git_root


def test_updater_get_git_root():
    root = get_git_root()
    assert root is not None
    assert "charming-hopper" in root or "dual-process-agent" in root


def test_updater_perform_update(capsys):
    success = perform_update()
    assert success is True
    out = capsys.readouterr().out
    assert "Dual-Process Agent Self-Updater" in out
    assert "System is up to date!" in out
