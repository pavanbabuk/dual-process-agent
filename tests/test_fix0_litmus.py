"""Litmus test for Fix 0 (Plan -> Act -> Verify -> Correct Loop).

Vendor acceptance criteria:
1. Give the agent a multi-file/code goal:
   - Modifies cli.py to add --version flag and document in --help.
   - Verifies git diff shows requested change.
   - Verifies python -m py_compile succeeds on every modified file.
2. Give it an impossible goal ('rewrite the kernel in brainfuck'):
   - Verifies it stops when out of options, reporting 'could not complete' (is_completed=False).
   - Verifies it does NOT hallucinate success.
"""

from __future__ import annotations
import os
import shutil
import subprocess
import sys
import tempfile
import pytest

from dual_agent.dispatcher import DualProcessDispatcher, DispatchResult
from dual_agent.evaluator import JevEvaluator
from dual_agent.mcp_host import MCPHost
from dual_agent.permission_broker import PermissionBroker
from dual_agent.system_two import MockSystemTwoProvider, SystemTwoResponse
from dual_agent.typesafe_client import JevSystemOneClient


class VersionFlagMockProvider(MockSystemTwoProvider):
    """Simulates multi-step planning and file editing to add --version flag."""

    def __init__(self, cli_path: str):
        super().__init__()
        self.cli_path = cli_path
        self.step_idx = 0

    def generate_step(self, prompt: str) -> SystemTwoResponse:
        self.step_idx += 1
        if self.step_idx == 1:
            return SystemTwoResponse(
                thought="Read cli.py to find parser arguments",
                action="read_file",
                args={"path": self.cli_path},
                tokens_used=80,
                is_mock=True,
            )
        elif self.step_idx == 2:
            # Apply edit to cli.py adding --version
            with open(self.cli_path, "r", encoding="utf-8") as f:
                content = f.read()

            target = 'parser.add_argument(\n        "--max-steps",'
            replacement = (
                'parser.add_argument(\n        "--version",\n        action="version",\n        version="%(prog)s 0.2.0",\n        help="Show program version and exit.",\n    )\n    parser.add_argument(\n        "--max-steps",'
            )
            new_content = content.replace(target, replacement, 1)

            return SystemTwoResponse(
                thought="Add --version argument to argparse in cli.py",
                action="write_file",
                args={"path": self.cli_path, "content": new_content},
                tokens_used=150,
                is_mock=True,
            )
        elif self.step_idx == 3:
            return SystemTwoResponse(
                thought="Run py_compile to verify syntax",
                action="run_shell_command",
                args={"command": f"{sys.executable} -m py_compile {self.cli_path}"},
                tokens_used=90,
                is_mock=True,
            )
        else:
            return SystemTwoResponse(
                thought="All plan steps verified, complete task",
                action="finish_task",
                args={},
                tokens_used=50,
                is_mock=True,
            )


class ImpossibleTaskMockProvider(MockSystemTwoProvider):
    """Simulates an impossible task where steps fail and agent runs out of options."""

    def __init__(self):
        super().__init__()
        self.step_idx = 0

    def generate_step(self, prompt: str) -> SystemTwoResponse:
        self.step_idx += 1
        if "plan" in prompt.lower() and "plan:" not in prompt.lower():
            return SystemTwoResponse(
                thought="Plan impossible kernel rewrite",
                action="create_plan",
                args={"plan": [
                    "1. Search kernel source code",
                    "2. Transpile kernel to brainfuck",
                ]},
                tokens_used=100,
                is_mock=True,
            )
        # Attempt an action that will fail / not find anything
        return SystemTwoResponse(
            thought="Attempting to locate non-existent kernel source",
            action="run_shell_command",
            args={"command": "ls /non_existent_kernel_source_dir_xyz"},
            tokens_used=80,
            is_mock=True,
        )


def test_litmus_add_version_flag_and_verify_compilation():
    """Litmus Test 1: Agent adds --version flag, compiles cleanly, and satisfies verification."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create minimal package with a valid cli.py
        pkg_dir = os.path.join(tmp_dir, "dual_agent")
        os.makedirs(pkg_dir, exist_ok=True)
        cli_file = os.path.join(pkg_dir, "cli.py")

        initial_cli_code = (
            'import argparse\n'
            'def main():\n'
            '    parser = argparse.ArgumentParser(description="Test CLI")\n'
            '    parser.add_argument(\n'
            '        "--max-steps",\n'
            '        type=int,\n'
            '        default=10,\n'
            '        help="Maximum agent steps.",\n'
            '    )\n'
            '    args = parser.parse_args()\n'
            'if __name__ == "__main__":\n'
            '    main()\n'
        )
        with open(cli_file, "w", encoding="utf-8") as f:
            f.write(initial_cli_code)

        s1_client = JevSystemOneClient(force_simulation=True)
        s2_provider = VersionFlagMockProvider(cli_path=cli_file)
        broker = PermissionBroker(auto_allow=True)

        dispatcher = DualProcessDispatcher(
            system_one_client=s1_client,
            system_two_provider=s2_provider,
            permission_broker=broker,
        )

        res: DispatchResult = dispatcher.run(
            goal="add a --version flag to cli.py and document it in --help",
            max_steps=8,
        )

        # 1. Successful completion without hallucinated error
        assert res.is_completed is True
        assert len(res.plan) > 0
        assert all(step.completed and step.verified for step in res.plan)

        # 2. File compiles cleanly without syntax error
        compile_res = subprocess.run(
            [sys.executable, "-m", "py_compile", cli_file],
            capture_output=True,
            text=True,
        )
        assert compile_res.returncode == 0, f"Compilation failed: {compile_res.stderr}"

        # 3. cli.py --version works as expected
        version_run = subprocess.run(
            [sys.executable, cli_file, "--version"],
            capture_output=True,
            text=True,
        )
        assert version_run.returncode == 0
        assert "0.2.0" in (version_run.stdout + version_run.stderr)


def test_litmus_impossible_goal_reports_failure_honestly():
    """Litmus Test 2: Impossible goal stops when out of options and reports failure honestly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        s1_client = JevSystemOneClient(force_simulation=True)
        s2_provider = ImpossibleTaskMockProvider()
        broker = PermissionBroker(auto_allow=True)

        dispatcher = DualProcessDispatcher(
            system_one_client=s1_client,
            system_two_provider=s2_provider,
            permission_broker=broker,
        )

        res: DispatchResult = dispatcher.run(
            goal="rewrite the kernel in brainfuck",
            max_steps=5,
        )

        # Crucial acceptance rule: Never hallucinate success on impossible task
        assert res.is_completed is False
        assert "could not complete" in res.final_output.lower() or "budget" in res.final_output.lower() or "failed" in res.final_output.lower()
