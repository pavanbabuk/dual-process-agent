"""Tests for Fix 10: mid-run recall context recomputation and expanded cron syntax support."""

import datetime
from unittest.mock import MagicMock
from dual_agent.dispatcher import DualProcessDispatcher
from dual_agent.typesafe_client import JevDecision
from dual_agent.system_two import SystemTwoResponse
from dual_agent.memory import MemoryEngine
from dual_agent.scheduler import CronScheduler


def test_cron_is_due_supports_extended_syntax():
    scheduler = CronScheduler(memory_engine=MagicMock())

    # Tuesday (dow=2), 2026-06-16 14:30:00 UTC
    dt = datetime.datetime(2026, 6, 16, 14, 30, 0, tzinfo=datetime.timezone.utc)

    # 1. Wildcards
    assert scheduler._is_due("* * * * *", None, dt) is True

    # 2. Step notation
    assert scheduler._is_due("*/15 * * * *", None, dt) is True
    assert scheduler._is_due("*/20 * * * *", None, dt) is False  # 30 % 20 != 0

    # 3. Exact values
    assert scheduler._is_due("30 14 16 6 *", None, dt) is True
    assert scheduler._is_due("31 14 16 6 *", None, dt) is False

    # 4. Comma-separated lists
    assert scheduler._is_due("15,30,45 * * * *", None, dt) is True
    assert scheduler._is_due("0,15,45 * * * *", None, dt) is False

    # 5. Ranges
    assert scheduler._is_due("20-35 * * * *", None, dt) is True
    assert scheduler._is_due("0-20 * * * *", None, dt) is False

    # 6. Range with step
    assert scheduler._is_due("20-40/5 * * * *", None, dt) is True
    assert scheduler._is_due("20-40/4 * * * *", None, dt) is False  # (30-20) % 4 != 0

    # 7. Day of week names (dt is Tuesday = 2)
    assert scheduler._is_due("* * * * tue", None, dt) is True
    assert scheduler._is_due("* * * * mon-fri", None, dt) is True
    assert scheduler._is_due("* * * * sat,sun", None, dt) is False

    # 8. Month names (dt is June = 6 = jun)
    assert scheduler._is_due("* * * jun *", None, dt) is True
    assert scheduler._is_due("* * * jan-mar *", None, dt) is False


def test_recall_context_updates_mid_run(tmp_path):
    mem_dir = tmp_path / "mem"
    mem_dir.mkdir()
    memory = MemoryEngine(db_path=str(mem_dir / "mem.db"))

    # Save a past session that mentions "run_shell_command" and "deploy"
    memory.save_session(
        goal="run_shell_command deploy service",
        outcome="Successfully deployed cluster service",
        is_completed=True,
        total_steps=1,
        system_one_steps=1,
        system_two_steps=0,
        total_latency_ms=10.0,
        tokens_used=0,
        token_savings_pct=None,
        steps=[],
    )

    # Initial goal does not mention "deploy" or "shell"
    goal = "inspect repository files"
    
    # Verify initially that recall context does NOT match the deploy session
    initial_recall = memory.build_recall_context(goal)
    assert "Successfully deployed cluster service" not in initial_recall

    # Run dispatcher where step 1 executes run_shell_command, escalates to S2 on step 2
    s1 = MagicMock()
    step_num = [0]
    def s1_route(*args, **kwargs):
        step_num[0] += 1
        if step_num[0] == 1:
            return JevDecision(
                is_terminal=False,
                selected_tool="list_directory",
                confidence=0.95,
                needs_generation=False,
            )
        # Step 2: escalate to S2
        return JevDecision(
            is_terminal=False,
            selected_tool="run_shell_command",
            confidence=0.1,
            needs_generation=True,
        )
    s1.evaluate_state_and_route.side_effect = s1_route
    s1.force_simulation = False
    s1.simulation_reason = None

    captured_prompts = []
    s2 = MagicMock()
    def s2_generate(prompt):
        captured_prompts.append(prompt)
        return SystemTwoResponse(
            action="finish_task",
            arguments={},
            thought="finished",
            generated_content="Done",
            latency_ms=50.0,
            tokens_used=10,
            is_mock=False,
        )
    s2.generate_step.side_effect = s2_generate

    dispatcher = DualProcessDispatcher(
        system_one_client=s1,
        system_two_provider=s2,
        memory_engine=memory,
        confidence_threshold=0.85,
    )

    res = dispatcher.run(goal=goal, max_steps=3)
    assert res.is_completed is True
    assert len(captured_prompts) >= 1
    
    # Prompt passed to System 2 on step 2 must reflect earlier action terms
    # and contain the newly matched past session outcome that was absent in initial recall
    s2_prompt = captured_prompts[0]
    assert "Successfully deployed cluster service" in s2_prompt, f"Expected past session in prompt, got: {s2_prompt}"

