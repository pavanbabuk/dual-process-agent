"""Tests for Phase 1 Fix 2: Evaluation Harness."""

import pytest
from dual_agent.eval_harness import (
    EvalTask,
    DEFAULT_EVAL_DATASET,
    EvaluationReport,
    run_evaluation_harness,
)


def test_eval_dataset_integrity():
    """Dataset holds goaled tasks with known ground-truth tool sequences."""
    assert len(DEFAULT_EVAL_DATASET) >= 4
    for task in DEFAULT_EVAL_DATASET:
        assert isinstance(task, EvalTask)
        assert task.goal
        assert len(task.expected_tools) >= 1


def test_eval_harness_runs_offline_and_is_deterministic():
    """Harness runs offline with simulated routing and produces deterministic measured report."""
    report1: EvaluationReport = run_evaluation_harness(
        tasks=DEFAULT_EVAL_DATASET,
        force_simulation=True,
    )
    report2: EvaluationReport = run_evaluation_harness(
        tasks=DEFAULT_EVAL_DATASET,
        force_simulation=True,
    )

    # Invariants on measurement fields
    assert 0.0 <= report1.routing_accuracy <= 1.0
    assert 0.0 <= report1.unnecessary_escalation_rate <= 1.0
    assert report1.dual_process_steps >= len(DEFAULT_EVAL_DATASET)
    assert report1.single_model_steps >= len(DEFAULT_EVAL_DATASET)
    assert report1.dual_process_cost_usd >= 0.0
    assert report1.single_model_cost_usd >= 0.0

    # Determinism assertion
    assert report1.routing_accuracy == report2.routing_accuracy
    assert report1.unnecessary_escalation_rate == report2.unnecessary_escalation_rate
    assert report1.dual_process_steps == report2.dual_process_steps
    assert report1.single_model_steps == report2.single_model_steps
    assert report1.dual_process_cost_usd == report2.dual_process_cost_usd
    assert report1.single_model_cost_usd == report2.single_model_cost_usd


def test_eval_report_format_plain_measurements_not_fabricated():
    """Report formatting displays plain measured values without fabricated savings claims."""
    report = run_evaluation_harness(
        tasks=DEFAULT_EVAL_DATASET,
        force_simulation=True,
    )
    summary_text = report.to_plain_text()
    assert "Routing Accuracy:" in summary_text
    assert "Unnecessary Escalation Rate:" in summary_text
    assert "Dual-Process Steps:" in summary_text
    assert "Baseline Steps:" in summary_text
    assert "Dual-Process Estimated Cost ($):" in summary_text
    assert "Baseline Estimated Cost ($):" in summary_text
