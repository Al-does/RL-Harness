"""Tests for compact training curve projection."""

from __future__ import annotations

import json

from harness.training_curves import (
    TRAINING_CURVES_FILENAME,
    compact_training_curve_row,
    training_iteration_from_row,
)


def test_compact_training_curve_row_projects_short_keys():
    flattened = {
        "training_iteration": 3,
        "num_env_steps_sampled_lifetime": 98304.0,
        "env_runners/episode_return_mean": 12.5,
        "learners/default_policy/entropy": 1.2,
        "time_this_iter_s": 4.5,
        "env_runners/env_step_timer": 0.001,
    }
    row = compact_training_curve_row(flattened)
    assert row == {
        "iteration": 3,
        "steps": 98304.0,
        "return_mean": 12.5,
        "entropy": 1.2,
        "time_iter_s": 4.5,
    }


def test_training_iteration_from_row_accepts_compact_and_verbose():
    assert training_iteration_from_row({"iteration": 2}) == 2.0
    assert training_iteration_from_row({"training_iteration": 2}) == 2.0


def test_append_result_writes_compact_and_verbose(tmp_path):
    from harness.artifacts import RunArtifacts

    results = tmp_path / "results" / "run"
    artifacts = tmp_path / "artifacts" / "run"
    run = RunArtifacts(results, artifacts)
    run.append_result(
        {
            "training_iteration": 1,
            "episode_return_mean": 5.0,
            "learners": {"default_policy": {"entropy": 0.9, "policy_loss": 0.1}},
            "num_env_steps_sampled_lifetime": 32768,
            "time_this_iter_s": 2.0,
        }
    )

    compact_lines = run.training_curves_path.read_text().splitlines()
    verbose_lines = run.verbose_progress_path.read_text().splitlines()
    assert len(compact_lines) == 1
    assert len(verbose_lines) == 1
    compact = json.loads(compact_lines[0])
    verbose = json.loads(verbose_lines[0])
    assert compact["iteration"] == 1
    assert compact["return_mean"] == 5.0
    assert compact["entropy"] == 0.9
    assert "learners/default_policy/entropy" in verbose
    assert run.training_curves_path.name == TRAINING_CURVES_FILENAME
