"""Tests for harness metrics recording."""

from __future__ import annotations

import json

from harness.artifacts import METRICS_FILENAME, RunArtifacts


def test_append_result_writes_verbose_metrics_to_artifacts(tmp_path):
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

    lines = run.metrics_path.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["training_iteration"] == 1
    assert row["learners/default_policy/entropy"] == 0.9
    assert run.metrics_path.name == METRICS_FILENAME
    assert not (results / "training_curves.jsonl").exists()


def test_append_jsonl_supports_results_and_artifacts(tmp_path):
    run = RunArtifacts(tmp_path / "results", tmp_path / "artifacts")
    run.append_jsonl("custom.jsonl", {"value": 1}, dest="results")
    run.append_jsonl("custom.jsonl", {"value": 2}, dest="artifacts")
    assert json.loads(
        (tmp_path / "results" / "custom.jsonl").read_text().strip()
    ) == {"value": 1}
    assert json.loads(
        (tmp_path / "artifacts" / "custom.jsonl").read_text().strip()
    ) == {"value": 2}
