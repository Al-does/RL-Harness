"""Contract tests for the generic experiment harness."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from harness.artifacts import (
    RunArtifacts,
    finish_run_manifest,
    record_result,
    start_run_manifest,
)
from harness.cli import execute_experiment, load_experiment, make_run_context
from harness.context import RunContext
from harness.runners import build_tuner, run_algorithm, run_tune
from harness.seeding import (
    child_seed_sequence,
    named_seed_sequences,
    seed_sequence_to_int,
)


def make_context(tmp_path: Path, **overrides) -> RunContext:
    values = {
        "experiment_dir": tmp_path / "experiment",
        "results_dir": tmp_path / "experiment" / "results" / "run",
        "artifacts_dir": tmp_path / "experiment" / "artifacts" / "run",
        "run_id": "run",
    }
    values.update(overrides)
    return RunContext(**values)


def test_run_context_defaults_to_seed_42_and_is_immutable(tmp_path):
    context = make_context(tmp_path)

    assert context.seed == 42
    with pytest.raises(FrozenInstanceError):
        context.seed = 7
    with pytest.raises(ValueError, match="distinct"):
        make_context(
            tmp_path,
            results_dir=tmp_path / "same",
            artifacts_dir=tmp_path / "same",
        )


def test_named_seed_streams_are_stable_distinct_and_accept_zero():
    keys = {
        "probe_train": (0,),
        "probe_test": (1,),
    }
    reordered_keys = {
        "probe_test": (1,),
        "probe_train": (0,),
    }

    first = named_seed_sequences(0, keys)
    second = named_seed_sequences(0, keys)
    reordered = named_seed_sequences(0, reordered_keys)
    first_values = {
        name: np.random.default_rng(stream).integers(2**31, size=8)
        for name, stream in first.items()
    }
    second_values = {
        name: np.random.default_rng(stream).integers(2**31, size=8)
        for name, stream in second.items()
    }
    reordered_values = {
        name: np.random.default_rng(stream).integers(2**31, size=8)
        for name, stream in reordered.items()
    }

    np.testing.assert_array_equal(
        first_values["probe_train"],
        second_values["probe_train"],
    )
    np.testing.assert_array_equal(
        first_values["probe_test"],
        second_values["probe_test"],
    )
    np.testing.assert_array_equal(
        first_values["probe_train"],
        reordered_values["probe_train"],
    )
    assert not np.array_equal(
        first_values["probe_train"],
        first_values["probe_test"],
    )
    assert seed_sequence_to_int(child_seed_sequence(0, (0, 3))) == (
        seed_sequence_to_int(child_seed_sequence(0, (0, 3)))
    )
    assert seed_sequence_to_int(child_seed_sequence(0, (0, 3))) != (
        seed_sequence_to_int(child_seed_sequence(0, (1, 3)))
    )


def test_results_artifacts_manifest_and_metrics_remain_separate(tmp_path):
    source = tmp_path / "experiment" / "experiment.py"
    source.parent.mkdir()
    source.write_text("def run(context):\n    return None\n")
    context = make_context(tmp_path)

    manifest = start_run_manifest(
        context,
        experiment_module="example.experiment",
        experiment_file=source,
        command=["rl-harness", "example.experiment"],
        runtime_overrides={"seed": 42},
    )
    record_result(
        context,
        {
            "training_iteration": 1,
            "env_runners": {"episode_return_mean": 2.5},
            "histogram": [1, 2, 3],
        },
    )
    completed = finish_run_manifest(context, status="completed")

    paths = RunArtifacts.from_context(context)
    assert paths.manifest_path.parent == context.results_dir
    assert paths.checkpoints_dir.is_relative_to(context.artifacts_dir)
    assert not context.results_dir.is_relative_to(context.artifacts_dir)
    assert not context.artifacts_dir.is_relative_to(context.results_dir)
    assert manifest["runtime"]["seed"] == 42
    assert manifest["experiment"]["source_sha256"]
    assert completed["status"] == "completed"
    progress = json.loads(paths.progress_path.read_text())
    assert progress["training_iteration"] == 1
    assert progress["env_runners/episode_return_mean"] == 2.5
    assert "histogram" not in progress


class FakeAlgorithm:
    def __init__(self, results):
        self.results = iter(results)
        self.stopped = False
        self.saved_paths = []

    def train(self):
        return next(self.results)

    def stop(self):
        self.stopped = True

    def save_to_path(self, path):
        self.saved_paths.append(Path(path))
        return str(path)


class FakeConfig:
    algo_class = object

    def __init__(self, algorithm=None, param_space=None):
        self.algorithm = algorithm
        self.param_space = param_space or {"seed": 42}

    def build_algo(self):
        return self.algorithm

    def to_dict(self):
        return dict(self.param_space)


def test_direct_runner_stops_records_and_cleans_up(tmp_path):
    algorithm = FakeAlgorithm(
        [{"training_iteration": 1}, {"training_iteration": 2}]
    )
    records = []
    context = make_context(tmp_path)

    final = run_algorithm(
        FakeConfig(algorithm),
        context,
        should_stop=lambda result: result["training_iteration"] >= 2,
        recorder=lambda ctx, result: records.append((ctx, result)),
        checkpoint_interval=2,
        checkpoint_at_end=True,
    )

    assert final["training_iteration"] == 2
    assert [result["training_iteration"] for _, result in records] == [1, 2]
    assert algorithm.stopped
    assert algorithm.saved_paths == [
        context.artifacts_dir / "checkpoints" / "iteration_000002"
    ]


def test_direct_runner_stops_algorithm_when_stopping_check_raises(tmp_path):
    algorithm = FakeAlgorithm([{"training_iteration": 1}])

    with pytest.raises(RuntimeError, match="bad stop"):
        run_algorithm(
            FakeConfig(algorithm),
            make_context(tmp_path),
            should_stop=lambda result: (_ for _ in ()).throw(
                RuntimeError("bad stop")
            ),
            recorder=lambda context, result: None,
        )

    assert algorithm.stopped


def test_tune_single_trial_construction_uses_artifact_storage(
    tmp_path, monkeypatch
):
    from ray import tune

    captured = {}

    class CapturingTuner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(tune, "Tuner", CapturingTuner)
    context = make_context(tmp_path)
    config = FakeConfig(param_space={"seed": 42})

    tuner = build_tuner(
        config,
        context,
        stop={"training_iteration": 1},
    )

    assert isinstance(tuner, CapturingTuner)
    assert captured["trainable"] is object
    assert captured["param_space"] == {"seed": 42}
    assert captured["tune_config"] is None
    assert captured["run_config"].storage_path == str(context.artifacts_dir)
    assert captured["run_config"].name == "tune"


def test_tune_runner_writes_compact_trial_summary(tmp_path, monkeypatch):
    checkpoint = SimpleNamespace(path="/tmp/checkpoint")
    result = SimpleNamespace(
        metrics={
            "trial_id": "trial-1",
            "training_iteration": 1,
            "config": {"large": {"tree": True}},
        },
        checkpoint=checkpoint,
        config={"seed": 7},
        path="/tmp/trial",
        error=None,
    )
    tuner = SimpleNamespace(fit=lambda: [result])
    monkeypatch.setattr(
        "harness.runners.build_tuner",
        lambda *args, **kwargs: tuner,
    )
    context = make_context(tmp_path)

    returned = run_tune(
        FakeConfig(),
        context,
        stop={"training_iteration": 1},
    )

    assert returned == [result]
    summary = json.loads(
        (context.results_dir / "tune_summary.json").read_text()
    )
    assert summary["trials"][0]["resolved_seed"] == 7
    assert "config/large/tree" not in summary["trials"][0]["metrics"]


def test_cli_loads_leaf_and_records_success(tmp_path, monkeypatch):
    package = tmp_path / "sample_study" / "condition"
    package.mkdir(parents=True)
    (tmp_path / "sample_study" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "experiment.py").write_text(
        "def run(context):\n"
        "    (context.results_dir / 'summary.json').write_text('{}')\n"
        "    return context.seed\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    experiment = load_experiment("sample_study.condition.experiment")
    context = make_run_context(
        experiment,
        run_id="test-run",
        hardware_profile="cpu",
    )
    result = execute_experiment(experiment, context, command=["test"])

    assert result == 42
    assert context.results_dir == package / "results" / "test-run"
    assert context.artifacts_dir == package / "artifacts" / "test-run"
    manifest = json.loads(
        (context.results_dir / "run_manifest.json").read_text()
    )
    assert manifest["status"] == "completed"


def test_packages_import_outside_repository_root(tmp_path):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import analysis, envs, experiments, harness, learners, losses",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
