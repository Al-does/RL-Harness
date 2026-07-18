"""Short end-to-end checks across real RLlib and supervised execution paths."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig

from harness.context import RunContext
from harness.hardware import PROFILES
from harness.runners import run_algorithm, run_tune


class TinyEnv(gym.Env):
    """Inline deterministic task for generic runner integration tests."""

    observation_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(4,),
        dtype=np.float32,
    )
    action_space = gym.spaces.Discrete(2)

    def __init__(self, config=None):
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        observation = np.full(4, self._step / 4, dtype=np.float32)
        terminated = self._step >= 4
        return observation, float(action == self._step % 2), terminated, False, {}


def make_context(tmp_path: Path, name: str) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / name / "results",
        artifacts_dir=tmp_path / name / "artifacts",
        run_id=name,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


def tiny_ppo_config() -> PPOConfig:
    return (
        PPOConfig()
        .environment(TinyEnv)
        .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
        .learners(num_learners=0, num_gpus_per_learner=0)
        .training(
            train_batch_size_per_learner=32,
            minibatch_size=16,
            num_epochs=1,
        )
        .debugging(seed=42)
    )


def test_tiny_direct_rllib_ppo_run(tmp_path):
    context = make_context(tmp_path, "direct")

    result = run_algorithm(
        tiny_ppo_config(),
        context,
        should_stop=lambda values: values["training_iteration"] >= 1,
    )

    assert result["training_iteration"] == 1
    records = context.results_dir.joinpath("progress.jsonl").read_text().splitlines()
    assert len(records) == 1


def test_tiny_tune_managed_ppo_run(tmp_path):
    context = make_context(tmp_path, "tune")

    result_grid = run_tune(
        tiny_ppo_config(),
        context,
        stop={"training_iteration": 1},
        run_config_kwargs={"verbose": 0},
    )

    assert len(result_grid) == 1
    summary = json.loads(
        context.results_dir.joinpath("tune_summary.json").read_text()
    )
    assert summary["num_trials"] == 1
    assert summary["trials"][0]["status"] == "completed"
    assert summary["trials"][0]["resolved_seed"] == 42

