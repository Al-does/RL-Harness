"""Short end-to-end checks across real RLlib and supervised execution paths."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from harness.context import RunContext
from harness.hardware import PROFILES
from harness.runners import run_algorithm, run_tune
from learners import (
    ConfigurableOptimizerMixin,
    IQNPPOTorchLearner,
    QRPPOTorchLearner,
)
from learners.models import IQNValueMixin, MLPModel, QRValueMixin
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner


class AdamWLearner(ConfigurableOptimizerMixin, PPOTorchLearner):
    """Inline Learner leaf for optimizer integration coverage."""


class MuonLearner(ConfigurableOptimizerMixin, PPOTorchLearner):
    """Inline Learner leaf for Muon (+ AdamW aux) integration coverage."""


class IQNTinyModel(IQNValueMixin, MLPModel):
    """Inline actor-critic composition for IQN integration coverage."""


class QRTinyModel(QRValueMixin, MLPModel):
    """Inline actor-critic composition for fixed-quantile coverage."""


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


def test_tiny_ppo_with_configurable_adamw(tmp_path):
    context = make_context(tmp_path, "adamw")
    config = (
        PPOConfig()
        .environment(TinyEnv)
        .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
        .learners(
            num_learners=0,
            num_gpus_per_learner=0,
            learner_class=AdamWLearner,
            learner_config_dict={
                "optimizer/type": "adamw",
                "optimizer/kwargs": {"weight_decay": 0.01},
            },
        )
        .training(
            lr=3e-4,
            train_batch_size_per_learner=32,
            minibatch_size=16,
            num_epochs=1,
        )
        .debugging(seed=42)
    )

    result = run_algorithm(
        config,
        context,
        should_stop=lambda values: values["training_iteration"] >= 1,
    )

    assert result["training_iteration"] == 1


def test_tiny_ppo_with_configurable_muon(tmp_path):
    context = make_context(tmp_path, "muon")
    config = (
        PPOConfig()
        .environment(TinyEnv)
        .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
        .learners(
            num_learners=0,
            num_gpus_per_learner=0,
            learner_class=MuonLearner,
            learner_config_dict={
                "optimizer/type": "muon",
                "optimizer/kwargs": {"momentum": 0.95},
            },
        )
        .training(
            lr=3e-4,
            train_batch_size_per_learner=32,
            minibatch_size=16,
            num_epochs=1,
        )
        .debugging(seed=42)
    )

    result = run_algorithm(
        config,
        context,
        should_stop=lambda values: values["training_iteration"] >= 1,
    )

    assert result["training_iteration"] == 1


def test_tiny_ppo_with_iqn_value_critic(tmp_path):
    context = make_context(tmp_path, "iqn")
    config = (
        PPOConfig()
        .environment(TinyEnv)
        .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
        .learners(
            num_learners=0,
            num_gpus_per_learner=0,
            learner_class=IQNPPOTorchLearner,
            learner_config_dict={
                "iqn_value/loss_coefficient": 0.5,
                "iqn_value/huber_kappa": 1.0,
            },
        )
        .training(
            lr=3e-4,
            vf_loss_coeff=0.0,
            train_batch_size_per_learner=32,
            minibatch_size=16,
            num_epochs=1,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=IQNTinyModel,
                model_config={
                    "hidden_dims": (16, 16),
                    "iqn_value": {
                        "train_quantiles": 8,
                        "value_quantiles": 16,
                        "n_cosines": 16,
                    },
                },
            )
        )
        .debugging(seed=42)
    )

    result = run_algorithm(
        config,
        context,
        should_stop=lambda values: values["training_iteration"] >= 1,
    )

    assert result["training_iteration"] == 1
    learner_metrics = result["learners"]["default_policy"]
    assert "iqn_value/loss" in learner_metrics
    assert "iqn_value/mean_quantile_spread" in learner_metrics


def test_tiny_ppo_with_qr_value_critic(tmp_path):
    context = make_context(tmp_path, "qr")
    config = (
        PPOConfig()
        .environment(TinyEnv)
        .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
        .learners(
            num_learners=0,
            num_gpus_per_learner=0,
            learner_class=QRPPOTorchLearner,
            learner_config_dict={
                "qr_value/loss_coefficient": 0.5,
                "qr_value/huber_kappa": 1.0,
            },
        )
        .training(
            lr=3e-4,
            vf_loss_coeff=0.0,
            train_batch_size_per_learner=32,
            minibatch_size=16,
            num_epochs=1,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=QRTinyModel,
                model_config={
                    "hidden_dims": (16, 16),
                    "qr_value": {"num_quantiles": 16},
                },
            )
        )
        .debugging(seed=42)
    )

    result = run_algorithm(
        config,
        context,
        should_stop=lambda values: values["training_iteration"] >= 1,
    )

    assert result["training_iteration"] == 1
    learner_metrics = result["learners"]["default_policy"]
    assert "qr_value/loss" in learner_metrics
    assert "qr_value/mean_quantile_spread" in learner_metrics


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

