"""Short end-to-end checks across real RLlib and supervised execution paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import ray
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from harness.context import RunContext
from harness.hardware import PROFILES
from harness.runners import run_algorithm, run_tune
from learners import (
    ConfigurableOptimizerMixin,
    IQNPPOTorchLearner,
    PPGConfig,
    PPGTorchLearner,
    QRPPOTorchLearner,
)
from learners.ppg import PPG_STATE_KEY
from learners.models import (
    IQNValueMixin,
    MLPModel,
    PPGAuxiliaryValueHead,
    QRValueMixin,
    TransformerModel,
)
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner


class AdamWLearner(ConfigurableOptimizerMixin, PPOTorchLearner):
    """Inline Learner leaf for optimizer integration coverage."""


class MuonLearner(ConfigurableOptimizerMixin, PPOTorchLearner):
    """Inline Learner leaf for Muon (+ AdamW aux) integration coverage."""


class IQNTinyModel(IQNValueMixin, MLPModel):
    """Inline actor-critic composition for IQN integration coverage."""


class QRTinyModel(QRValueMixin, MLPModel):
    """Inline actor-critic composition for fixed-quantile coverage."""


class QRTransformerTinyModel(QRValueMixin, TransformerModel):
    """Inline recurrent actor-critic composition for fixed-quantile coverage."""


class PPGTinyModel(PPGAuxiliaryValueHead, MLPModel):
    """Inline actor-critic composition for PPG integration coverage."""


class TrackingPPGLearner(PPGTorchLearner):
    """Records connector inputs and target signatures for replay assertions."""

    batch_records = []

    def _make_batch_if_necessary(self, training_data):
        source_is_episodes = training_data.episodes is not None
        batch = super()._make_batch_if_necessary(training_data)
        targets = next(iter(batch.policy_batches.values()))[
            Postprocessing.VALUE_TARGETS
        ]
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().contiguous().numpy()
        else:
            targets = np.ascontiguousarray(targets)
        target_multiset = np.sort(targets.reshape(-1))
        self.batch_records.append(
            {
                "phase": self._ppg_phase,
                "source_is_episodes": source_is_episodes,
                "env_steps": batch.env_steps(),
                "target_signature": hashlib.sha256(
                    target_multiset.tobytes()
                ).hexdigest(),
            }
        )
        return batch


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


def tiny_ppg_config(
    *,
    policy_iterations_per_aux=2,
    aux_epochs=1,
    learner_class=PPGTorchLearner,
) -> PPGConfig:
    return (
        PPGConfig()
        .environment(TinyEnv)
        .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
        .learners(
            num_learners=0,
            num_gpus_per_learner=0,
            learner_class=learner_class,
        )
        .training(
            lr=3e-4,
            train_batch_size_per_learner=32,
            minibatch_size=16,
            num_epochs=1,
            policy_iterations_per_aux=policy_iterations_per_aux,
            aux_epochs=aux_epochs,
            aux_minibatch_size=16,
            aux_lr=3e-4,
            beta_clone=1.0,
            aux_value_loss_coeff=0.1,
            aux_true_value_loss_coeff=0.1,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=PPGTinyModel,
                model_config={"hidden_dims": (16, 16)},
            )
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


def test_tiny_ppo_with_recurrent_qr_value_critic(tmp_path):
    context = make_context(tmp_path, "qr-transformer")
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
                module_class=QRTransformerTinyModel,
                model_config={
                    "d_model": 16,
                    "n_layers": 1,
                    "n_heads": 2,
                    "context_len": 4,
                    "max_seq_len": 4,
                    "qr_value": {"num_quantiles": 8},
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


def test_tiny_phasic_policy_gradient_runs_both_phases(tmp_path):
    context = make_context(tmp_path, "ppg")
    TrackingPPGLearner.batch_records = []
    config = tiny_ppg_config(
        aux_epochs=2,
        learner_class=TrackingPPGLearner,
    )

    result = run_algorithm(
        config,
        context,
        should_stop=lambda values: values["training_iteration"] >= 2,
    )

    assert result["training_iteration"] == 2
    assert result["ppg/aux_phase_triggered"] == 1
    assert result["ppg/policy_iterations_since_aux"] == 0
    learner_metrics = result["learners"]["default_policy"]
    assert "ppg/aux_policy_kl" in learner_metrics
    assert "ppg/aux_value_loss" in learner_metrics
    assert "ppg/aux_true_value_loss" in learner_metrics

    policy_records = [
        record
        for record in TrackingPPGLearner.batch_records
        if record["phase"] == "policy"
    ]
    auxiliary_records = [
        record
        for record in TrackingPPGLearner.batch_records
        if record["phase"] == "auxiliary"
    ]
    assert len(policy_records) == 2
    assert len(auxiliary_records) == 4
    assert [
        (record["env_steps"], record["target_signature"])
        for record in auxiliary_records
    ] == [
        (record["env_steps"], record["target_signature"])
        for record in policy_records * 2
    ]
    assert all(record["source_is_episodes"] for record in policy_records)
    assert not any(record["source_is_episodes"] for record in auxiliary_records)


def test_ppg_checkpoint_roundtrip_preserves_partial_phase_state(tmp_path):
    context = make_context(tmp_path, "ppg-checkpoint")
    config = tiny_ppg_config(policy_iterations_per_aux=2)

    run_algorithm(
        config,
        context,
        should_stop=lambda values: values["training_iteration"] >= 1,
        checkpoint_at_end=True,
    )
    checkpoint = next(context.artifacts_dir.glob("checkpoints/*"))
    restored = Algorithm.from_checkpoint(str(checkpoint))
    try:
        algorithm_state = restored.get_state()
        assert algorithm_state[PPG_STATE_KEY] == {"policy_iterations": 1}
        assert PPG_STATE_KEY not in restored.get_state(
            components="learner_group"
        )
        assert PPG_STATE_KEY not in restored.get_state(
            not_components=PPG_STATE_KEY
        )
        assert restored.get_state(components=PPG_STATE_KEY)[PPG_STATE_KEY] == {
            "policy_iterations": 1
        }

        learner_state = restored.learner_group.get_state()["learner"]
        replay_batches = learner_state[PPG_STATE_KEY]["replay_batches"]
        assert len(replay_batches) == 1
        assert all(
            isinstance(value, torch.Tensor) and value.device.type == "cpu"
            for batch in replay_batches
            for module_batch in batch.policy_batches.values()
            for value in module_batch.values()
        )

        result = restored.train()
        assert result["ppg/aux_phase_triggered"] == 1
        assert result["ppg/policy_iterations_since_aux"] == 0
        assert (
            restored.learner_group.get_state()["learner"][PPG_STATE_KEY][
                "replay_batches"
            ]
            == []
        )
    finally:
        restored.stop()
        ray.shutdown()


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
    progress = [
        json.loads(line)
        for line in (context.results_dir / "progress.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(progress) == 1
    assert progress[0]["training_iteration"] == 1

