"""Tests for domain-agnostic analysis operations."""

from __future__ import annotations

import numpy as np

from analysis.checkpoints import discover_checkpoints
from analysis.probes import (
    conditional_residual_r2,
    fit_affine_probe,
    probe_predict,
    r2_score,
    split_indices,
)
from analysis.rollouts import (
    collect_batched_rollout_data,
    collect_rollout_data,
)
from harness.artifacts import RunArtifacts


def test_affine_probe_fit_split_and_metrics():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(100, 3))
    weight = np.array([[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]])
    bias = np.array([0.2, -0.7])
    targets = features @ weight + bias
    train, test = split_indices(len(features), seed=9)

    fitted_weight, fitted_bias = fit_affine_probe(
        features[train],
        targets[train],
    )
    predicted = probe_predict(
        fitted_weight,
        fitted_bias,
        features[test],
    )

    assert r2_score(predicted, targets[test]) > 0.999999
    groups = np.arange(len(test)) % 2
    assert (
        conditional_residual_r2(
            predicted,
            targets[test],
            groups,
        )
        > 0.999999
    )


class TinyActionSpace:
    def seed(self, seed):
        self.rng = np.random.default_rng(seed)


class TinyEnv:
    action_space = TinyActionSpace()

    def reset(self, *, seed):
        self.value = 0
        return np.array([0.0]), {"target": np.array([0.0])}

    def step(self, action):
        reward = float(action)
        self.value += 1
        done = self.value == 3
        return (
            np.array([float(self.value)]),
            reward,
            False,
            done,
            {"target": np.array([float(self.value)])},
        )

    def close(self):
        pass


def test_rollout_collection_uses_injected_representation_and_target_adapters():
    def step_adapter(observation, state, rng):
        return 1, state, observation * 2

    data = collect_rollout_data(
        TinyEnv,
        step_adapter,
        lambda observation, info: info["target"],
        n_steps=5,
        seed=42,
    )

    assert data.representations.shape == (5, 1)
    assert data.targets.shape == (5, 1)
    assert data.actions.shape == (5, 1)
    assert np.all(data.rewards == 1.0)


def test_batched_rollouts_preserve_alignment_and_reset_selected_state():
    reset_calls = []

    def initial_state(n_envs):
        return np.zeros(n_envs, dtype=np.int64)

    def reset_state(state, indices):
        reset_calls.append(tuple(indices))
        updated = state.copy()
        updated[indices] = 0
        return updated

    def step_adapter(observations, state, rng, action_spaces):
        del rng, action_spaces
        actions = np.ones(len(observations), dtype=np.int64)
        representations = np.concatenate(
            [observations, state[:, None]],
            axis=1,
        )
        return actions, state + 1, representations

    def target_adapter(observations, infos, episode_steps):
        del observations
        return {
            "target": np.stack([info["target"] for info in infos]),
            "episode_step": episode_steps,
        }

    data = collect_batched_rollout_data(
        TinyEnv,
        step_adapter,
        target_adapter,
        n_steps=5,
        seed=42,
        n_envs=2,
        initial_state=initial_state,
        reset_state=reset_state,
        warmup=1,
    )

    assert data.representations.shape == (5, 2)
    assert data.actions.shape == (5, 1)
    assert data.targets.keys() == {"target", "episode_step"}
    np.testing.assert_array_equal(
        data.targets["episode_step"],
        [1, 1, 2, 2, 1],
    )
    np.testing.assert_array_equal(
        data.representations[:, 1],
        data.targets["episode_step"],
    )
    assert reset_calls == [(np.int64(0), np.int64(1))]


def test_checkpoint_discovery_uses_complete_directory_markers(tmp_path):
    artifacts = RunArtifacts(
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
    )
    direct = artifacts.checkpoints_dir / "iteration_000001"
    direct.mkdir(parents=True)
    (direct / "rllib_checkpoint.json").write_text("{}")
    tune = artifacts.tune_dir / "trial" / "checkpoint_000002"
    tune.mkdir(parents=True)

    assert discover_checkpoints(artifacts) == [direct, tune]
