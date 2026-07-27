"""Tests for domain-agnostic analysis operations."""

from __future__ import annotations

import numpy as np

from analysis.checkpoints import discover_checkpoints
from analysis.contexts import (
    discrete_context_count,
    iter_discrete_context_batches,
)
from analysis.probes import (
    cluster_bootstrap_statistics,
    conditional_mse_metrics,
    conditional_residual_r2,
    fit_affine_probe,
    global_mse_metrics,
    held_out_permutation_null,
    mean_squared_error,
    percentile_interval,
    predictive_belief_sequence,
    predictive_belief_update,
    probe_predict,
    r2_score,
    split_group_indices,
    split_indices,
    target_variance,
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


def test_affine_probe_uses_stable_solver_and_unpenalized_intercept():
    offset = 1e10
    coordinate = np.linspace(-1.0, 1.0, 200)
    features = np.column_stack(
        [
            offset + coordinate,
            offset + coordinate + 1e-5 * coordinate**2,
        ]
    )
    targets = np.column_stack(
        [
            3.0 * coordinate + 7.0,
            -2.0 * coordinate - 4.0,
        ]
    )

    weight, bias = fit_affine_probe(features, targets, ridge=0.0)
    predicted = probe_predict(weight, bias, features)
    assert mean_squared_error(predicted, targets) < 1e-10

    constant_targets = np.full((len(features), 1), 12.5)
    _, regularized_bias = fit_affine_probe(
        np.zeros_like(features),
        constant_targets,
        ridge=1e6,
    )
    np.testing.assert_allclose(regularized_bias, [12.5])


def test_group_split_keeps_dependent_samples_together():
    groups = np.repeat(np.arange(10), 3)
    first_train, first_test = split_group_indices(groups, seed=7)
    second_train, second_test = split_group_indices(groups, seed=7)

    np.testing.assert_array_equal(first_train, second_train)
    np.testing.assert_array_equal(first_test, second_test)
    assert set(groups[first_train]).isdisjoint(groups[first_test])
    assert len(np.unique(groups[first_test])) == 2


def test_mse_metrics_preserve_global_and_conditional_r2_interpretation():
    target = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [10.0, 0.0],
            [12.0, 0.0],
        ]
    )
    predicted = target + np.array([0.5, 0.0])
    groups = np.array([0, 0, 1, 1])

    global_metrics = global_mse_metrics(predicted, target)
    fine_metrics = conditional_mse_metrics(predicted, target, groups)

    assert mean_squared_error(predicted, target) == 0.125
    assert target_variance(target) == 13.0
    assert global_metrics == {
        "mse": 0.125,
        "target_variance": 13.0,
        "global_mse_ratio": 0.125 / 13.0,
    }
    assert fine_metrics == {
        "fine_evaluation_mse": 0.125,
        "branch_baseline_mse": 0.5,
        "fine_mse_ratio": 0.25,
        "fine_mse_improvement": 0.375,
        "n_evaluated": 4,
    }
    np.testing.assert_allclose(
        r2_score(predicted, target),
        1.0 - global_metrics["global_mse_ratio"],
    )
    np.testing.assert_allclose(
        conditional_residual_r2(
            predicted,
            target,
            groups,
        ),
        1.0 - fine_metrics["fine_mse_ratio"],
    )


def test_conditional_mse_metrics_report_empty_filtered_evaluation():
    metrics = conditional_mse_metrics(
        np.array([[0.0], [1.0]]),
        np.array([[0.0], [1.0]]),
        np.array([0, 1]),
        min_group_size=2,
    )

    assert metrics["n_evaluated"] == 0
    for key in (
        "fine_evaluation_mse",
        "branch_baseline_mse",
        "fine_mse_ratio",
        "fine_mse_improvement",
    ):
        assert np.isnan(metrics[key])


def test_uniform_discrete_context_batches_cover_lexicographic_product():
    batches = list(
        iter_discrete_context_batches(
            2,
            3,
            batch_size=3,
            dtype=np.int32,
        )
    )
    contexts = np.concatenate(batches)

    assert discrete_context_count(2, 3) == 8
    assert [len(batch) for batch in batches] == [3, 3, 2]
    assert contexts.dtype == np.int32
    np.testing.assert_array_equal(
        contexts,
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
    )


def test_cluster_bootstrap_is_deterministic_and_returns_percentile_interval():
    values = np.array([0.0, 0.0, 1.0, 1.0, 3.0, 3.0])
    clusters = np.repeat(np.arange(3), 2)

    first = cluster_bootstrap_statistics(
        clusters,
        lambda indices: float(values[indices].mean()),
        n_resamples=100,
        seed=11,
    )
    second = cluster_bootstrap_statistics(
        clusters,
        lambda indices: float(values[indices].mean()),
        n_resamples=100,
        seed=11,
    )
    low, high = percentile_interval(first, confidence=0.90)

    np.testing.assert_array_equal(first, second)
    assert low < values.mean() < high


def test_held_out_permutation_null_refits_against_shuffled_labels():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(300, 4))
    targets = features @ np.array(
        [
            [1.0, -0.5],
            [0.3, 0.7],
            [-0.2, 0.4],
            [0.8, -0.1],
        ]
    )
    train, test = split_indices(len(features), seed=8)

    def fit_predict(permuted_targets):
        weight, bias = fit_affine_probe(
            features[train],
            permuted_targets,
            ridge=0.0,
        )
        return probe_predict(weight, bias, features[test])

    real_prediction = fit_predict(targets[train])
    real_mse = mean_squared_error(real_prediction, targets[test])
    first = held_out_permutation_null(
        targets[train],
        fit_predict,
        targets[test],
        n_permutations=20,
        seed=9,
    )
    second = held_out_permutation_null(
        targets[train],
        fit_predict,
        targets[test],
        n_permutations=20,
        seed=9,
    )

    np.testing.assert_array_equal(first, second)
    assert real_mse < 1e-20
    assert np.all(first > 0.1)


def test_transducer_beliefs_are_action_conditioned_and_include_initial_state():
    initial = np.array([0.5, 0.5])
    action_0_outcome_0 = np.array(
        [
            [0.6, 0.1],
            [0.0, 0.2],
        ]
    )
    action_1_outcome_0 = np.array(
        [
            [0.1, 0.0],
            [0.2, 0.6],
        ]
    )

    after_action_0 = predictive_belief_update(
        initial,
        action_0_outcome_0,
    )
    after_action_1 = predictive_belief_update(
        initial,
        action_1_outcome_0,
    )
    np.testing.assert_allclose(after_action_0, [2.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_allclose(after_action_1, [1.0 / 3.0, 2.0 / 3.0])

    sequence = predictive_belief_sequence(
        initial,
        [action_0_outcome_0, action_1_outcome_0],
    )
    assert sequence.shape == (3, 2)
    np.testing.assert_allclose(sequence[0], initial)
    np.testing.assert_allclose(
        sequence[2],
        predictive_belief_update(
            after_action_0,
            action_1_outcome_0,
        ),
    )


def test_transducer_belief_update_rejects_invalid_or_impossible_operators():
    initial = np.array([0.5, 0.5])

    with np.testing.assert_raises_regex(ValueError, "zero probability"):
        predictive_belief_update(initial, np.zeros((2, 2)))
    with np.testing.assert_raises_regex(ValueError, "substochastic"):
        predictive_belief_update(
            initial,
            np.array([[0.8, 0.3], [0.1, 0.2]]),
        )


def test_action_free_hmm_operator_is_transducer_special_case():
    initial = np.array([0.4, 0.6])
    likelihood = np.array([[0.8, 0.2], [0.3, 0.7]])
    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    operator = np.diag(likelihood[:, 1]) @ transition

    measured = initial * likelihood[:, 1]
    measured /= measured.sum()
    expected = measured @ transition

    np.testing.assert_allclose(
        predictive_belief_update(initial, operator),
        expected,
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


def test_rollout_seed_zero_reproduces_policy_actions_and_episode_resets():
    def collect(*, extra_policy_draws: int = 0):
        reset_seeds = []

        class StochasticEnv:
            def __init__(self):
                self.action_space = TinyActionSpace()

            def reset(self, *, seed):
                reset_seeds.append(seed)
                self.rng = np.random.default_rng(seed)
                self.step_index = 0
                value = int(self.rng.integers(1000))
                return np.array([value]), {"target": np.array([value])}

            def step(self, action):
                self.step_index += 1
                value = int(self.rng.integers(1000))
                return (
                    np.array([value]),
                    float(action),
                    False,
                    self.step_index == 2,
                    {"target": np.array([value])},
                )

            def close(self):
                pass

        def step_adapter(observation, state, randomness):
            randomness.numpy.random(extra_policy_draws)
            action = int(randomness.numpy.integers(2))
            return action, state, observation

        data = collect_rollout_data(
            StochasticEnv,
            step_adapter,
            lambda observation, info: info["target"],
            n_steps=7,
            seed=0,
        )
        return data, reset_seeds

    first, first_resets = collect()
    second, second_resets = collect()
    np.testing.assert_array_equal(first.representations, second.representations)
    np.testing.assert_array_equal(first.targets, second.targets)
    np.testing.assert_array_equal(first.actions, second.actions)
    np.testing.assert_array_equal(first.rewards, second.rewards)
    assert first_resets == second_resets

    _, resets_after_extra_policy_draws = collect(extra_policy_draws=5)
    assert first_resets == resets_after_extra_policy_draws


def test_batched_rollouts_preserve_alignment_and_reset_selected_state():
    reset_calls = []

    def initial_state(n_envs):
        return np.zeros(n_envs, dtype=np.int64)

    def reset_state(state, indices):
        reset_calls.append(tuple(indices))
        updated = state.copy()
        updated[indices] = 0
        return updated

    def step_adapter(observations, state, randomness, action_spaces):
        del randomness, action_spaces
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
