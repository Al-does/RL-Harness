from __future__ import annotations

import json

import numpy as np
import pytest

from analysis.probes import controls
from analysis.probes.controls import (
    fit_grouped_affine,
    gaussian_feature_null,
    matched_feature_null,
    paired_comparison,
    paired_target_comparison,
    score_prediction,
    suffix_keys,
)


def test_grouped_affine_exact_recovery_and_determinism():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(120, 4)) + 5.0
    expected_weight = rng.normal(size=(4, 3))
    expected_bias = np.array([3.0, -7.0, 0.1])
    y = x @ expected_weight + expected_bias
    groups = np.repeat(np.arange(12), 10)
    weight, bias, fit = fit_grouped_affine(x, y, groups)
    np.testing.assert_allclose(weight, expected_weight, atol=1e-12)
    np.testing.assert_allclose(bias, expected_bias, atol=1e-12)
    held_out = rng.normal(size=(50, 4))
    np.testing.assert_allclose(
        held_out @ weight + bias,
        held_out @ expected_weight + expected_bias,
        atol=1e-12,
    )
    repeated = fit_grouped_affine(x, y, groups)
    np.testing.assert_array_equal(weight, repeated[0])
    np.testing.assert_array_equal(bias, repeated[1])
    assert fit == repeated[2]
    assert fit["n_groups"] == 12
    assert fit["folds"] == 5
    assert fit["n_samples"] == len(x)
    assert fit["fit_source"] == "train"
    json.dumps(fit, allow_nan=False)


def test_grouped_cv_reuses_svd_and_uses_only_fold_training_data(monkeypatch):
    rng = np.random.default_rng(8)
    groups = np.repeat(np.array(["a", "b", "c", "d"]), [7, 11, 5, 13])
    x = rng.normal(size=(len(groups), 3))
    x[:, 1] = x[:, 0] + 1e-5 * x[:, 1]
    x[groups == "a"] += 12.0
    y = rng.normal(size=(len(groups), 2))
    y[groups == "c"] += 30.0
    rconds = (1e-12, 1e-8, 1e-4, 1e-2)
    designs = []
    original_svd = np.linalg.svd

    def track_svd(design, **kwargs):
        designs.append(design.copy())
        return original_svd(design, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", track_svd)
    weight, bias, fit = fit_grouped_affine(x, y, groups, folds=9, rconds=rconds)
    assert len(designs) == 5
    assert fit["requested_folds"] == 9
    assert fit["folds"] == fit["n_groups"] == 4
    assert fit["rconds"] == list(rconds)
    assert sorted(group for fold in fit["fold_validation_groups"] for group in fold) == ["a", "b", "c", "d"]
    errors = np.empty((4, len(rconds)))
    for fold, validation_groups in enumerate(fit["fold_validation_groups"]):
        validation = np.isin(groups, validation_groups)
        training = ~validation
        assert not set(groups[training]) & set(groups[validation])
        np.testing.assert_allclose(designs[fold], x[training] - x[training].mean(axis=0), atol=1e-13)
        assert fit["fold_validation_sizes"][fold] == int(validation.sum())
        for candidate, rcond in enumerate(rconds):
            w = np.linalg.lstsq(
                x[training] - x[training].mean(axis=0),
                y[training] - y[training].mean(axis=0),
                rcond=rcond,
            )[0]
            b = y[training].mean(axis=0) - x[training].mean(axis=0) @ w
            errors[fold, candidate] = np.square(x[validation] @ w + b - y[validation]).mean()
    np.testing.assert_allclose(fit["fold_mse"], errors, rtol=1e-8)
    expected_cv = np.average(errors, axis=0, weights=fit["fold_validation_sizes"])
    np.testing.assert_allclose(fit["cv_mse"], expected_cv, rtol=1e-8)
    assert fit["selected_rcond"] == rconds[int(np.argmin(expected_cv))]
    refit = np.linalg.lstsq(x - x.mean(axis=0), y - y.mean(axis=0), rcond=fit["selected_rcond"])[0]
    np.testing.assert_allclose(weight, refit, rtol=1e-8)
    np.testing.assert_allclose(bias, y.mean(axis=0) - x.mean(axis=0) @ refit, rtol=1e-8)
    np.testing.assert_allclose(designs[-1], x - x.mean(axis=0), atol=1e-13)


def test_grouped_affine_collinear_and_constant_features():
    coordinate = np.linspace(-2.0, 2.0, 80)
    x = np.column_stack([coordinate, 2.0 * coordinate, np.ones(80)])
    y = np.column_stack([4.0 * coordinate + 8.0, np.full(80, 0.1)])
    groups = np.arange(80) % 8
    weight, bias, fit = fit_grouped_affine(x, y, groups)
    np.testing.assert_allclose(x @ weight + bias, y, atol=1e-12)
    assert fit["rank"] == 1
    for constant_x in (np.zeros((80, 3)), np.full((80, 3), 0.1)):
        weight, bias, fit = fit_grouped_affine(constant_x, y, groups)
        np.testing.assert_array_equal(weight, 0.0)
        np.testing.assert_allclose(bias, y.mean(axis=0), atol=1e-12)
        assert fit["rank"] == 0
        json.dumps(fit, allow_nan=False)


def test_grouped_affine_cutoff_does_not_penalize_intercept():
    x = np.arange(24, dtype=float).reshape(12, 2)
    y = np.full((12, 1), 17.0)
    weight, bias, fit = fit_grouped_affine(x, y, np.arange(12) % 3, rconds=(1.0,))
    np.testing.assert_array_equal(weight, 0.0)
    np.testing.assert_array_equal(bias, [17.0])
    assert fit["rank"] == 0


@pytest.mark.parametrize(
    "x,y,groups,kwargs",
    [
        (np.ones((3, 2)), np.ones((3, 1)), [1, 1, 1], {}),
        (np.empty((0, 2)), np.empty((0, 1)), [], {}),
        (np.ones((3, 0)), np.ones((3, 1)), [0, 1, 2], {}),
        (np.ones(3), np.ones((3, 1)), [0, 1, 2], {}),
        (np.ones((3, 2)), np.ones((2, 1)), [0, 1, 2], {}),
        (np.full((3, 2), np.nan), np.ones((3, 1)), [0, 1, 2], {}),
        (np.ones((3, 2)), np.full((3, 1), np.inf), [0, 1, 2], {}),
        (np.ones((3, 2)), np.ones((3, 1)), [0, 1], {}),
        (np.ones((3, 2)), np.ones((3, 1)), [0, np.nan, 2], {}),
        (np.ones((3, 2)), np.ones((3, 1)), [[0], [1], [2]], {}),
        (np.ones((3, 2)), np.ones((3, 1)), [0, 1, 2], {"folds": 1}),
        (np.ones((3, 2)), np.ones((3, 1)), [0, 1, 2], {"folds": 2.5}),
        (np.ones((3, 2)), np.ones((3, 1)), [0, 1, 2], {"rconds": ()}),
        (np.ones((3, 2)), np.ones((3, 1)), [0, 1, 2], {"rconds": (-1.0,)}),
        (np.ones((3, 2)), np.ones((3, 1)), [0, 1, 2], {"rconds": (np.nan,)}),
    ],
)
def test_grouped_affine_rejects_invalid_inputs(x, y, groups, kwargs):
    with pytest.raises(ValueError):
        fit_grouped_affine(x, y, groups, **kwargs)


def test_score_prediction_global_variance_weighting():
    target = np.array([[-1.0, -10.0], [1.0, 10.0]])
    prediction = target.copy()
    prediction[:, 0] = 0.0
    result = score_prediction(prediction, target)
    assert result == {
        "mse": 0.5,
        "target_variance": 50.5,
        "normalized_mse": 1.0 / 101.0,
        "r_squared": 100.0 / 101.0,
        "n_evaluated": 2,
    }
    json.dumps(result, allow_nan=False)
    assert score_prediction(target[:, 0], target[:, 0])["r_squared"] == 1.0


@pytest.mark.parametrize("prediction_offset", [0.0, 1.0])
def test_score_zero_variance_is_json_safe_without_fake_ratios(prediction_offset):
    target = np.full((31, 3), 0.1)
    result = score_prediction(target + prediction_offset, target)
    assert result["mse"] == pytest.approx(prediction_offset**2)
    assert result["target_variance"] == 0.0
    assert result["normalized_mse"] is None
    assert result["r_squared"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "prediction,target",
    [
        ([], []),
        (np.ones((2, 1)), np.ones(2)),
        (np.ones((2, 0)), np.ones((2, 0))),
        ([np.inf], [0.0]),
        ([0.0], [np.nan]),
        ([1e308], [-1e308]),
        ([1.0j], [0.0]),
    ],
)
def test_score_rejects_invalid_or_unrepresentable_inputs(prediction, target):
    with pytest.raises(ValueError):
        score_prediction(prediction, target)


def test_paired_comparison_matches_whole_cluster_bootstrap(monkeypatch):
    target = np.arange(14, dtype=float).reshape(7, 2)
    groups = np.array(["a", "b", "a", "c", "a", "c", "a"])
    prediction = target + np.arange(7)[:, None] / 5.0
    baseline = target + np.array([3, 1, 3, 2, 3, 2, 3])[:, None]

    def no_fit(*args, **kwargs):
        raise AssertionError("evaluation must not refit")

    monkeypatch.setattr(controls, "fit_grouped_affine", no_fit)
    result = paired_comparison(prediction, baseline, target, groups, n_resamples=97)
    row_improvement = (np.square(baseline - target) - np.square(prediction - target)).mean(axis=1)
    expected_improvement = row_improvement.mean()
    expected_variance = np.square(target - target.mean(axis=0)).mean()
    assert result["mse_improvement"] == pytest.approx(expected_improvement)
    assert result["delta_r_squared"] == pytest.approx(expected_improvement / expected_variance)
    assert result["residual_fraction_recovered"] == pytest.approx(
        1.0 - np.square(prediction - target).mean() / np.square(baseline - target).mean()
    )
    rng = np.random.default_rng(42)
    members = [np.flatnonzero(groups == group) for group in ["a", "b", "c"]]
    improvements, deltas = [], []
    for _ in range(97):
        indices = np.concatenate([members[index] for index in rng.integers(0, 3, size=3)])
        improvement = row_improvement[indices].mean()
        sample = target[indices]
        variance = np.square(sample - sample.mean(axis=0)).mean()
        improvements.append(improvement)
        deltas.append(None if variance == 0 else improvement / variance)
    np.testing.assert_allclose(result["mse_improvement_ci"], np.quantile(improvements, [0.025, 0.975]))
    if None not in deltas:
        np.testing.assert_allclose(result["delta_r_squared_ci"], np.quantile(deltas, [0.025, 0.975]))
    else:
        assert result["delta_r_squared_ci"] is None
        assert result["delta_r_squared_ci_reason"] == "zero_target_variance_in_bootstrap_samples"
    assert result["bootstrap_refit"] is False
    assert result["n_groups"] == 3
    assert result["n_resamples"] == 97
    assert result == paired_comparison(prediction, baseline, target, groups, n_resamples=97)
    json.dumps(result, allow_nan=False)


def test_paired_comparison_has_delta_r_squared_interval_when_variance_is_positive():
    target = np.tile(np.arange(5, dtype=float), 4)
    groups = np.repeat(np.arange(4), 5)
    result = paired_comparison(target + 0.5, target + 1.0, target, groups)
    np.testing.assert_allclose(result["mse_improvement_ci"], [0.75, 0.75])
    np.testing.assert_allclose(result["delta_r_squared_ci"], [0.375, 0.375])
    assert result["delta_r_squared_valid_resamples"] == 200
    assert result["ci_reason"] is None


def test_paired_comparison_zero_variance_and_perfect_baseline():
    target = np.full((12, 2), 0.1)
    result = paired_comparison(target + 1, target, target, np.repeat(np.arange(3), 4))
    assert result["mse_improvement"] == pytest.approx(-1.0)
    np.testing.assert_allclose(result["mse_improvement_ci"], [-1.0, -1.0])
    assert result["delta_r_squared"] is None
    assert result["residual_fraction_recovered"] is None
    assert result["delta_r_squared_ci"] is None
    assert result["delta_r_squared_ci_reason"] == "zero_target_variance"
    assert result["delta_r_squared_valid_resamples"] == 0
    json.dumps(result, allow_nan=False)


def test_paired_comparison_single_episode_has_no_ci():
    target = np.arange(10, dtype=float)[:, None]
    result = paired_comparison(target, target + 2.0, target, np.zeros(10))
    assert result["mse_improvement"] == 4.0
    assert result["residual_fraction_recovered"] == 1.0
    assert result["mse_improvement_ci"] is None
    assert result["delta_r_squared_ci"] is None
    assert result["ci_reason"] == "at_least_two_groups_required"
    assert result["n_resamples"] == 0
    with pytest.raises(ValueError):
        paired_comparison(target, target, target, np.zeros(9))
    with pytest.raises(ValueError):
        paired_comparison(target, target, target, np.zeros(10), n_resamples=0)


def test_paired_target_comparison_point_delta_and_independent_scaling():
    target = np.array([-1.0, 1.0, -1.0, 1.0])
    prediction = target + 0.5
    alternative_target = target[:, None] * np.array([2.0, 4.0])
    alternative_prediction = alternative_target + [2.0, 4.0]
    groups = np.repeat([0, 1], 2)
    result = paired_target_comparison(
        prediction, target, alternative_prediction, alternative_target, groups
    )
    assert result["r_squared_difference"] == 0.75
    assert result["r_squared_difference"] == (
        result["true"]["r_squared"] - result["alternative"]["r_squared"]
    )
    assert result["r_squared_difference_ci"] == [0.75, 0.75]
    assert result["true"] == score_prediction(prediction, target)
    assert result["alternative"] == score_prediction(alternative_prediction, alternative_target)
    assert result["true"]["mse"] == 0.25
    assert result["alternative"]["mse"] == 10.0
    assert result["alternative"]["target_variance"] == 10.0
    assert result["valid_resamples"] == result["n_resamples"] == 200
    assert result["ci_reason"] is None
    assert result["estimate_reason"] is None
    assert "residual_fraction_recovered" not in result
    scaled = paired_target_comparison(
        -7.0 * prediction + 9.0,
        -7.0 * target + 9.0,
        3.0 * alternative_prediction + [-4.0, 12.0],
        3.0 * alternative_target + [-4.0, 12.0],
        groups,
    )
    assert scaled["r_squared_difference"] == pytest.approx(result["r_squared_difference"])
    np.testing.assert_allclose(scaled["r_squared_difference_ci"], result["r_squared_difference_ci"])
    json.dumps(result, allow_nan=False)


def test_paired_target_comparison_recomputes_variances_in_paired_episode_bootstrap(monkeypatch):
    groups = np.repeat(["b", "a", "c"], [3, 6, 9])
    coordinate = np.arange(len(groups), dtype=float)
    target = coordinate[:, None] / 5.0
    alternative_target = np.column_stack([np.sin(coordinate), (coordinate - 5.0)**2])
    prediction = target + np.where(groups == "b", 3.0, 0.2)[:, None]
    alternative_prediction = alternative_target + np.where(groups == "c", 8.0, 1.0)[:, None]
    before = [values.copy() for values in (prediction, target, alternative_prediction, alternative_target)]

    def no_refit_or_dummy_comparison(*args, **kwargs):
        raise AssertionError("comparison must use fixed predictions and their actual targets")

    monkeypatch.setattr(controls, "fit_grouped_affine", no_refit_or_dummy_comparison)
    monkeypatch.setattr(controls, "paired_comparison", no_refit_or_dummy_comparison)
    result = paired_target_comparison(
        prediction, target, alternative_prediction, alternative_target, groups, n_resamples=301
    )
    true_mse = np.square(prediction - target).mean(axis=1)
    alternative_mse = np.square(alternative_prediction - alternative_target).mean(axis=1)
    true_variance = np.square(target - target.mean(axis=0)).mean()
    alternative_variance = np.square(alternative_target - alternative_target.mean(axis=0)).mean()
    assert result["r_squared_difference"] == pytest.approx(
        alternative_mse.mean() / alternative_variance - true_mse.mean() / true_variance
    )

    def expected_statistic(indices):
        true_sample = target[indices]
        alternative_sample = alternative_target[indices]
        sampled_true_variance = np.square(true_sample - true_sample.mean(axis=0)).mean()
        sampled_alternative_variance = np.square(alternative_sample - alternative_sample.mean(axis=0)).mean()
        return (
            alternative_mse[indices].mean() / sampled_alternative_variance
            - true_mse[indices].mean() / sampled_true_variance
        )

    rng = np.random.default_rng(42)
    row_rng = np.random.default_rng(42)
    members = [np.flatnonzero(groups == group) for group in ["b", "a", "c"]]
    episode_estimates, frozen_variance_estimates, row_estimates = [], [], []
    for _ in range(301):
        indices = np.concatenate([members[group] for group in rng.integers(0, 3, size=3)])
        episode_estimates.append(expected_statistic(indices))
        frozen_variance_estimates.append(
            alternative_mse[indices].mean() / alternative_variance
            - true_mse[indices].mean() / true_variance
        )
        row_estimates.append(expected_statistic(row_rng.integers(0, len(target), size=len(target))))
    expected_ci = np.quantile(episode_estimates, [0.025, 0.975])
    np.testing.assert_allclose(result["r_squared_difference_ci"], expected_ci)
    assert not np.allclose(expected_ci, np.quantile(frozen_variance_estimates, [0.025, 0.975]))
    assert not np.allclose(expected_ci, np.quantile(row_estimates, [0.025, 0.975]))
    assert result["n_groups"] == 3
    assert result["valid_resamples"] == result["n_resamples"] == 301
    assert result["bootstrap_refit"] is False
    assert result["bootstrap_unit"] == "group"
    assert result == paired_target_comparison(
        prediction, target, alternative_prediction, alternative_target, groups, n_resamples=301
    )
    first_draw = paired_target_comparison(
        prediction, target, alternative_prediction, alternative_target, groups, n_resamples=1
    )
    other_seed = paired_target_comparison(
        prediction, target, alternative_prediction, alternative_target, groups, seed=7, n_resamples=1
    )
    assert other_seed["r_squared_difference"] == result["r_squared_difference"]
    assert other_seed["r_squared_difference_ci"] != first_draw["r_squared_difference_ci"]
    scaled = paired_target_comparison(
        prediction * 4.0 - 2.0,
        target * 4.0 - 2.0,
        alternative_prediction * -3.0 + [1.0, 2.0],
        alternative_target * -3.0 + [1.0, 2.0],
        groups,
        n_resamples=301,
    )
    np.testing.assert_allclose(scaled["r_squared_difference_ci"], expected_ci)
    for values, original in zip((prediction, target, alternative_prediction, alternative_target), before):
        np.testing.assert_array_equal(values, original)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("constant_sides", [("true",), ("alternative",), ("true", "alternative")])
def test_paired_target_comparison_constant_targets_have_explicit_reasons(constant_sides):
    target = np.full((6, 1), 0.1) if "true" in constant_sides else np.arange(6)[:, None]
    alternative_target = np.full((6, 2), 0.1) if "alternative" in constant_sides else np.arange(12).reshape(6, 2)
    result = paired_target_comparison(
        target + 1.0, target, alternative_target + 2.0, alternative_target, np.repeat([0, 1, 2], 2)
    )
    assert result["r_squared_difference"] is None
    assert result["r_squared_difference_ci"] is None
    assert result["estimate_reason"] == result["ci_reason"] == "zero_target_variance"
    assert result["zero_variance_targets"] == list(constant_sides)
    assert result["valid_resamples"] == result["n_resamples"] == 0
    assert result["requested_resamples"] == 200
    assert result["true"]["mse"] == pytest.approx(1.0)
    assert result["alternative"]["mse"] == pytest.approx(4.0)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("degenerate_side", ["true", "alternative"])
def test_paired_target_comparison_does_not_drop_zero_variance_bootstrap_samples(degenerate_side):
    groups = np.repeat([0, 1], 3)
    constant_within_episode = np.repeat([0.1, 2.0], 3)
    varying_within_episode = np.tile(np.arange(3, dtype=float), 2)
    target, alternative_target = (
        (constant_within_episode, varying_within_episode)
        if degenerate_side == "true" else (varying_within_episode, constant_within_episode)
    )
    result = paired_target_comparison(
        target + 1.0, target, alternative_target + 2.0, alternative_target, groups, n_resamples=97
    )
    rng = np.random.default_rng(42)
    expected_valid = sum(np.unique(rng.integers(0, 2, size=2)).size == 2 for _ in range(97))
    assert result["r_squared_difference"] is not None
    assert result["r_squared_difference_ci"] is None
    assert result["estimate_reason"] is None
    assert result["ci_reason"] == "zero_target_variance_in_bootstrap_samples"
    assert 0 < result["valid_resamples"] == expected_valid < result["n_resamples"] == 97
    json.dumps(result, allow_nan=False)


def test_paired_target_comparison_single_group_keeps_estimate_without_ci():
    target = np.arange(6, dtype=float)
    alternative_target = np.arange(12, dtype=float).reshape(6, 2)
    result = paired_target_comparison(
        target, target, alternative_target + 1.0, alternative_target, np.zeros(6)
    )
    assert result["r_squared_difference"] == pytest.approx(1.0 / np.var(alternative_target, axis=0).mean())
    assert result["r_squared_difference_ci"] is None
    assert result["ci_reason"] == "at_least_two_groups_required"
    assert result["estimate_reason"] is None
    assert result["n_groups"] == 1
    assert result["n_resamples"] == result["valid_resamples"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"prediction": np.ones(5), "target": np.ones(5)},
        {"alternative_prediction": np.ones((5, 2)), "alternative_target": np.ones((5, 2))},
        {"prediction": np.ones((6, 1))},
        {"alternative_prediction": np.ones((6, 3))},
        {"groups": np.zeros(5)},
        {"groups": np.full(6, np.nan)},
        {"prediction": np.full(6, np.inf)},
        {"target": np.full(6, np.nan)},
        {"alternative_prediction": np.full((6, 2), np.nan)},
        {"alternative_target": np.full((6, 2), np.inf)},
        {"n_resamples": 0},
        {"n_resamples": 1.5},
    ],
)
def test_paired_target_comparison_validates_alignment_and_finiteness(overrides):
    arguments = {
        "prediction": np.arange(6, dtype=float),
        "target": np.arange(6, dtype=float),
        "alternative_prediction": np.arange(12, dtype=float).reshape(6, 2),
        "alternative_target": np.arange(12, dtype=float).reshape(6, 2),
        "groups": np.repeat([0, 1], 3),
    }
    arguments.update(overrides)
    with pytest.raises(ValueError):
        paired_target_comparison(**arguments)


def test_matched_null_exact_rows_self_exclusion_and_fallback_counts():
    train = np.column_stack([np.arange(5), 10.0 + np.arange(5)**2])
    test = np.full((4, 2), -100.0)
    train_keys = np.array([[0, 0], [0, 0], [1, 0], [1, 0], [2, 0]])
    test_keys = np.array([[0, 0], [1, 0], [2, 0], [9, 0]])
    train_before, test_before = train.copy(), test.copy()
    train_null, test_null, metadata = matched_feature_null(train, test, train_keys, test_keys)
    np.testing.assert_array_equal(train_null[:4], train[[1, 0, 3, 2]])
    assert train_null[4, 0] != train[4, 0]
    assert test_null[0, 0] in [0, 1]
    assert test_null[1, 0] in [2, 3]
    for row in np.concatenate([train_null, test_null]):
        assert np.any(np.all(train == row, axis=1))
    assert metadata["train"]["matched_rows"] == 4
    assert metadata["train"]["fallback_rows"] == 1
    assert metadata["train"]["singleton_fallback_rows"] == 1
    assert metadata["train"]["self_excluded_rows"] == 5
    assert metadata["train"]["self_donor_rows"] == 0
    assert metadata["test"]["matched_rows"] == 2
    assert metadata["test"]["exact_key_rows"] == 3
    assert metadata["test"]["fallback_rows"] == 2
    assert metadata["test"]["singleton_fallback_rows"] == 1
    assert metadata["test"]["unseen_fallback_rows"] == 1
    assert metadata["test"]["exact_key_coverage"] == 0.75
    assert metadata["test"]["matched_fraction"] == 0.5
    np.testing.assert_array_equal(train, train_before)
    np.testing.assert_array_equal(test, test_before)
    json.dumps(metadata, allow_nan=False)


def test_matched_null_never_uses_test_features_and_draws_independently():
    train = np.arange(800, dtype=float).reshape(400, 2)
    keys = np.zeros(400, dtype=int)
    first = matched_feature_null(train, train, keys, keys)
    second = matched_feature_null(train, train + 1e6, keys, keys)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert np.all(first[0][:, 0] != train[:, 0])
    assert np.unique(first[0][:, 0]).size < len(train)
    assert abs(np.corrcoef(first[0][:, 0], first[1][:, 0])[0, 1]) < 0.15
    short = matched_feature_null(train, train[:20], keys, keys[:20])
    np.testing.assert_array_equal(first[0], short[0])
    np.testing.assert_array_equal(first[1][:20], short[1])
    alternate = matched_feature_null(train, train, keys, keys, seed=43)
    assert not np.array_equal(first[0], alternate[0])


def test_matched_null_single_training_row_and_empty_test():
    train = np.array([[4.0, -8.0]])
    train_null, test_null, metadata = matched_feature_null(train, train, ["a"], ["a"])
    np.testing.assert_array_equal(train_null, train)
    np.testing.assert_array_equal(test_null, train)
    assert metadata["train"]["self_donor_rows"] == 1
    assert metadata["train"]["singleton_fallback_rows"] == 1
    assert metadata["test"]["singleton_fallback_rows"] == 1
    _, empty_null, empty_metadata = matched_feature_null(train, np.empty((0, 2)), ["a"], np.array([], dtype=str))
    assert empty_null.shape == (0, 2)
    assert empty_metadata["test"]["exact_key_coverage"] is None
    json.dumps(empty_metadata, allow_nan=False)


def test_matched_null_supports_empty_suffix_keys_and_rejects_invalid_keys():
    train = np.arange(8, dtype=float).reshape(4, 2)
    keys = np.empty((4, 0), dtype=int)
    _, _, metadata = matched_feature_null(train, train, keys, keys)
    assert metadata["n_train_keys"] == 1
    assert metadata["test"]["matched_rows"] == 4
    for bad_keys in ([0, 1], [0, 1, np.nan, 2], np.zeros((4, 2, 1))):
        with pytest.raises(ValueError):
            matched_feature_null(train, train, bad_keys, np.zeros(4))
    with pytest.raises(ValueError):
        matched_feature_null(train, train, np.zeros((4, 2)), np.zeros(4))
    with pytest.raises(ValueError):
        matched_feature_null(train, np.ones((4, 3)), np.zeros(4), np.zeros(4))


def test_gaussian_null_matches_full_training_covariance_and_is_independent():
    rng = np.random.default_rng(61)
    base = rng.normal(size=(6000, 2))
    train = base @ np.array([[1.0, 2.0, -1.0], [0.0, 0.5, 2.0]]) + [3.0, 6.0, 2.0]
    before = train.copy()
    train_null, test_null, metadata = gaussian_feature_null(train, 30000)
    assert train_null.shape == train.shape
    assert test_null.shape == (30000, 3)
    assert metadata["empirical_rank"] == 2
    assert metadata["fit_source"] == "train"
    assert metadata["covariance_ddof"] == 1
    np.testing.assert_allclose(test_null.mean(axis=0), train.mean(axis=0), atol=0.04)
    np.testing.assert_allclose(np.cov(test_null, rowvar=False), np.cov(train, rowvar=False), rtol=0.035, atol=0.04)
    assert abs(np.corrcoef(train_null[:, 0], train[:, 0])[0, 1]) < 0.05
    assert abs(np.corrcoef(train_null[:, 0], test_null[:len(train), 0])[0, 1]) < 0.05
    _, _, vt = np.linalg.svd(train - train.mean(axis=0), full_matrices=False)
    np.testing.assert_allclose((test_null - train.mean(axis=0)) @ vt[-1], 0.0, atol=1e-12)
    repeated = gaussian_feature_null(train, 30000)
    np.testing.assert_array_equal(train_null, repeated[0])
    np.testing.assert_array_equal(test_null, repeated[1])
    assert metadata == repeated[2]
    shorter = gaussian_feature_null(train, 10)
    np.testing.assert_array_equal(train_null, shorter[0])
    np.testing.assert_array_equal(test_null[:10], shorter[1])
    changed = gaussian_feature_null(train, 10, seed=43)
    assert not np.array_equal(train_null, changed[0])
    np.testing.assert_array_equal(train, before)
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("n_train", [1, 20])
def test_gaussian_null_constant_and_singleton_features(n_train):
    train = np.full((n_train, 3), 0.1)
    train_null, test_null, metadata = gaussian_feature_null(train, 7)
    np.testing.assert_array_equal(train_null, train)
    np.testing.assert_array_equal(test_null, np.full((7, 3), 0.1))
    assert metadata["empirical_rank"] == 0
    assert gaussian_feature_null(train, 0)[1].shape == (0, 3)
    for n_test in (-1, 2.5, True):
        with pytest.raises(ValueError):
            gaussian_feature_null(train, n_test)
    with pytest.raises(ValueError):
        gaussian_feature_null(np.empty((0, 3)), 2)


def test_suffix_keys_interleaved_episodes_include_current_symbol_and_warmup():
    symbols = np.array([1, 8, 2, 9, 3, 10, 4])
    episodes = np.array(["a", "b", "a", "b", "a", "b", "a"])
    steps = np.array([0, 0, 1, 1, 2, 2, 3])
    expected = np.array([
        [-1, -1, 1],
        [-1, -1, 8],
        [-1, 1, 2],
        [-1, 8, 9],
        [1, 2, 3],
        [8, 9, 10],
        [2, 3, 4],
    ])
    keys = suffix_keys(symbols, episodes, steps, 3)
    np.testing.assert_array_equal(keys, expected)
    np.testing.assert_array_equal(keys[steps >= 2], [[1, 2, 3], [8, 9, 10], [2, 3, 4]])
    np.testing.assert_array_equal(suffix_keys(symbols, episodes, steps, 1), symbols[:, None])
    assert suffix_keys(symbols, episodes, steps, 0).shape == (7, 0)


def test_suffix_keys_partial_histories_and_empty_integer_arrays():
    np.testing.assert_array_equal(suffix_keys([3, 4], [0, 0], [5, 6], 3), [[-1, -1, 3], [-1, 3, 4]])
    empty = np.array([], dtype=int)
    assert suffix_keys(empty, empty, empty, 3).shape == (0, 3)


@pytest.mark.parametrize(
    "symbols,episodes,steps,length",
    [
        ([1, 2], [0, 0], [1, 0], 2),
        ([1, 2], [0, 0], [0, 0], 2),
        ([1, 2], [0, 0], [0, 2], 2),
        ([1, 2], [0, 0], [0, 2], 0),
        ([1, 2], [0], [0, 1], 2),
        ([1, 2], [0, np.nan], [0, 1], 2),
        ([1, 2], [0, 0], [-1, 0], 2),
        ([1, 2], [0, 0], [0.0, 1.5], 2),
        ([1, -1], [0, 0], [0, 1], 2),
        ([1.0, 2.0], [0, 0], [0, 1], 2),
        ([[1], [2]], [0, 0], [0, 1], 2),
        ([1, 2], [0, 0], [0, 1], -1),
        ([1, 2], [0, 0], [0, 1], 1.5),
    ],
)
def test_suffix_keys_reject_invalid_chronology_or_alignment(symbols, episodes, steps, length):
    with pytest.raises(ValueError):
        suffix_keys(symbols, episodes, steps, length)
