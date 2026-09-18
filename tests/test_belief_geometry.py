from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

import analysis.belief_geometry as geometry
from analysis.belief_geometry import (
    ProbeBatteryResult,
    evaluate_belief_geometry,
    prediction_null_basis,
    select_alternative_beliefs,
)
from analysis.probes import controls
from analysis.probes.transducer import filter_operator_histories, predictive_belief_sequence


@pytest.fixture
def battery():
    rng = np.random.default_rng(17)
    train = rng.dirichlet(np.ones(3), size=120)
    test = rng.dirichlet(np.ones(3), size=80)
    mixing = np.array([[2.0, -1.0, 3.0, 4.0], [-2.0, 5.0, 1.0, 3.0], [3.0, 0.0, -2.0, 1.0]])
    prediction_map = np.array([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8]])
    return {
        "train_features": {"arbitrary/name": train @ mixing + 9.0},
        "test_features": {"arbitrary/name": test @ mixing + 9.0},
        "train_beliefs": train,
        "test_beliefs": test,
        "train_groups": np.repeat(np.arange(12), 10),
        "test_groups": np.repeat(np.arange(8) + 100, 10),
        "nuisance_features": {"oracle": (train @ prediction_map, test @ prediction_map)},
        "contrasts": {"unpredicted": np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)},
        "initialization_features": {"arbitrary/name": (rng.normal(size=(120, 4)), rng.normal(size=(80, 4)))},
        "matched_keys": (np.arange(120) % 4, np.arange(80) % 4),
        "n_null_repeats": 2,
        "n_resamples": 19,
    }


def _records(result):
    yield from result.report["baselines"].values()
    yield from result.report["initialization"].values()
    for record in result.report["representations"].values():
        yield record
        for repeats in record["nulls"].values():
            yield from repeats


@pytest.mark.parametrize("row", [[1.0000001, 0, 0], [0.3, 0.3, 0.40000005], [1.000000005, 0, 0]])
def test_battery_probability_contract_matches_plot_inputs(battery, row):
    battery["train_beliefs"][0] = row
    with pytest.raises(ValueError, match="probability"):
        evaluate_belief_geometry(**battery)


def test_battery_linear_recovery_contrast_surplus_and_grouped_controls(battery):
    result = evaluate_belief_geometry(**battery)
    assert isinstance(result, ProbeBatteryResult)
    json.dumps(result.report, allow_nan=False)
    np.testing.assert_allclose(result.predictions["arbitrary/name"], battery["test_beliefs"], atol=1e-12)
    record = result.report["representations"]["arbitrary/name"]
    assert record["metrics"]["mse"] < 1e-25
    assert record["metrics"]["r_squared"] == pytest.approx(1.0)
    assert record["metrics"]["max_sum_error"] < 1e-12
    assert record["metrics"]["outside_simplex_fraction"] == 0.0
    oracle = record["comparisons"]["baselines"]["nuisance/oracle"]
    contrast = oracle["contrasts"]["unpredicted"]
    assert contrast["mse_improvement"] > 0.01
    assert contrast["residual_fraction_recovered"] == pytest.approx(1.0)
    assert contrast["mse_improvement_ci"][0] > 0.0
    assert contrast["bootstrap_unit"] == "group"
    assert contrast["bootstrap_refit"] is False
    assert contrast["n_groups"] == 8
    assert record["comparisons"]["initialization"]["mse_improvement"] > 0.01
    np.testing.assert_allclose(
        result.baseline_predictions["train_mean"],
        np.broadcast_to(battery["train_beliefs"].mean(axis=0), (80, 3)),
    )
    for control in _records(result):
        fit = control["fit"]
        assert fit["fit_source"] == "train"
        if fit["method"] == "train_mean":
            continue
        assert fit["method"] == "grouped_svd_cutoff_cv"
        folds = fit["fold_validation_groups"]
        assert sorted(group for fold in folds for group in fold) == list(range(12))
        for fold in folds:
            mask = np.isin(battery["train_groups"], fold)
            assert not set(battery["train_groups"][mask]) & set(battery["train_groups"][~mask])
            assert not set(fold) & set(battery["test_groups"])
    for kind, repeats in record["nulls"].items():
        assert len(repeats) == 2
        assert len({item["generation"]["seed"] for item in repeats}) == 2
        for item in repeats:
            prediction = result.baseline_predictions[item["prediction_key"]]
            assert prediction.shape == (80, 3)
            assert item["metrics"]["mse"] > 0.01
            assert item["metrics"]["mse"] == pytest.approx(np.square(prediction - battery["test_beliefs"]).mean())
            if kind == "matched_features":
                assert item["generation"]["donor_source"] == "train"
                assert item["generation"]["test"]["matched_fraction"] == 1.0
    assert len(result.baseline_predictions) == 9


def test_battery_deterministic_training_only_and_immutable(battery, monkeypatch):
    original = deepcopy(battery)
    calls = []
    fit = controls.fit_grouped_affine

    def tracked_fit(x, y, groups, **kwargs):
        assert len(x) == len(y) == len(battery["train_beliefs"])
        np.testing.assert_array_equal(groups, battery["train_groups"])
        calls.append((x.copy(), y.copy()))
        return fit(x, y, groups, **kwargs)

    monkeypatch.setattr(controls, "fit_grouped_affine", tracked_fit)
    first = evaluate_belief_geometry(**battery)
    second = evaluate_belief_geometry(**battery)
    assert first.report == second.report
    for name in first.predictions:
        np.testing.assert_array_equal(first.predictions[name], second.predictions[name])
    changed = deepcopy(battery)
    changed["test_beliefs"] = np.roll(changed["test_beliefs"], 1, axis=1)
    third = evaluate_belief_geometry(**changed)
    assert first.report["representations"]["arbitrary/name"]["metrics"] != third.report["representations"]["arbitrary/name"]["metrics"]
    for left, right in zip(_records(first), _records(third)):
        assert left["fit"] == right["fit"]
    for key, prediction in first.baseline_predictions.items():
        np.testing.assert_array_equal(prediction, third.baseline_predictions[key])
    np.testing.assert_array_equal(first.predictions["arbitrary/name"], third.predictions["arbitrary/name"])
    changed["test_features"]["arbitrary/name"] *= 11.0
    changed["nuisance_features"]["oracle"][1][:] = 0.5
    fourth = evaluate_belief_geometry(**changed)
    for left, right in zip(_records(first), _records(fourth)):
        assert left["fit"] == right["fit"]
    n_fits = len(calls) // 4
    for offset in (n_fits, 2 * n_fits, 3 * n_fits):
        for (x, y), (other_x, other_y) in zip(calls[:n_fits], calls[offset:offset + n_fits]):
            np.testing.assert_array_equal(x, other_x)
            np.testing.assert_array_equal(y, other_y)
    for key in ("train_beliefs", "test_beliefs", "train_groups", "test_groups"):
        np.testing.assert_array_equal(battery[key], original[key])
    for key in ("train_features", "test_features", "contrasts"):
        for name in battery[key]:
            np.testing.assert_array_equal(battery[key][name], original[key][name])
    for key in ("nuisance_features", "initialization_features"):
        for name in battery[key]:
            for left, right in zip(battery[key][name], original[key][name]):
                np.testing.assert_array_equal(left, right)
    for left, right in zip(battery["matched_keys"], original["matched_keys"]):
        np.testing.assert_array_equal(left, right)


def test_battery_constant_targets_are_json_safe(battery):
    for key in ("train_beliefs", "test_beliefs"):
        battery[key][:] = [0.2, 0.3, 0.5]
    result = evaluate_belief_geometry(**battery)
    for record in _records(result):
        assert record["metrics"]["target_variance"] == 0.0
        assert record["metrics"]["r_squared"] is None
        assert record["metrics"]["normalized_mse"] is None
    comparison = result.report["representations"]["arbitrary/name"]["comparisons"]["baselines"]["train_mean"]
    assert comparison["delta_r_squared"] is None
    assert comparison["delta_r_squared_ci"] is None
    json.dumps(result.report, allow_nan=False)


def test_battery_raw_negative_predictions_are_not_clipped():
    x = np.linspace(0.2, 0.8, 40)[:, None]
    result = evaluate_belief_geometry(
        {"not a layer": x}, {"not a layer": np.array([[-2.0], [3.0]])},
        np.column_stack([x, 1.0 - x]), np.array([[0.3, 0.7], [0.6, 0.4]]),
        train_groups=np.arange(40) % 5, test_groups=np.array(["a", "b"]),
        n_null_repeats=1, n_resamples=3,
    )
    prediction = result.predictions["not a layer"]
    np.testing.assert_allclose(prediction, [[-2.0, 3.0], [3.0, -2.0]], atol=1e-12)
    metrics = result.report["representations"]["not a layer"]["metrics"]
    assert metrics["outside_simplex_fraction"] == 1.0
    assert metrics["mse"] == pytest.approx((2.3**2 + 2.4**2) / 2)
    assert metrics["max_sum_error"] < 1e-12
    assert set(result.report["representations"]["not a layer"]["nulls"]) == {"permuted_labels", "gaussian_features"}
    json.dumps(result.report, allow_nan=False)


def test_arbitrary_names_are_collision_free_and_order_independent(battery):
    x = battery["train_features"]["arbitrary/name"]
    test = battery["test_features"]["arbitrary/name"]
    battery["train_features"] = {"train_mean": x, "nuisance/train_mean": x}
    battery["test_features"] = {"nuisance/train_mean": test, "train_mean": test}
    battery["initialization_features"] = None
    battery["nuisance_features"] = {"train_mean": (x, test), "": (x, test)}
    result = evaluate_belief_geometry(**battery)
    assert set(result.report["baselines"]) == {"train_mean", "nuisance/train_mean", "nuisance/"}
    battery["train_features"] = dict(reversed(list(battery["train_features"].items())))
    repeated = evaluate_belief_geometry(**battery)
    assert result.report == repeated.report
    assert set(result.predictions) == {"train_mean", "nuisance/train_mean"}
    for key, prediction in result.baseline_predictions.items():
        np.testing.assert_array_equal(prediction, repeated.baseline_predictions[key])


@pytest.mark.parametrize("key,value", [
    ("train_features", {}),
    ("test_features", {"wrong": np.ones((80, 4))}),
    ("train_features", {"arbitrary/name": np.ones((119, 4))}),
    ("test_features", {"arbitrary/name": np.ones((80, 3))}),
    ("test_features", {"arbitrary/name": np.full((80, 4), np.inf)}),
    ("train_features", {"arbitrary/name": np.ones((120, 4), dtype=complex)}),
    ("train_features", {3: np.ones((120, 4))}),
    ("train_beliefs", np.full((120, 3), 0.5)),
    ("test_beliefs", np.tile([-0.1, 0.5, 0.6], (80, 1))),
    ("test_beliefs", np.full((80, 2), 0.5)),
    ("test_beliefs", np.full((80, 3), np.nan)),
    ("test_beliefs", np.empty((0, 3))),
    ("train_groups", np.zeros(120)),
    ("train_groups", np.zeros(119)),
    ("test_groups", np.full(80, np.nan)),
    ("test_groups", np.zeros((80, 1))),
    ("nuisance_features", {"a": (np.ones((120, 2)), np.ones((80, 1)))}),
    ("nuisance_features", {"a": np.ones((120, 2))}),
    ("initialization_features", {"missing": (np.ones((120, 4)), np.ones((80, 4)))}),
    ("initialization_features", {"arbitrary/name": (np.ones((120, 2)), np.ones((80, 2)))}),
    ("contrasts", {"a": np.ones((3, 1))}),
    ("contrasts", {"a": np.array([1.0, np.nan, 0.0])}),
    ("contrasts", {"a": np.ones(2)}),
    ("contrasts", {"a": np.ones(3, dtype=complex)}),
    ("matched_keys", (np.zeros(119), np.zeros(80))),
    ("matched_keys", (np.zeros((120, 2)), np.zeros(80))),
    ("matched_keys", (np.empty((120, 0)), np.empty((80, 0)))),
    ("matched_keys", (np.zeros(120),)),
    ("n_null_repeats", 0),
    ("n_null_repeats", 1.5),
    ("n_resamples", 0),
    ("seed", True),
])
def test_battery_rejects_invalid_inputs(battery, key, value):
    battery[key] = value
    with pytest.raises(ValueError):
        evaluate_belief_geometry(**battery)


def test_single_test_group_keeps_metrics_without_fake_intervals(battery):
    battery["test_groups"] = np.zeros(80)
    result = evaluate_belief_geometry(**battery)
    comparison = result.report["representations"]["arbitrary/name"]["comparisons"]["baselines"]["train_mean"]
    assert comparison["mse_improvement"] > 0.0
    assert comparison["mse_improvement_ci"] is None
    assert comparison["ci_reason"] == "at_least_two_groups_required"
    json.dumps(result.report, allow_nan=False)


@pytest.mark.parametrize("prediction_map,dimension", [
    (np.array([[0.8, 0.2], [0.8, 0.2], [0.3, 0.7]]), 1),
    (np.array([[-1.0, 2.0], [-1.0, 2.0], [2.0, -1.0]]), 1),
    (np.ones((5, 2)), 4),
    (np.zeros((4, 2, 3)), 3),
    (np.eye(4), 0),
    (np.ones((1, 3)), 0),
    (np.arange(12.0).reshape(3, 2, 2), 1),
])
def test_prediction_null_basis_shape_orthonormality_and_constraints(prediction_map, dimension):
    original = prediction_map.copy()
    basis = prediction_null_basis(prediction_map)
    assert basis.shape == (len(prediction_map), dimension)
    np.testing.assert_allclose(np.ones(len(prediction_map)) @ basis, 0.0, atol=1e-14)
    np.testing.assert_allclose(prediction_map.reshape(len(prediction_map), -1).T @ basis, 0.0, atol=1e-13)
    np.testing.assert_allclose(basis.T @ basis, np.eye(dimension), atol=1e-14)
    np.testing.assert_array_equal(prediction_map, original)


@pytest.mark.parametrize("prediction_map", [np.ones(3), np.empty((0, 2)), np.empty((3, 0)), np.full((3, 2), np.inf), np.ones((3, 2), dtype=complex)])
def test_prediction_null_basis_rejects_bad_maps(prediction_map):
    with pytest.raises(ValueError):
        prediction_null_basis(prediction_map)


@pytest.mark.parametrize("rcond", [-1.0, 1.0, np.inf, np.nan, True, [1e-4], 1j])
def test_prediction_null_basis_rejects_bad_tolerance(rcond):
    with pytest.raises(ValueError):
        prediction_null_basis(np.eye(3), rcond=rcond)


def test_prediction_null_basis_uses_caller_tolerance():
    prediction_map = np.array([[0.8, 0.2, 1e-10], [0.8, 0.2, 0.0], [0.3, 0.7, 0.0]])
    assert prediction_null_basis(prediction_map).shape == (3, 0)
    basis = prediction_null_basis(prediction_map, rcond=1e-8)
    assert basis.shape == (3, 1)
    np.testing.assert_allclose(prediction_map.T @ basis, 0.0, atol=1e-8)


@pytest.fixture
def alternatives():
    rng = np.random.default_rng(11)
    beliefs = rng.dirichlet(np.ones(3), size=240)
    predictions = beliefs @ np.array([[0.8, 0.2], [0.8, 0.2], [0.3, 0.7]])
    distinct = rng.dirichlet(np.ones(3), size=240)
    return {
        "train_beliefs": beliefs,
        "train_predictions": predictions,
        "candidates": {
            "permuted": (beliefs[:, [2, 0, 1]], predictions.copy()),
            "constant": (np.tile([0.2, 0.3, 0.5], (240, 1)), predictions.copy()),
            "distinct": (distinct, predictions.copy()),
            "same residual": (distinct.copy(), predictions.copy()),
            "over budget": (distinct.copy(), np.tile([0.99, 0.01], (240, 1))),
        },
        "groups": np.repeat(np.arange(12), 20),
        "max_prediction_kl": 0.001,
        "min_transfer_error": 0.01,
        "n_select": 3,
    }


def test_alternative_selection_rejects_equivalence_and_ranks_training_only(alternatives, monkeypatch):
    original = deepcopy(alternatives)
    fit = geometry.fit_affine_probe
    fits = []

    def tracked_fit(x, y, **kwargs):
        assert kwargs == {"ridge": 0.0}
        fits.append(x.copy())
        return fit(x, y, **kwargs)

    monkeypatch.setattr(geometry, "fit_affine_probe", tracked_fit)
    result = select_alternative_beliefs(**alternatives)
    assert result["selected"] == ["distinct", "same residual"]
    assert not result["candidates"]["permuted"]["eligible"]
    assert result["candidates"]["permuted"]["transfer_error"] < 1e-25
    assert result["candidates"]["constant"]["transfer_error"] is None
    assert result["candidates"]["constant"]["rejection_reasons"] == ["constant_holdout_target"]
    assert result["candidates"]["over budget"]["prediction_kl"] > 0.001
    assert result["selection_source"] == "train"
    assert len(result["fit_groups"]) == 9
    assert len(result["holdout_groups"]) == 3
    assert not set(result["fit_groups"]) & set(result["holdout_groups"])
    mask = np.isin(alternatives["groups"], result["fit_groups"])
    for x in fits:
        np.testing.assert_array_equal(x, alternatives["train_beliefs"][mask])
    assert result == select_alternative_beliefs(**alternatives)
    json.dumps(result, allow_nan=False)
    for name in alternatives["candidates"]:
        for left, right in zip(alternatives["candidates"][name], original["candidates"][name]):
            np.testing.assert_array_equal(left, right)
    alternatives["n_select"] = 1
    assert select_alternative_beliefs(**alternatives)["selected"] == ["distinct"]
    alternatives["min_transfer_error"] = 10.0
    assert select_alternative_beliefs(**alternatives)["selected"] == []


def test_alternative_selection_rejects_relabeling_when_fit_groups_lack_support():
    beliefs = np.tile([1.0, 0.0, 0.0], (16, 1))
    beliefs[12:] = np.tile([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], (2, 1))
    predictions = np.tile([0.4, 0.6], (16, 1))
    result = select_alternative_beliefs(
        beliefs, predictions, {"relabeled": (beliefs[:, [2, 0, 1]], predictions)},
        np.repeat(np.arange(4), 4), max_prediction_kl=0.0, min_transfer_error=0.01,
    )
    assert result["selected"] == []


def test_forward_kl_penalizes_but_does_not_forbid_extra_support(alternatives):
    alternatives["train_predictions"] = np.tile([1.0, 0.0], (240, 1))
    alternatives["candidates"] = {
        "extra_support": (alternatives["candidates"]["distinct"][0], np.tile([0.9, 0.1], (240, 1)))
    }
    alternatives["max_prediction_kl"] = 0.05
    result = select_alternative_beliefs(**alternatives)
    assert result["candidates"]["extra_support"]["prediction_kl"] == pytest.approx(-np.log(0.9))
    assert result["selected"] == []
    alternatives["max_prediction_kl"] = 0.2
    assert select_alternative_beliefs(**alternatives)["selected"] == ["extra_support"]


def test_alternative_kl_support_zeros_are_json_safe(alternatives):
    alternatives["train_predictions"] = np.tile([0.0, 0.4, 0.6], (240, 1))
    distinct = alternatives["candidates"]["distinct"][0]
    alternatives["candidates"] = {
        "finite": (distinct, np.tile([0.0, 0.5, 0.5], (240, 1))),
        "impossible": (distinct, np.tile([0.0, 0.0, 1.0], (240, 1))),
    }
    alternatives["max_prediction_kl"] = 0.1
    result = select_alternative_beliefs(**alternatives)
    assert result["selected"] == ["finite"]
    assert result["candidates"]["finite"]["prediction_kl"] == pytest.approx(0.4 * np.log(0.8) + 0.6 * np.log(1.2))
    impossible = result["candidates"]["impossible"]
    assert impossible["prediction_kl"] is None
    assert impossible["prediction_kl_is_infinite"] is True
    assert impossible["rejection_reasons"] == ["infinite_prediction_kl_support_mismatch"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("key,value", [
    ("train_beliefs", np.full((240, 3), np.nan)),
    ("train_predictions", np.full((239, 2), 0.5)),
    ("groups", np.zeros(240)),
    ("groups", np.zeros(239)),
    ("candidates", {"bad": (np.full((240, 2), 0.5), np.full((240, 2), 0.5))}),
    ("candidates", {"bad": (np.full((240, 3), 1 / 3), np.full((240, 3), 1 / 3))}),
    ("candidates", {"bad": (np.tile([-0.1, 0.5, 0.6], (240, 1)), np.full((240, 2), 0.5))}),
    ("candidates", {"bad": (np.full((240, 3), 1 / 3), np.ones((240, 2)))}),
    ("max_prediction_kl", -1.0),
    ("max_prediction_kl", np.inf),
    ("max_prediction_kl", True),
    ("min_transfer_error", np.nan),
    ("min_transfer_error", 1j),
    ("min_transfer_error", [0.1]),
    ("n_select", 0),
    ("seed", -1),
])
def test_alternative_selection_rejects_bad_inputs(alternatives, key, value):
    alternatives[key] = value
    with pytest.raises(ValueError):
        select_alternative_beliefs(**alternatives)


@pytest.fixture
def histories():
    groups = np.array(["x", "y", "x", "z", "y", "x", "z", "y", "x"])
    steps = np.array([0, 0, 1, 0, 1, 2, 1, 2, 3])
    rng = np.random.default_rng(3)
    operators = rng.uniform(0.01, 0.2, size=(len(groups), 3, 3))
    operators[steps == 0] = np.eye(3)
    return {"initial_belief": np.array([0.2, 0.3, 0.5]), "operators": operators, "groups": groups, "steps": steps}


@pytest.mark.parametrize("suffix_length", [None, 0, 1, 2, 3, 4, 100])
def test_operator_histories_interleaved_full_and_suffix_match_reference(histories, suffix_length):
    original = deepcopy(histories)
    beliefs = filter_operator_histories(**histories, suffix_length=suffix_length)
    assert beliefs.shape == (9, 3)
    for index, group in enumerate(histories["groups"]):
        rows = np.flatnonzero(histories["groups"][:index + 1] == group)
        if suffix_length is not None:
            rows = rows[-suffix_length:] if suffix_length else rows[:0]
        reference = predictive_belief_sequence(histories["initial_belief"], histories["operators"][rows])[-1]
        np.testing.assert_allclose(beliefs[index], reference, atol=1e-14)
    full = filter_operator_histories(**histories)
    if suffix_length:
        prefix = histories["steps"] < suffix_length
        np.testing.assert_allclose(beliefs[prefix], full[prefix], atol=1e-14)
    for key in histories:
        np.testing.assert_array_equal(histories[key], original[key])


def test_operator_histories_apply_caller_reset_emission_and_empty_histories(histories):
    histories["operators"][0] = np.diag([0.1, 0.2, 0.6])
    beliefs = filter_operator_histories(**histories)
    expected = histories["initial_belief"] @ histories["operators"][0]
    np.testing.assert_allclose(beliefs[0], expected / expected.sum())
    result = filter_operator_histories(np.array([0.4, 0.6]), np.empty((0, 2, 2)), np.array([], dtype=int), np.array([], dtype=int))
    assert result.shape == (0, 2)


@pytest.mark.parametrize("key,value", [
    ("initial_belief", np.array([0.2, 0.3, 0.6])),
    ("initial_belief", np.array([-0.2, 0.7, 0.5])),
    ("initial_belief", np.ones(3, dtype=complex) / 3),
    ("operators", np.ones((9, 2, 2))),
    ("operators", np.ones((9, 3, 3))),
    ("operators", np.full((9, 3, 3), np.nan)),
    ("operators", np.full((9, 3, 3), -0.1)),
    ("operators", np.zeros((9, 3, 3))),
    ("steps", np.arange(9)),
    ("steps", np.array([0, 0, 1, 0, 1, 3, 1, 2, 4])),
    ("steps", np.array([0, 0, 1, 0, 1, 1, 1, 2, 3])),
    ("steps", np.array([0.0, 0, 1, 0, 1, 2, 1, 2, 3])),
    ("steps", np.arange(8)),
    ("groups", np.full(9, np.nan)),
    ("groups", np.ones(8)),
    ("suffix_length", -1),
    ("suffix_length", True),
])
def test_operator_histories_reject_invalid_inputs(histories, key, value):
    histories[key] = value
    with pytest.raises(ValueError):
        filter_operator_histories(**histories)


def test_operator_histories_reject_warmup_mask_and_zero_mass_suffix(histories):
    for key in ("operators", "groups", "steps"):
        histories[key] = histories[key][2:]
    with pytest.raises(ValueError, match="start at step zero"):
        filter_operator_histories(**histories, suffix_length=0)
    with pytest.raises(ValueError, match="zero probability"):
        filter_operator_histories(
            np.array([1.0, 0.0]), np.array([np.eye(2), np.diag([0.0, 1.0])]),
            np.array([0, 0]), np.array([0, 1]), suffix_length=1,
        )


def test_operator_histories_twenty_thousand_rows():
    n_rows = 20_000
    operator = np.array([[0.2, 0.1, 0.1], [0.1, 0.3, 0.1], [0.05, 0.1, 0.2]])
    prior = np.array([0.2, 0.3, 0.5])
    operators = np.broadcast_to(operator, (n_rows, 3, 3))
    groups = np.arange(n_rows) % 100
    steps = np.arange(n_rows) // 100
    full = filter_operator_histories(prior, operators, groups, steps)
    suffix = filter_operator_histories(prior, operators, groups, steps, suffix_length=32)
    for index in [0, 137, 931, 7_935, 19_999]:
        sequence = predictive_belief_sequence(prior, np.broadcast_to(operator, (int(steps[index]) + 1, 3, 3)))
        np.testing.assert_allclose(full[index], sequence[-1], atol=1e-14)
        np.testing.assert_allclose(suffix[index], sequence[min(32, len(sequence) - 1)], atol=1e-14)
