from __future__ import annotations

from collections import deque

import numpy as np

from analysis.probes.resampling import (
    cluster_bootstrap_statistics,
    percentile_interval,
)


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _matrix(values: np.ndarray, name: str, *, allow_empty: bool = False) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a matrix with nonzero width")
    if not allow_empty and len(values) == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def _key_rows(values: np.ndarray, name: str, n_rows: int) -> tuple[list[tuple], int]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) != n_rows:
        raise ValueError(f"{name} must contain one scalar or key row per sample")
    rows = []
    for row in values.tolist():
        for value in row:
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"{name} must contain scalar strings or finite numbers")
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"{name} must contain finite labels")
        rows.append(tuple(row))
    return rows, values.shape[1]


def _group_data(groups: np.ndarray, n_rows: int) -> tuple[list, np.ndarray]:
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    rows, _ = _key_rows(groups, "groups", n_rows)
    labels = {}
    codes = np.empty(n_rows, dtype=np.int64)
    for index, row in enumerate(rows):
        if row[0] not in labels:
            labels[row[0]] = len(labels)
        codes[index] = labels[row[0]]
    return list(labels), codes


def _mean(values: np.ndarray) -> np.ndarray:
    return values[0] + (values - values[0]).mean(axis=0)


def _finite_float(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("numerical result exceeds the finite float64 range")
    return value


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else _finite_float(numerator / denominator)


def _svd_design(x: np.ndarray, y: np.ndarray) -> tuple:
    x_mean = _mean(x)
    y_mean = _mean(y)
    u, singular_values, vt = np.linalg.svd(x - x_mean, full_matrices=False)
    return x_mean, y_mean, singular_values, vt, u.T @ (y - y_mean)


def _svd_weight(decomposition: tuple, rcond: float) -> tuple[np.ndarray, np.ndarray, int]:
    x_mean, y_mean, singular_values, vt, projected = decomposition
    keep = singular_values > rcond * singular_values[0]
    inverse = np.zeros_like(singular_values)
    np.divide(1.0, singular_values, out=inverse, where=keep)
    weight = vt.T @ (inverse[:, None] * projected)
    return weight, y_mean - x_mean @ weight, int(keep.sum())


def fit_grouped_affine(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int = 42,
    folds: int = 5,
    rconds: tuple[float, ...] = (1e-12, 1e-8, 1e-4, 1e-2),
) -> tuple[np.ndarray, np.ndarray, dict]:
    x = _matrix(x, "x")
    y = _matrix(y, "y")
    if len(x) != len(y):
        raise ValueError("x and y must contain equal samples")
    labels, group_codes = _group_data(groups, len(x))
    if len(labels) < 2:
        raise ValueError("at least two groups are required; timestep splitting is not allowed")
    seed = _integer(seed, "seed", 0)
    requested_folds = _integer(folds, "folds", 2)
    cutoffs = np.asarray(rconds, dtype=np.float64)
    if cutoffs.ndim != 1 or len(cutoffs) == 0:
        raise ValueError("rconds must be a nonempty one-dimensional sequence")
    if not np.isfinite(cutoffs).all() or (cutoffs < 0.0).any():
        raise ValueError("rconds must be finite and nonnegative")
    folds = min(requested_folds, len(labels))
    assignments = np.array_split(np.random.default_rng(seed).permutation(len(labels)), folds)
    fold_mse = np.empty((folds, len(cutoffs)), dtype=np.float64)
    validation_sizes = []
    for fold, validation_groups in enumerate(assignments):
        validation = np.isin(group_codes, validation_groups)
        validation_sizes.append(int(validation.sum()))
        decomposition = _svd_design(x[~validation], y[~validation])
        for candidate, rcond in enumerate(cutoffs):
            weight, bias, _ = _svd_weight(decomposition, float(rcond))
            fold_mse[fold, candidate] = _finite_float(
                np.square(x[validation] @ weight + bias - y[validation]).mean()
            )
    cv_mse = np.average(fold_mse, axis=0, weights=validation_sizes)
    selected = int(np.argmin(cv_mse))
    decomposition = _svd_design(x, y)
    weight, bias, rank = _svd_weight(decomposition, float(cutoffs[selected]))
    fit = {
        "method": "grouped_svd_cutoff_cv",
        "seed": seed,
        "n_samples": len(x),
        "n_features": x.shape[1],
        "n_targets": y.shape[1],
        "n_groups": len(labels),
        "requested_folds": requested_folds,
        "folds": folds,
        "rconds": cutoffs.tolist(),
        "selected_rcond": float(cutoffs[selected]),
        "cv_mse": cv_mse.tolist(),
        "fold_mse": fold_mse.tolist(),
        "cv_weighting": "sample",
        "fold_validation_groups": [[labels[int(code)] for code in fold] for fold in assignments],
        "fold_validation_sizes": validation_sizes,
        "rank": rank,
        "singular_value_cutoff": _finite_float(cutoffs[selected] * decomposition[2][0]),
        "fit_source": "train",
    }
    return weight, bias, fit


def _prediction_pair(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if pred.ndim == 1:
        pred = pred[:, None]
        target = target[:, None]
    return _matrix(pred, "prediction"), _matrix(target, "target")


def score_prediction(pred: np.ndarray, target: np.ndarray) -> dict:
    pred, target = _prediction_pair(pred, target)
    with np.errstate(over="ignore", invalid="ignore"):
        mse = _finite_float(np.square(pred - target).mean())
        variance = _finite_float(np.square(target - _mean(target)).mean())
    normalized_mse = _ratio(mse, variance)
    return {
        "mse": mse,
        "target_variance": variance,
        "normalized_mse": normalized_mse,
        "r_squared": None if normalized_mse is None else 1.0 - normalized_mse,
        "n_evaluated": len(target),
    }


def paired_comparison(
    prediction: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int = 42,
    n_resamples: int = 200,
) -> dict:
    prediction, target_matrix = _prediction_pair(prediction, target)
    baseline, _ = _prediction_pair(baseline, target)
    target = target_matrix
    labels, group_codes = _group_data(groups, len(target))
    seed = _integer(seed, "seed", 0)
    n_resamples = _integer(n_resamples, "n_resamples", 1)
    prediction_score = score_prediction(prediction, target)
    baseline_score = score_prediction(baseline, target)
    improvement = baseline_score["mse"] - prediction_score["mse"]
    residual_ratio = _ratio(prediction_score["mse"], baseline_score["mse"])
    result = {
        "prediction": prediction_score,
        "baseline": baseline_score,
        "mse_improvement": improvement,
        "delta_r_squared": _ratio(improvement, prediction_score["target_variance"]),
        "residual_fraction_recovered": None if residual_ratio is None else 1.0 - residual_ratio,
        "mse_improvement_ci": None,
        "delta_r_squared_ci": None,
        "ci_reason": None,
        "delta_r_squared_ci_reason": None,
        "delta_r_squared_valid_resamples": 0,
        "n_evaluated": len(target),
        "n_groups": len(labels),
        "n_resamples": 0,
        "requested_resamples": n_resamples,
        "confidence": 0.95,
        "seed": seed,
        "bootstrap_unit": "group",
        "bootstrap_refit": False,
    }
    if len(labels) < 2:
        result["ci_reason"] = "at_least_two_groups_required"
        result["delta_r_squared_ci_reason"] = result["ci_reason"]
        return result
    row_improvement = (
        np.square(baseline - target).mean(axis=1)
        - np.square(prediction - target).mean(axis=1)
    )
    delta_resamples = []

    def statistic(indices: np.ndarray) -> float:
        mse_improvement = _finite_float(row_improvement[indices].mean())
        sampled_target = target[indices]
        variance = _finite_float(np.square(sampled_target - _mean(sampled_target)).mean())
        delta_resamples.append(_ratio(mse_improvement, variance))
        return mse_improvement

    estimates = cluster_bootstrap_statistics(
        group_codes,
        statistic,
        n_resamples=n_resamples,
        seed=seed,
    )
    result["n_resamples"] = n_resamples
    result["mse_improvement_ci"] = list(percentile_interval(estimates))
    result["delta_r_squared_valid_resamples"] = sum(value is not None for value in delta_resamples)
    if result["delta_r_squared_valid_resamples"] == n_resamples:
        result["delta_r_squared_ci"] = list(percentile_interval(np.asarray(delta_resamples)))
    else:
        result["delta_r_squared_ci_reason"] = (
            "zero_target_variance" if prediction_score["target_variance"] == 0.0
            else "zero_target_variance_in_bootstrap_samples"
        )
    return result


def paired_target_comparison(
    prediction: np.ndarray,
    target: np.ndarray,
    alternative_prediction: np.ndarray,
    alternative_target: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int = 42,
    n_resamples: int = 200,
) -> dict:
    prediction, target = _prediction_pair(prediction, target)
    alternative_prediction, alternative_target = _prediction_pair(
        alternative_prediction, alternative_target
    )
    if len(target) != len(alternative_target):
        raise ValueError("true and alternative targets must contain equal samples")
    labels, group_codes = _group_data(groups, len(target))
    seed = _integer(seed, "seed", 0)
    n_resamples = _integer(n_resamples, "n_resamples", 1)
    true_score = score_prediction(prediction, target)
    alternative_score = score_prediction(alternative_prediction, alternative_target)
    zero_variance_targets = [
        name for name, score in (("true", true_score), ("alternative", alternative_score))
        if score["target_variance"] == 0.0
    ]
    result = {
        "true": true_score,
        "alternative": alternative_score,
        "r_squared_difference": None if zero_variance_targets else _finite_float(
            alternative_score["normalized_mse"] - true_score["normalized_mse"]
        ),
        "r_squared_difference_ci": None,
        "estimate_reason": "zero_target_variance" if zero_variance_targets else None,
        "ci_reason": None,
        "zero_variance_targets": zero_variance_targets,
        "valid_resamples": 0,
        "n_evaluated": len(target),
        "n_groups": len(labels),
        "n_resamples": 0,
        "requested_resamples": n_resamples,
        "confidence": 0.95,
        "seed": seed,
        "bootstrap_unit": "group",
        "bootstrap_refit": False,
    }
    if len(labels) < 2:
        result["ci_reason"] = "at_least_two_groups_required"
        return result
    if zero_variance_targets:
        result["ci_reason"] = "zero_target_variance"
        return result
    true_row_mse = np.square(prediction - target).mean(axis=1)
    alternative_row_mse = np.square(alternative_prediction - alternative_target).mean(axis=1)

    def statistic(indices: np.ndarray) -> float:
        true_sample = target[indices]
        alternative_sample = alternative_target[indices]
        true_variance = _finite_float(np.square(true_sample - _mean(true_sample)).mean())
        alternative_variance = _finite_float(
            np.square(alternative_sample - _mean(alternative_sample)).mean()
        )
        if true_variance == 0.0 or alternative_variance == 0.0:
            return float("nan")
        true_ratio = _ratio(_finite_float(true_row_mse[indices].mean()), true_variance)
        alternative_ratio = _ratio(
            _finite_float(alternative_row_mse[indices].mean()), alternative_variance
        )
        return _finite_float(alternative_ratio - true_ratio)

    estimates = cluster_bootstrap_statistics(
        group_codes, statistic, n_resamples=n_resamples, seed=seed
    )
    result["n_resamples"] = n_resamples
    result["valid_resamples"] = int(np.isfinite(estimates).sum())
    if result["valid_resamples"] == n_resamples:
        result["r_squared_difference_ci"] = list(percentile_interval(estimates))
    else:
        result["ci_reason"] = "zero_target_variance_in_bootstrap_samples"
    return result


def matched_feature_null(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_keys: np.ndarray,
    test_keys: np.ndarray,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    train_features = _matrix(train_features, "train_features")
    test_features = _matrix(test_features, "test_features", allow_empty=True)
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("train and test features must have equal widths")
    train_rows, key_width = _key_rows(train_keys, "train_keys", len(train_features))
    test_rows, test_key_width = _key_rows(test_keys, "test_keys", len(test_features))
    if key_width != test_key_width:
        raise ValueError("train and test keys must have equal widths")
    seed = _integer(seed, "seed", 0)
    pools = {}
    positions = np.empty(len(train_features), dtype=np.int64)
    for index, key in enumerate(train_rows):
        pool = pools.setdefault(key, [])
        positions[index] = len(pool)
        pool.append(index)
    train_stream, test_stream = np.random.SeedSequence(seed).spawn(2)

    def sample(rows: list[tuple], stream: np.random.SeedSequence, training: bool) -> tuple:
        rng = np.random.default_rng(stream)
        indices = np.empty(len(rows), dtype=np.int64)
        counts = {
            "n_rows": len(rows),
            "exact_key_rows": 0,
            "matched_rows": 0,
            "fallback_rows": 0,
            "singleton_fallback_rows": 0,
            "unseen_fallback_rows": 0,
            "self_excluded_rows": 0,
            "self_donor_rows": 0,
        }
        for index, key in enumerate(rows):
            pool = pools.get(key, [])
            counts["exact_key_rows"] += int(bool(pool))
            if len(pool) >= 2:
                counts["matched_rows"] += 1
                if training:
                    choice = int(rng.integers(len(pool) - 1))
                    choice += int(choice >= positions[index])
                    counts["self_excluded_rows"] += 1
                else:
                    choice = int(rng.integers(len(pool)))
                indices[index] = pool[choice]
            else:
                counts["fallback_rows"] += 1
                counts["singleton_fallback_rows" if pool else "unseen_fallback_rows"] += 1
                if training and len(train_features) > 1:
                    choice = int(rng.integers(len(train_features) - 1))
                    indices[index] = choice + int(choice >= index)
                    counts["self_excluded_rows"] += 1
                else:
                    indices[index] = int(rng.integers(len(train_features)))
            if training:
                counts["self_donor_rows"] += int(indices[index] == index)
        counts["exact_key_coverage"] = _ratio(counts["exact_key_rows"], len(rows))
        counts["matched_fraction"] = _ratio(counts["matched_rows"], len(rows))
        return train_features[indices], counts

    train_null, train_counts = sample(train_rows, train_stream, True)
    test_null, test_counts = sample(test_rows, test_stream, False)
    metadata = {
        "method": "train_row_resampling_conditional_on_exact_key",
        "seed": seed,
        "donor_source": "train",
        "fallback": "training_marginal",
        "minimum_key_pool_size": 2,
        "training_self_exclusion": "whenever_an_alternative_exists",
        "n_train": len(train_features),
        "n_test": len(test_features),
        "n_features": train_features.shape[1],
        "key_width": key_width,
        "n_train_keys": len(pools),
        "train": train_counts,
        "test": test_counts,
    }
    return train_null, test_null, metadata


def gaussian_feature_null(
    train_features: np.ndarray,
    n_test: int,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    train_features = _matrix(train_features, "train_features")
    n_test = _integer(n_test, "n_test", 0)
    seed = _integer(seed, "seed", 0)
    mean = _mean(train_features)
    _, singular_values, vt = np.linalg.svd(train_features - mean, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(train_features.shape) * singular_values[0]
    keep = singular_values > tolerance
    rank = int(keep.sum())
    ddof = int(len(train_features) > 1)
    factor = vt[keep].T * (singular_values[keep] / np.sqrt(len(train_features) - ddof))
    train_stream, test_stream = np.random.SeedSequence(seed).spawn(2)
    train_null = np.random.default_rng(train_stream).standard_normal((len(train_features), rank)) @ factor.T + mean
    test_null = np.random.default_rng(test_stream).standard_normal((n_test, rank)) @ factor.T + mean
    metadata = {
        "method": "gaussian_train_full_covariance",
        "seed": seed,
        "fit_source": "train",
        "n_train": len(train_features),
        "n_test": n_test,
        "n_features": train_features.shape[1],
        "empirical_rank": rank,
        "covariance_ddof": ddof,
        "singular_value_cutoff": _finite_float(tolerance),
        "training_mean": mean.tolist(),
    }
    return train_null, test_null, metadata


def suffix_keys(
    symbols: np.ndarray,
    episode_ids: np.ndarray,
    steps: np.ndarray,
    length: int,
) -> np.ndarray:
    length = _integer(length, "length", 0)
    symbols = np.asarray(symbols)
    steps = np.asarray(steps)
    if symbols.ndim != 1 or steps.shape != symbols.shape:
        raise ValueError("symbols and steps must be aligned one-dimensional arrays")
    for name, values in (("symbols", symbols), ("steps", steps)):
        if values.dtype.kind not in "iu" or (values < 0).any():
            raise ValueError(f"{name} must contain nonnegative integers")
        if values.size and int(values.max()) > np.iinfo(np.int64).max:
            raise ValueError(f"{name} must fit in int64")
    _, episode_codes = _group_data(episode_ids, len(symbols))
    keys = np.full((len(symbols), length), -1, dtype=np.int64)
    histories = {}
    previous_steps = {}
    for index, (symbol, episode, step) in enumerate(zip(symbols, episode_codes, steps)):
        episode = int(episode)
        step = int(step)
        if episode in previous_steps and step != previous_steps[episode] + 1:
            raise ValueError("steps must be consecutive and increasing within each episode")
        previous_steps[episode] = step
        history = histories.setdefault(episode, deque(maxlen=length))
        history.append(int(symbol))
        if history:
            keys[index, -len(history):] = history
    return keys
