from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from analysis.probes import controls
from analysis.probes.linear import fit_affine_probe, split_group_indices


@dataclass
class ProbeBatteryResult:
    report: dict
    predictions: dict[str, np.ndarray]
    baseline_predictions: dict[str, np.ndarray]


def _probabilities(values: np.ndarray, name: str) -> np.ndarray:
    values = controls._matrix(values, name)
    if (values < 0.0).any() or (values > 1.0).any() or not np.allclose(
        values.sum(axis=1), 1.0, rtol=0.0, atol=1e-8
    ):
        raise ValueError(f"{name} must contain probability rows")
    return values


def _named(values: Mapping, name: str) -> Mapping:
    if not isinstance(values, Mapping) or any(not isinstance(key, str) for key in values):
        raise ValueError(f"{name} must be a mapping with string names")
    return values


def _feature_pairs(values: Mapping, name: str, n_train: int, n_test: int) -> dict:
    pairs = {}
    for key, pair in _named(values, name).items():
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"{name}[{key!r}] must be a train/test pair")
        train = controls._matrix(pair[0], f"{name}[{key!r}].train")
        test = controls._matrix(pair[1], f"{name}[{key!r}].test")
        if len(train) != n_train or len(test) != n_test or train.shape[1] != test.shape[1]:
            raise ValueError(f"{name}[{key!r}] must have aligned rows and equal widths")
        pairs[key] = (train, test)
    return pairs


def _metrics(prediction: np.ndarray, target: np.ndarray, contrasts: Mapping) -> dict:
    score = controls.score_prediction(prediction, target)
    sum_error = np.abs(prediction.sum(axis=1) - 1.0)
    score.update(
        outside_simplex_fraction=float(np.mean(
            (prediction < -1e-8).any(axis=1)
            | (prediction > 1.0 + 1e-8).any(axis=1)
            | (sum_error > 1e-8)
        )),
        simplex_tolerance=1e-8,
        max_sum_error=controls._finite_float(sum_error.max()),
        contrasts={
            name: controls.score_prediction(prediction @ vector, target @ vector)
            for name, vector in contrasts.items()
        },
    )
    return score


def evaluate_belief_geometry(
    train_features: Mapping[str, np.ndarray],
    test_features: Mapping[str, np.ndarray],
    train_beliefs: np.ndarray,
    test_beliefs: np.ndarray,
    *,
    train_groups: np.ndarray,
    test_groups: np.ndarray,
    nuisance_features: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    contrasts: Mapping[str, np.ndarray] | None = None,
    initialization_features: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    matched_keys: tuple[np.ndarray, np.ndarray] | None = None,
    seed: int = 42,
    n_null_repeats: int = 5,
    n_resamples: int = 200,
) -> ProbeBatteryResult:
    seed = controls._integer(seed, "seed", 0)
    n_null_repeats = controls._integer(n_null_repeats, "n_null_repeats", 1)
    n_resamples = controls._integer(n_resamples, "n_resamples", 1)
    train_beliefs = _probabilities(train_beliefs, "train_beliefs")
    test_beliefs = _probabilities(test_beliefs, "test_beliefs")
    if train_beliefs.shape[1] != test_beliefs.shape[1]:
        raise ValueError("train and test beliefs must have equal widths")
    n_train, n_test = len(train_beliefs), len(test_beliefs)
    train_labels, _ = controls._group_data(train_groups, n_train)
    test_labels, test_codes = controls._group_data(test_groups, n_test)
    if len(train_labels) < 2:
        raise ValueError("at least two training groups are required")
    train_features = _named(train_features, "train_features")
    test_features = _named(test_features, "test_features")
    if not train_features or train_features.keys() != test_features.keys():
        raise ValueError("train and test features must have the same nonempty set of names")
    features = _feature_pairs(
        {name: (value, test_features[name]) for name, value in train_features.items()},
        "features", n_train, n_test,
    )
    nuisance = _feature_pairs(
        {} if nuisance_features is None else nuisance_features, "nuisance_features", n_train, n_test
    )
    initialization = _feature_pairs(
        {} if initialization_features is None else initialization_features,
        "initialization_features", n_train, n_test,
    )
    for name, (train, _) in initialization.items():
        if name not in features or train.shape[1] != features[name][0].shape[1]:
            raise ValueError("initialization features must match representation names and widths")
    vectors = {}
    for name, vector in _named({} if contrasts is None else contrasts, "contrasts").items():
        if np.iscomplexobj(vector):
            raise ValueError("contrasts must be real")
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (train_beliefs.shape[1],) or not np.isfinite(vector).all():
            raise ValueError("contrasts must be finite vectors matching the belief width")
        vectors[name] = vector
    if matched_keys is not None:
        if not isinstance(matched_keys, (tuple, list)) or len(matched_keys) != 2:
            raise ValueError("matched_keys must be a train/test pair")
        _, train_width = controls._key_rows(matched_keys[0], "train_keys", n_train)
        _, test_width = controls._key_rows(matched_keys[1], "test_keys", n_test)
        if train_width == 0 or train_width != test_width:
            raise ValueError("matched keys must have equal nonzero widths")

    predictions = {}
    baseline_predictions = {}
    report = {
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        "n_states": train_beliefs.shape[1],
        "n_train_groups": len(train_labels),
        "n_test_groups": len(test_labels),
        "n_null_repeats": n_null_repeats,
        "n_resamples": n_resamples,
        "fit_source": "train",
        "contrasts": {name: vector.tolist() for name, vector in vectors.items()},
        "baselines": {},
        "initialization": {},
        "representations": {},
    }

    def fit(train: np.ndarray, test: np.ndarray, target: np.ndarray) -> tuple:
        weight, bias, metadata = controls.fit_grouped_affine(
            train, target, train_groups, seed=seed
        )
        prediction = test @ weight + bias
        return prediction, {"fit": metadata, "metrics": _metrics(prediction, test_beliefs, vectors)}

    def store(key: str, prediction: np.ndarray, record: dict) -> dict:
        baseline_predictions[key] = prediction
        return {"prediction_key": key, **record}

    def compare(prediction: np.ndarray, baseline: np.ndarray) -> dict:
        result = controls.paired_comparison(
            prediction, baseline, test_beliefs, test_codes, seed=seed, n_resamples=n_resamples
        )
        result["contrasts"] = {
            name: controls.paired_comparison(
                prediction @ vector, baseline @ vector, test_beliefs @ vector,
                test_codes, seed=seed, n_resamples=n_resamples,
            )
            for name, vector in vectors.items()
        }
        return result

    mean_prediction = np.broadcast_to(controls._mean(train_beliefs), test_beliefs.shape).copy()
    report["baselines"]["train_mean"] = store("train_mean", mean_prediction, {
        "fit": {"method": "train_mean", "fit_source": "train", "n_samples": n_train},
        "metrics": _metrics(mean_prediction, test_beliefs, vectors),
    })
    for name, (train, test) in nuisance.items():
        prediction, record = fit(train, test, train_beliefs)
        key = f"nuisance/{name}"
        report["baselines"][key] = store(key, prediction, record)
    for name, (train, test) in initialization.items():
        prediction, record = fit(train, test, train_beliefs)
        report["initialization"][name] = store(f"initialization/{name}", prediction, record)

    repeat_seeds = [
        int(stream.generate_state(1)[0])
        for stream in np.random.SeedSequence(seed).spawn(n_null_repeats)
    ]
    for name, (train, test) in features.items():
        prediction, record = fit(train, test, train_beliefs)
        predictions[name] = prediction
        record["prediction_key"] = name
        record["comparisons"] = {
            "baselines": {
                key: compare(prediction, baseline_predictions[key]) for key in report["baselines"]
            },
            "initialization": (
                compare(prediction, baseline_predictions[f"initialization/{name}"])
                if name in initialization else None
            ),
        }
        record["nulls"] = {"permuted_labels": [], "gaussian_features": []}
        if matched_keys is not None:
            record["nulls"]["matched_features"] = []
        for repeat, repeat_seed in enumerate(repeat_seeds):
            shuffled = np.random.default_rng(repeat_seed).permutation(n_train)
            null_prediction, null_record = fit(train, test, train_beliefs[shuffled])
            null_record["generation"] = {
                "method": "training_target_row_permutation", "seed": repeat_seed,
                "permutation_source": "train", "evaluation_target": "unpermuted_test",
            }
            key = f"null/permuted_labels/{repeat}/{name}"
            record["nulls"]["permuted_labels"].append(store(key, null_prediction, null_record))
            null_features = {
                "gaussian_features": controls.gaussian_feature_null(train, n_test, seed=repeat_seed)
            }
            if matched_keys is not None:
                null_features["matched_features"] = controls.matched_feature_null(
                    train, test, *matched_keys, seed=repeat_seed
                )
            for kind, (null_train, null_test, metadata) in null_features.items():
                null_prediction, null_record = fit(null_train, null_test, train_beliefs)
                null_record["generation"] = metadata
                key = f"null/{kind}/{repeat}/{name}"
                record["nulls"][kind].append(store(key, null_prediction, null_record))
        report["representations"][name] = record
    return ProbeBatteryResult(report, predictions, baseline_predictions)


def prediction_null_basis(prediction_map: np.ndarray, *, rcond: float | None = None) -> np.ndarray:
    if np.iscomplexobj(prediction_map):
        raise ValueError("prediction_map must be real")
    prediction_map = np.asarray(prediction_map, dtype=np.float64)
    if prediction_map.ndim < 2 or not all(prediction_map.shape) or not np.isfinite(prediction_map).all():
        raise ValueError("prediction_map must have finite nonempty state and output dimensions")
    if rcond is not None:
        if isinstance(rcond, (bool, np.bool_)) or np.ndim(rcond) != 0 or np.iscomplexobj(rcond):
            raise ValueError("rcond must be a finite scalar in [0, 1)")
        rcond = float(rcond)
        if not np.isfinite(rcond) or not 0.0 <= rcond < 1.0:
            raise ValueError("rcond must be a finite scalar in [0, 1)")
    n_states = len(prediction_map)
    prediction_map = prediction_map.reshape(n_states, -1)
    _, _, vt = np.linalg.svd(np.ones((1, n_states)), full_matrices=True)
    tangent = vt[1:].T
    if n_states == 1:
        return tangent
    scale = np.abs(prediction_map).max()
    projected = (prediction_map / (scale if scale else 1.0)).T @ tangent
    _, singular_values, vt = np.linalg.svd(
        projected, full_matrices=projected.shape[0] < projected.shape[1]
    )
    tolerance = (
        np.finfo(np.float64).eps * max(prediction_map.shape) if rcond is None else rcond
    )
    rank = int(np.count_nonzero(singular_values > tolerance * max(1.0, singular_values[0])))
    return tangent @ vt[rank:].T


def _threshold(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0 or np.iscomplexobj(value) or value is None:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return value


def select_alternative_beliefs(
    train_beliefs: np.ndarray,
    train_predictions: np.ndarray,
    candidates: Mapping[str, tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
    *,
    max_prediction_kl: float,
    min_transfer_error: float,
    n_select: int = 3,
    seed: int = 42,
) -> dict:
    from scipy.optimize import linear_sum_assignment

    train_beliefs = _probabilities(train_beliefs, "train_beliefs")
    train_predictions = _probabilities(train_predictions, "train_predictions")
    if len(train_beliefs) != len(train_predictions):
        raise ValueError("beliefs and predictions must contain equal samples")
    labels, codes = controls._group_data(groups, len(train_beliefs))
    seed = controls._integer(seed, "seed", 0)
    n_select = controls._integer(n_select, "n_select", 1)
    max_prediction_kl = _threshold(max_prediction_kl, "max_prediction_kl")
    min_transfer_error = _threshold(min_transfer_error, "min_transfer_error")
    fit_indices, holdout_indices = split_group_indices(codes, test_fraction=0.25, seed=seed)
    result = {
        "seed": seed,
        "selection_source": "train",
        "holdout_fraction": 0.25,
        "n_fit": len(fit_indices),
        "n_holdout": len(holdout_indices),
        "fit_groups": [labels[int(code)] for code in np.unique(codes[fit_indices])],
        "holdout_groups": [labels[int(code)] for code in np.unique(codes[holdout_indices])],
        "transfer_method": "centered_ols",
        "transfer_ridge": 0.0,
        "transfer_error_metric": "normalized_mse",
        "prediction_kl_direction": "true_to_candidate",
        "prediction_kl_source": "all_training_rows",
        "max_prediction_kl": max_prediction_kl,
        "min_transfer_error": min_transfer_error,
        "n_select": n_select,
        "ranking": "descending_heldout_normalized_transfer_mse",
        "candidates": {},
        "selected": [],
    }
    positive = train_predictions > 0.0
    numerical_tolerance = np.finfo(np.float64).eps * max(train_beliefs.shape)
    result["transfer_numerical_tolerance"] = numerical_tolerance
    result["state_permutation_tolerance"] = 1e-10
    for name, pair in _named(candidates, "candidates").items():
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each candidate must be a beliefs/predictions pair")
        beliefs = _probabilities(pair[0], f"candidates[{name!r}].beliefs")
        predictions = _probabilities(pair[1], f"candidates[{name!r}].predictions")
        if beliefs.shape != train_beliefs.shape or predictions.shape != train_predictions.shape:
            raise ValueError("candidate belief and prediction shapes must match the true targets")
        support_mismatch = bool(np.any(positive & (predictions == 0.0)))
        kl = None
        if not support_mismatch:
            kl = max(0.0, controls._finite_float(np.sum(
                train_predictions[positive]
                * (np.log(train_predictions[positive]) - np.log(predictions[positive]))
            ) / len(train_predictions)))
        weight, bias = fit_affine_probe(train_beliefs[fit_indices], beliefs[fit_indices], ridge=0.0)
        transfer = controls.score_prediction(
            train_beliefs[holdout_indices] @ weight + bias, beliefs[holdout_indices]
        )
        permutation_cost = np.stack([
            np.square(train_beliefs[:, index, None] - beliefs).mean(axis=0)
            for index in range(train_beliefs.shape[1])
        ])
        rows, columns = linear_sum_assignment(permutation_cost)
        permutation_equivalent = bool(np.allclose(
            train_beliefs[:, rows], beliefs[:, columns], rtol=0.0,
            atol=result["state_permutation_tolerance"],
        ))
        reasons = []
        if permutation_equivalent:
            reasons.append("state_permutation_equivalent")
        if support_mismatch:
            reasons.append("infinite_prediction_kl_support_mismatch")
        elif kl > max_prediction_kl:
            reasons.append("prediction_kl_exceeds_budget")
        error = transfer["normalized_mse"]
        if error is None:
            reasons.append("constant_holdout_target")
        elif error <= max(min_transfer_error, numerical_tolerance):
            reasons.append("affine_equivalent_or_below_transfer_threshold")
        result["candidates"][name] = {
            "prediction_kl": kl,
            "prediction_kl_is_infinite": support_mismatch,
            "transfer": transfer,
            "transfer_error": error,
            "state_permutation_equivalent": permutation_equivalent,
            "state_permutation_mse": float(permutation_cost[rows, columns].mean()),
            "state_permutation": columns.tolist(),
            "eligible": not reasons,
            "rejection_reasons": reasons,
        }
    eligible = [name for name, metadata in result["candidates"].items() if metadata["eligible"]]
    result["selected"] = sorted(
        eligible, key=lambda name: -result["candidates"][name]["transfer_error"]
    )[:n_select]
    return result


__all__ = [
    "ProbeBatteryResult",
    "evaluate_belief_geometry",
    "prediction_null_basis",
    "select_alternative_beliefs",
]
