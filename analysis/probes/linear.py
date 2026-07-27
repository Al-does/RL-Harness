"""Domain-agnostic affine probe fitting, splits, and metrics."""

from __future__ import annotations

import numpy as np


def split_indices(
    n_samples: int,
    *,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic shuffled train and test indices."""
    if n_samples < 2:
        raise ValueError("at least two samples are required")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_samples)
    test_size = min(n_samples - 1, max(1, round(n_samples * test_fraction)))
    return shuffled[test_size:], shuffled[:test_size]


def split_group_indices(
    groups: np.ndarray,
    *,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices while keeping every group wholly in one partition."""
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("at least two groups are required")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    test_size = min(
        len(unique_groups) - 1,
        max(1, round(len(unique_groups) * test_fraction)),
    )
    test_groups = shuffled[:test_size]
    test_mask = np.isin(groups, test_groups)
    indices = np.arange(len(groups))
    return indices[~test_mask], indices[test_mask]


def fit_affine_probe(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``features @ weight + bias`` by stable ridge least squares.

    The augmented least-squares system is solved by NumPy's SVD-backed
    ``lstsq`` rather than by forming the normal equations. Ridge regularizes
    feature weights but leaves the affine intercept unpenalized.
    """
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("features and targets must both be two-dimensional")
    if len(features) != len(targets):
        raise ValueError("features and targets must contain equal samples")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    feature_mean = features.mean(axis=0)
    target_mean = targets.mean(axis=0)
    design = features - feature_mean
    responses = targets - target_mean
    if ridge > 0.0:
        design = np.concatenate(
            [
                design,
                np.sqrt(ridge) * np.eye(features.shape[1]),
            ],
            axis=0,
        )
        responses = np.concatenate(
            [
                responses,
                np.zeros((features.shape[1], targets.shape[1])),
            ],
            axis=0,
        )
    weight, _, _, _ = np.linalg.lstsq(
        design,
        responses,
        rcond=None,
    )
    bias = target_mean - feature_mean @ weight
    return weight, bias


def probe_predict(
    weight: np.ndarray,
    bias: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    return np.asarray(features) @ weight + bias


def mean_squared_error(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    """Return mean squared probe error over every sample and target coordinate."""
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    if predicted.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    return float(np.square(predicted - target).mean())


def target_variance(target: np.ndarray) -> float:
    """Return the MSE of predicting each target coordinate by its global mean."""
    target = np.asarray(target)
    if target.size == 0:
        raise ValueError("target must not be empty")
    return float(np.square(target - target.mean(axis=0)).mean())


def global_mse_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Report absolute and baseline-normalized global probe error.

    ``global_mse_ratio`` is one minus global R². Keeping the absolute MSE and
    target variance makes the normalization reconstructible while retaining
    the units used by the target.
    """
    mse = mean_squared_error(predicted, target)
    variance = target_variance(target)
    return {
        "mse": mse,
        "target_variance": variance,
        "global_mse_ratio": (
            float("nan") if variance == 0.0 else mse / variance
        ),
    }


def conditional_mse_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    min_group_size: int = 1,
) -> dict[str, float | int]:
    """Compare probe MSE with a branch-centroid baseline.

    The branch baseline predicts each target by the empirical target centroid
    for its group. ``fine_mse_ratio`` is one minus conditional residual R²:
    zero is perfect, one matches the branch baseline, and values above one are
    worse than that baseline.

    Subtracting the same branch centroid from prediction and target does not
    change their difference. A literal residualized "fine MSE" is therefore
    the ordinary MSE on the retained groups; the baseline, ratio, and
    improvement provide the fine-grained interpretation.
    """
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    groups = np.asarray(groups)
    if predicted.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if groups.shape != (len(target),):
        raise ValueError("groups must contain one label per sample")
    if min_group_size <= 0:
        raise ValueError("min_group_size must be positive")

    target_residual = np.empty_like(target)
    keep = np.zeros(len(target), dtype=bool)
    for group in np.unique(groups):
        members = groups == group
        if int(members.sum()) < min_group_size:
            continue
        centroid = target[members].mean(axis=0)
        target_residual[members] = target[members] - centroid
        keep[members] = True
    if not keep.any():
        return {
            "fine_evaluation_mse": float("nan"),
            "branch_baseline_mse": float("nan"),
            "fine_mse_ratio": float("nan"),
            "fine_mse_improvement": float("nan"),
            "n_evaluated": 0,
        }

    mse = mean_squared_error(predicted[keep], target[keep])
    baseline = float(np.square(target_residual[keep]).mean())
    return {
        "fine_evaluation_mse": mse,
        "branch_baseline_mse": baseline,
        "fine_mse_ratio": (
            float("nan") if baseline == 0.0 else mse / baseline
        ),
        "fine_mse_improvement": baseline - mse,
        "n_evaluated": int(keep.sum()),
    }


def r2_score(predicted: np.ndarray, target: np.ndarray) -> float:
    """Global multivariate coefficient of determination."""
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    if predicted.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    residual = float(np.square(predicted - target).sum())
    total = float(np.square(target - target.mean(axis=0)).sum())
    return float("nan") if total == 0.0 else 1.0 - residual / total


def conditional_residual_r2(
    predicted: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    min_group_size: int = 1,
) -> float:
    """R² after subtracting each target group's centroid from both arrays."""
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    groups = np.asarray(groups)
    if predicted.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if groups.shape != (len(target),):
        raise ValueError("groups must contain one label per sample")
    if min_group_size <= 0:
        raise ValueError("min_group_size must be positive")

    predicted_residual = np.empty_like(predicted)
    target_residual = np.empty_like(target)
    keep = np.zeros(len(target), dtype=bool)
    for group in np.unique(groups):
        members = groups == group
        if int(members.sum()) < min_group_size:
            continue
        centroid = target[members].mean(axis=0)
        predicted_residual[members] = predicted[members] - centroid
        target_residual[members] = target[members] - centroid
        keep[members] = True
    if not keep.any():
        return float("nan")
    residual = float(
        np.square(predicted_residual[keep] - target_residual[keep]).sum()
    )
    total = float(np.square(target_residual[keep]).sum())
    return float("nan") if total == 0.0 else 1.0 - residual / total
