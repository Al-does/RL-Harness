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


def fit_affine_probe(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``features @ weight + bias`` by ridge-regularized least squares."""
    features = np.asarray(features)
    targets = np.asarray(targets)
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("features and targets must both be two-dimensional")
    if len(features) != len(targets):
        raise ValueError("features and targets must contain equal samples")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    augmented = np.concatenate(
        [features, np.ones((features.shape[0], 1))],
        axis=1,
    )
    system = augmented.T @ augmented
    system += ridge * np.eye(system.shape[0])
    coefficient = np.linalg.solve(system, augmented.T @ targets)
    return coefficient[:-1], coefficient[-1]


def probe_predict(
    weight: np.ndarray,
    bias: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    return np.asarray(features) @ weight + bias


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
