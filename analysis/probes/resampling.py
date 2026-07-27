"""Resampling controls for held-out probe evaluation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from analysis.probes.linear import mean_squared_error


def cluster_bootstrap_statistics(
    clusters: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_resamples: int = 1_000,
    seed: int = 42,
) -> np.ndarray:
    """Bootstrap a statistic by resampling whole dependent clusters.

    ``statistic`` receives sample indices. Use episode, environment-seed, or
    complete-context identifiers as clusters instead of treating correlated
    timesteps as independent observations.
    """
    clusters = np.asarray(clusters)
    if clusters.ndim != 1 or len(clusters) == 0:
        raise ValueError("clusters must be a non-empty one-dimensional array")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    unique_clusters = np.unique(clusters)
    members = [
        np.flatnonzero(clusters == cluster) for cluster in unique_clusters
    ]
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        selected = rng.integers(
            0,
            len(unique_clusters),
            size=len(unique_clusters),
        )
        sample_indices = np.concatenate([members[item] for item in selected])
        estimates[index] = float(statistic(sample_indices))
    return estimates


def percentile_interval(
    estimates: np.ndarray,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a two-sided percentile interval from finite estimates."""
    estimates = np.asarray(estimates, dtype=np.float64)
    if estimates.ndim != 1 or len(estimates) == 0:
        raise ValueError("estimates must be a non-empty one-dimensional array")
    if not np.isfinite(estimates).all():
        raise ValueError("estimates must be finite")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    return float(low), float(high)


def held_out_permutation_null(
    train_targets: np.ndarray,
    fit_predict: Callable[[np.ndarray], np.ndarray],
    test_targets: np.ndarray,
    *,
    score: Callable[[np.ndarray, np.ndarray], float] = mean_squared_error,
    n_permutations: int = 1_000,
    seed: int = 42,
) -> np.ndarray:
    """Refit on permuted train labels and score against true held-out labels.

    ``fit_predict`` receives one row-permuted copy of ``train_targets`` and
    returns predictions for the fixed held-out features. The resulting scores
    form a no-association null distribution; they do not improve the real
    probe.
    """
    train_targets = np.asarray(train_targets)
    test_targets = np.asarray(test_targets)
    if train_targets.ndim != 2 or len(train_targets) == 0:
        raise ValueError("train_targets must be a non-empty matrix")
    if test_targets.ndim != 2 or len(test_targets) == 0:
        raise ValueError("test_targets must be a non-empty matrix")
    if train_targets.shape[1] != test_targets.shape[1]:
        raise ValueError("train and test targets must have equal widths")
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")

    rng = np.random.default_rng(seed)
    scores = np.empty(n_permutations, dtype=np.float64)
    for index in range(n_permutations):
        permuted = train_targets[rng.permutation(len(train_targets))]
        predicted = np.asarray(fit_predict(permuted))
        scores[index] = float(score(predicted, test_targets))
    return scores
