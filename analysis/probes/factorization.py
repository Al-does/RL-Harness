"""Geometry diagnostics for joint versus factored representations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np

from analysis.probes.linear import fit_affine_probe


def _matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2:
        raise ValueError(f"{name} must have shape (n_samples, width) with n_samples >= 2")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _variance_fraction(value: float) -> float:
    fraction = float(value)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("variance fraction must lie in (0, 1]")
    return fraction


def variance_geometry(
    values: np.ndarray,
    *,
    thresholds: Sequence[float] = (0.90, 0.95, 0.99),
    max_spectrum_entries: int | None = None,
) -> dict[str, Any]:
    """Return PCA cumulative explained variance (CEV) diagnostics.

    CEV is the paper's *cumulative explained variance*. Effective dimension at
    threshold ``p`` is the smallest number of principal components whose CEV
    reaches ``p``.
    """

    matrix = _matrix(values, name="values")
    requested = tuple(_variance_fraction(value) for value in thresholds)
    if not requested:
        raise ValueError("at least one CEV threshold is required")
    if max_spectrum_entries is not None and max_spectrum_entries <= 0:
        raise ValueError("max_spectrum_entries must be positive when supplied")

    centered = matrix - matrix.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    spectrum = singular_values**2
    total = float(spectrum.sum())
    tolerance = (
        0.0
        if not len(singular_values)
        else np.finfo(np.float64).eps
        * max(centered.shape)
        * float(singular_values[0])
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    if total <= 0.0:
        fractions = np.zeros_like(spectrum)
        cumulative = np.zeros_like(spectrum)
        dimensions = {str(threshold): 0 for threshold in requested}
        participation_ratio = 0.0
    else:
        fractions = spectrum / total
        cumulative = np.cumsum(fractions)
        cumulative[-1] = 1.0
        dimensions = {
            str(threshold): min(
                len(cumulative),
                int(np.searchsorted(cumulative, threshold) + 1),
            )
            for threshold in requested
        }
        participation_ratio = float(
            np.square(spectrum.sum()) / np.square(spectrum).sum()
        )

    count = (
        len(fractions)
        if max_spectrum_entries is None
        else min(max_spectrum_entries, len(fractions))
    )
    report: dict[str, Any] = {
        "rank": rank,
        "effective_dimensions": dimensions,
        "participation_ratio": participation_ratio,
        "explained_variance_fraction": fractions[:count].tolist(),
        "cumulative_explained_variance": cumulative[:count].tolist(),
    }
    for threshold, dimension in dimensions.items():
        percentage = float(threshold) * 100.0
        if percentage.is_integer():
            report[f"cev{int(percentage)}_dimension"] = dimension
    return report


def effective_dimension(
    values: np.ndarray,
    *,
    variance_fraction: float = 0.95,
) -> int:
    """Return the number of principal components needed for the requested CEV."""

    fraction = _variance_fraction(variance_fraction)
    return int(
        variance_geometry(values, thresholds=(fraction,))[
            "effective_dimensions"
        ][str(fraction)]
    )


def principal_component_basis(
    values: np.ndarray,
    *,
    variance_fraction: float = 0.95,
    max_rank: int | None = None,
) -> np.ndarray:
    """Return dominant orthonormal feature directions of a point cloud."""

    matrix = _matrix(values, name="values")
    centered = matrix - matrix.mean(axis=0)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    if not len(singular_values) or float(np.square(singular_values).sum()) == 0.0:
        return np.empty((matrix.shape[1], 0), dtype=np.float64)
    rank = effective_dimension(matrix, variance_fraction=variance_fraction)
    if max_rank is not None:
        if max_rank <= 0:
            raise ValueError("max_rank must be positive when supplied")
        rank = min(rank, int(max_rank))
    return right[:rank].T


def orthonormal_basis(
    directions: np.ndarray,
    *,
    rank: int | None = None,
    relative_tolerance: float = 1e-10,
) -> np.ndarray:
    """Orthonormalize the column span of feature-space directions."""

    matrix = np.asarray(directions, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("directions must be two-dimensional")
    if not np.isfinite(matrix).all():
        raise ValueError("directions must contain only finite values")
    if relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be non-negative")
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if not len(singular_values) or singular_values[0] == 0.0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    numerical_rank = int(
        np.count_nonzero(
            singular_values > relative_tolerance * singular_values[0]
        )
    )
    if rank is not None:
        if rank < 0:
            raise ValueError("rank must be non-negative")
        numerical_rank = min(numerical_rank, int(rank))
    return left[:, :numerical_rank]


def readout_subspace(
    weight: np.ndarray,
    *,
    rank: int | None = None,
    relative_tolerance: float = 1e-10,
) -> np.ndarray:
    """Return the activation-space column span of a linear probe's weights."""

    return orthonormal_basis(
        weight,
        rank=rank,
        relative_tolerance=relative_tolerance,
    )


def subspace_overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Return normalized squared principal-angle overlap in ``[0, 1]``.

    Zero means orthogonal subspaces. One means the smaller subspace is wholly
    contained in the larger. Inputs need not already be orthonormal.
    """

    left_basis = orthonormal_basis(left)
    right_basis = orthonormal_basis(right)
    denominator = min(left_basis.shape[1], right_basis.shape[1])
    if denominator == 0:
        return float("nan")
    interaction = left_basis.T @ right_basis
    return float(np.square(interaction).sum() / denominator)


def pairwise_subspace_overlaps(
    subspaces: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Return principal-angle overlap for every named subspace pair."""

    return {
        f"{left}_vs_{right}": subspace_overlap(
            subspaces[left],
            subspaces[right],
        )
        for left, right in combinations(subspaces, 2)
    }


def regression_factor_subspaces(
    activations: np.ndarray,
    factor_targets: Mapping[str, np.ndarray],
    *,
    ridge: float = 1e-6,
    target_ranks: Mapping[str, int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Identify factor subspaces with one joint affine regression.

    Targets are concatenated for one fit, matching the paper's regression
    procedure. Each returned basis spans activation directions carrying the
    corresponding factor's predictive vector.
    """

    features = _matrix(activations, name="activations")
    if not factor_targets:
        raise ValueError("factor_targets must not be empty")
    names = tuple(factor_targets)
    targets = {
        name: _matrix(factor_targets[name], name=f"factor target {name!r}")
        for name in names
    }
    if any(len(target) != len(features) for target in targets.values()):
        raise ValueError("all factor targets must align with activations")

    widths = {name: targets[name].shape[1] for name in names}
    combined = np.concatenate([targets[name] for name in names], axis=1)
    weight, _ = fit_affine_probe(features, combined, ridge=ridge)
    bases: dict[str, np.ndarray] = {}
    ranks: dict[str, int] = {}
    offset = 0
    for name in names:
        width = widths[name]
        inferred = int(
            np.linalg.matrix_rank(
                targets[name] - targets[name].mean(axis=0),
            )
        )
        rank = (
            inferred
            if target_ranks is None or name not in target_ranks
            else int(target_ranks[name])
        )
        if not 0 <= rank <= width:
            raise ValueError(f"target rank for {name!r} must lie in [0, {width}]")
        ranks[name] = rank
        bases[name] = readout_subspace(
            weight[:, offset : offset + width],
            rank=rank,
        )
        offset += width
    return bases, ranks


def regression_factor_geometry(
    activations: np.ndarray,
    factor_targets: Mapping[str, np.ndarray],
    *,
    ridge: float = 1e-6,
    target_ranks: Mapping[str, int] | None = None,
    max_spectrum_entries: int | None = 32,
) -> dict[str, Any]:
    """Summarize CEV, factor readout dimensions, and subspace orthogonality."""

    bases, ranks = regression_factor_subspaces(
        activations,
        factor_targets,
        ridge=ridge,
        target_ranks=target_ranks,
    )
    overlaps = pairwise_subspace_overlaps(bases)
    finite_overlaps = [
        value for value in overlaps.values() if np.isfinite(value)
    ]
    union = np.concatenate(tuple(bases.values()), axis=1)
    return {
        "activation_pca": variance_geometry(
            activations,
            max_spectrum_entries=max_spectrum_entries,
        ),
        "factor_target_ranks": ranks,
        "factor_subspace_dimensions": {
            name: int(basis.shape[1]) for name, basis in bases.items()
        },
        "pairwise_subspace_overlap": overlaps,
        "mean_pairwise_subspace_overlap": (
            float(np.mean(finite_overlaps))
            if finite_overlaps
            else float("nan")
        ),
        "union_rank": int(np.linalg.matrix_rank(union)),
        "sum_factor_subspace_dimensions": int(
            sum(basis.shape[1] for basis in bases.values())
        ),
    }


def center_within_groups(
    values: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Mean-center samples within each fixed-context group for vary-one PCA."""

    matrix = _matrix(values, name="values")
    labels = np.asarray(groups)
    if labels.shape != (len(matrix),):
        raise ValueError("groups must contain one label per sample")
    centered = np.empty_like(matrix)
    for group in np.unique(labels):
        members = labels == group
        centered[members] = matrix[members] - matrix[members].mean(axis=0)
    return centered


def vary_one_subspace(
    activations: np.ndarray,
    fixed_context_groups: np.ndarray,
    *,
    variance_fraction: float = 0.95,
    max_rank: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Identify one factor's subspace from a controlled vary-one dataset."""

    centered = center_within_groups(activations, fixed_context_groups)
    basis = principal_component_basis(
        centered,
        variance_fraction=variance_fraction,
        max_rank=max_rank,
    )
    return basis, variance_geometry(centered)


def dimension_additivity(
    factor_activations: Mapping[str, np.ndarray],
    *,
    variance_fraction: float = 0.95,
) -> dict[str, Any]:
    """Compare factor-specific effective dimensions with their pooled union.

    A positive excess indicates shared linear directions. Zero excess does not
    by itself prove orthogonality: non-orthogonal subspaces can have a trivial
    intersection. Use :func:`subspace_overlap` for the principal-angle test.
    """

    if not factor_activations:
        raise ValueError("factor_activations must not be empty")
    matrices = {
        name: _matrix(values, name=f"factor activations {name!r}")
        for name, values in factor_activations.items()
    }
    matrices = {
        name: matrix - matrix.mean(axis=0)
        for name, matrix in matrices.items()
    }
    widths = {matrix.shape[1] for matrix in matrices.values()}
    if len(widths) != 1:
        raise ValueError("all factor activations must have the same width")
    dimensions = {
        name: effective_dimension(
            matrix,
            variance_fraction=variance_fraction,
        )
        for name, matrix in matrices.items()
    }
    union_dimension = effective_dimension(
        np.concatenate(tuple(matrices.values()), axis=0),
        variance_fraction=variance_fraction,
    )
    summed = int(sum(dimensions.values()))
    return {
        "variance_fraction": float(variance_fraction),
        "factor_effective_dimensions": dimensions,
        "sum_factor_effective_dimensions": summed,
        "union_effective_dimension": union_dimension,
        "dimension_excess": summed - union_dimension,
    }


def representation_dimension_predictions(
    factor_dimensions: Sequence[int],
) -> dict[str, int]:
    """Return direct-sum and full-joint normalized-state dimensions."""

    dimensions = tuple(int(value) for value in factor_dimensions)
    if not dimensions or any(value <= 0 for value in dimensions):
        raise ValueError("factor_dimensions must contain positive values")
    return {
        "factored": int(sum(value - 1 for value in dimensions)),
        "joint": int(np.prod(dimensions) - 1),
    }
