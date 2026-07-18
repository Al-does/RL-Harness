"""Predictive-belief targets for finite transducer probes."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _probability_vector(value: np.ndarray) -> np.ndarray:
    belief = np.asarray(value, dtype=np.float64)
    if belief.ndim != 1:
        raise ValueError("belief must be one-dimensional")
    if (
        not np.isfinite(belief).all()
        or (belief < 0.0).any()
        or not np.isclose(belief.sum(), 1.0)
    ):
        raise ValueError("belief must be a finite probability vector")
    return belief


def predictive_belief_update(
    belief: np.ndarray,
    action_outcome_operator: np.ndarray,
) -> np.ndarray:
    """Apply one action-conditioned transducer filtering update.

    This function uses the repository's row-vector convention. The operator
    entry ``K[i, j]`` is
    ``P(outcome, next_state=j | action, current_state=i)``. Consequently,
    this is Rosas et al.'s Eq. 8 transposed from their column-vector notation:
    ``b_next = b @ K / (b @ K @ 1)``.
    """

    prior = _probability_vector(belief)
    operator = np.asarray(action_outcome_operator, dtype=np.float64)
    expected_shape = (len(prior), len(prior))
    if operator.shape != expected_shape:
        raise ValueError(
            "action_outcome_operator must have shape "
            f"{expected_shape}, got {operator.shape}"
        )
    if not np.isfinite(operator).all() or (operator < 0.0).any():
        raise ValueError(
            "action_outcome_operator must contain finite non-negative values"
        )
    if (operator.sum(axis=1) > 1.0 + 1e-12).any():
        raise ValueError("action_outcome_operator must be substochastic")

    unnormalized = prior @ operator
    probability = float(unnormalized.sum())
    if probability <= 0.0:
        raise ValueError(
            "action-outcome update has zero probability under the belief"
        )
    return unnormalized / probability


def predictive_belief_sequence(
    initial_belief: np.ndarray,
    action_outcome_operators: Iterable[np.ndarray],
) -> np.ndarray:
    """Return beliefs before and after every operator in chronological order."""

    current = _probability_vector(initial_belief).copy()
    beliefs = [current]
    for operator in action_outcome_operators:
        current = predictive_belief_update(current, operator)
        beliefs.append(current)
    return np.stack(beliefs)


__all__ = [
    "predictive_belief_sequence",
    "predictive_belief_update",
]
