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
    "filter_operator_histories",
]


def filter_operator_histories(
    initial_belief: np.ndarray,
    operators: np.ndarray,
    groups: np.ndarray,
    steps: np.ndarray,
    *,
    suffix_length: int | None = None,
) -> np.ndarray:
    from analysis.probes.controls import _group_data, _integer

    if np.iscomplexobj(initial_belief) or np.iscomplexobj(operators):
        raise ValueError("belief and operators must be real")
    prior = _probability_vector(initial_belief)
    operators = np.asarray(operators, dtype=np.float64)
    if operators.ndim != 3 or operators.shape[1:] != (len(prior), len(prior)):
        raise ValueError("operators must have shape (n_samples, n_states, n_states)")
    if not np.isfinite(operators).all() or (operators < 0.0).any():
        raise ValueError("operators must contain finite non-negative values")
    if (operators.sum(axis=2) > 1.0 + 1e-12).any():
        raise ValueError("operators must be substochastic")
    steps = np.asarray(steps)
    if steps.shape != (len(operators),) or steps.dtype.kind not in "iu" or (steps < 0).any():
        raise ValueError("steps must be aligned nonnegative integers")
    labels, codes = _group_data(groups, len(operators))
    if suffix_length is not None:
        suffix_length = _integer(suffix_length, "suffix_length", 0)
    previous = np.full(len(operators), -1, dtype=np.int64)
    last = np.full(len(labels), -1, dtype=np.int64)
    next_steps = [0] * len(labels)
    for index, (group, step) in enumerate(zip(codes, steps)):
        if int(step) != next_steps[group]:
            raise ValueError("each group must start at step zero and have consecutive increasing steps")
        next_steps[group] += 1
        previous[index] = last[group]
        last[group] = index
    beliefs = np.broadcast_to(prior, (len(operators), len(prior))).copy()
    if not len(operators) or suffix_length == 0:
        return beliefs
    if suffix_length is None or suffix_length >= max(next_steps):
        current = np.broadcast_to(prior, (len(labels), len(prior))).copy()
        for index, group in enumerate(codes):
            updated = current[group] @ operators[index]
            mass = updated.sum()
            if mass <= 0.0:
                raise ValueError("operator history has zero probability under the belief")
            current[group] = updated / mass
            beliefs[index] = current[group]
        return beliefs
    history = [np.arange(len(operators))]
    for _ in range(suffix_length - 1):
        indices = history[-1]
        history.append(np.where(indices >= 0, previous[np.maximum(indices, 0)], -1))
    for indices in reversed(history):
        active = indices >= 0
        updated = np.einsum("ni,nij->nj", beliefs[active], operators[indices[active]])
        mass = updated.sum(axis=1)
        if (mass <= 0.0).any():
            raise ValueError("operator history has zero probability under the belief")
        beliefs[active] = updated / mass[:, None]
    return beliefs
