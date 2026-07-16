"""Exact Bayesian belief updates for finite discrete HMMs."""

from __future__ import annotations

import numpy as np


def measure(
    belief: np.ndarray,
    likelihood: np.ndarray,
    observation: int,
) -> np.ndarray:
    """Condition a state belief on one categorical observation."""

    prior = np.asarray(belief, dtype=np.float64)
    probabilities = np.asarray(likelihood, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != prior.shape[0]:
        raise ValueError("likelihood must have shape (n_states, n_observations)")
    if not 0 <= observation < probabilities.shape[1]:
        raise ValueError("observation index is outside the likelihood")

    posterior = prior * probabilities[:, observation]
    total = posterior.sum()
    if total <= 0.0:
        raise ValueError("measurement update produced zero probability mass")
    return posterior / total


def predict(belief: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
    """Push a row-vector belief through one transition matrix."""

    prior = np.asarray(belief, dtype=np.float64)
    matrix = np.asarray(transition_matrix, dtype=np.float64)
    if matrix.shape != (prior.shape[0], prior.shape[0]):
        raise ValueError("transition_matrix has the wrong shape for this belief")
    return prior @ matrix


def advance_belief(
    belief: np.ndarray,
    observation: int,
    likelihood: np.ndarray,
    transition_matrix: np.ndarray,
    *,
    delay: int,
) -> np.ndarray:
    """Advance a decision-time belief using the configured token timing."""

    if delay == 0:
        return measure(
            predict(belief, transition_matrix),
            likelihood,
            observation,
        )
    if delay == 1:
        return predict(
            measure(belief, likelihood, observation),
            transition_matrix,
        )
    raise ValueError("exact belief updates support delay 0 or 1")


class BeliefTracker:
    """Small stateful wrapper around the pure belief operations."""

    def __init__(self, initial_distribution: np.ndarray) -> None:
        initial = np.array(initial_distribution, dtype=np.float64, copy=True)
        if initial.ndim != 1 or (initial < 0.0).any():
            raise ValueError("initial belief must be a probability vector")
        if not np.isclose(initial.sum(), 1.0):
            raise ValueError("initial belief must sum to one")
        initial.setflags(write=False)
        self.initial_distribution = initial
        self.belief = initial.copy()

    def reset(
        self,
        observation: int | None = None,
        *,
        likelihood: np.ndarray | None = None,
    ) -> np.ndarray:
        self.belief = self.initial_distribution.copy()
        if observation is not None:
            if likelihood is None:
                raise ValueError("likelihood is required for a reset observation")
            self.belief = measure(self.belief, likelihood, observation)
        return self.belief

    def measure(
        self,
        observation: int,
        likelihood: np.ndarray,
    ) -> np.ndarray:
        self.belief = measure(self.belief, likelihood, observation)
        return self.belief

    def predict(self, transition_matrix: np.ndarray) -> np.ndarray:
        self.belief = predict(self.belief, transition_matrix)
        return self.belief

    def advance(
        self,
        observation: int,
        likelihood: np.ndarray,
        transition_matrix: np.ndarray,
        *,
        delay: int,
    ) -> np.ndarray:
        self.belief = advance_belief(
            self.belief,
            observation,
            likelihood,
            transition_matrix,
            delay=delay,
        )
        return self.belief
