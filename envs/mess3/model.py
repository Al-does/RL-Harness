"""MESS3 probability models, independent of task and reward semantics."""

from __future__ import annotations

import numpy as np

from envs.hmm import HMMModel, stationary_distribution

N_STATES = 3
N_TOKENS = 3

CONTROL_TRANSITION_MATRIX = np.array(
    [
        [0.75, 0.15, 0.10],
        [0.15, 0.75, 0.10],
        [0.45, 0.45, 0.10],
    ],
    dtype=np.float64,
)
CONTROL_TRANSITION_MATRIX.setflags(write=False)

PASSIVE_TRANSITION_MATRIX = np.array(
    [
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
    ],
    dtype=np.float64,
)
PASSIVE_TRANSITION_MATRIX.setflags(write=False)

STATE_LABELS = ("state_0", "state_1", "state_2")
TOKEN_LABELS = ("token_0", "token_1", "token_2")


def emission_matrix(alpha: float = 0.85) -> np.ndarray:
    """Return the symmetric three-token MESS3 emission channel."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    off_diagonal = (1.0 - alpha) / 2.0
    emission = np.full((N_STATES, N_TOKENS), off_diagonal)
    np.fill_diagonal(emission, alpha)
    return emission


def control_model(
    *,
    alpha: float = 0.85,
    initial_distribution: np.ndarray | None = None,
) -> HMMModel:
    """MESS3 control model whose zero-action dynamics use ``CONTROL_TRANSITION_MATRIX``."""

    if initial_distribution is None:
        initial_distribution = np.full(N_STATES, 1.0 / N_STATES)
    return HMMModel(
        initial_distribution=initial_distribution,
        transition_matrix=CONTROL_TRANSITION_MATRIX,
        emission_matrix=emission_matrix(alpha),
        state_labels=STATE_LABELS,
        token_labels=TOKEN_LABELS,
    )


def passive_model(
    *,
    alpha: float = 0.85,
    initial_distribution: np.ndarray | None = None,
) -> HMMModel:
    """Canonical symmetric passive MESS3 model."""

    if initial_distribution is None:
        initial_distribution = stationary_distribution(PASSIVE_TRANSITION_MATRIX)
    return HMMModel(
        initial_distribution=initial_distribution,
        transition_matrix=PASSIVE_TRANSITION_MATRIX,
        emission_matrix=emission_matrix(alpha),
        state_labels=STATE_LABELS,
        token_labels=TOKEN_LABELS,
    )


def state_guess_model(
    *,
    alpha: float = 0.85,
    initial_distribution: np.ndarray | None = None,
) -> HMMModel:
    """Passive ``CONTROL_TRANSITION_MATRIX`` model for state-estimation tasks."""

    if initial_distribution is None:
        initial_distribution = stationary_distribution(CONTROL_TRANSITION_MATRIX)
    return HMMModel(
        initial_distribution=initial_distribution,
        transition_matrix=CONTROL_TRANSITION_MATRIX,
        emission_matrix=emission_matrix(alpha),
        state_labels=STATE_LABELS,
        token_labels=TOKEN_LABELS,
    )
