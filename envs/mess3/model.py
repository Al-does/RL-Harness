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


def symmetric_transition_matrix(stay: float = 0.90) -> np.ndarray:
    """Return the symmetric three-state chain with the given self-transition."""

    if not 0.0 <= stay <= 1.0:
        raise ValueError("stay must lie in [0, 1]")
    matrix = np.full((N_STATES, N_STATES), (1.0 - stay) / (N_STATES - 1))
    np.fill_diagonal(matrix, stay)
    return matrix / matrix.sum(axis=1, keepdims=True)


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


def symmetric_model(
    *,
    stay: float = 0.90,
    alpha: float = 0.85,
    initial_distribution: np.ndarray | None = None,
) -> HMMModel:
    """Symmetric MESS3 with chain memory and channel fidelity both exposed.

    ``stay`` sets how long the chain remembers; ``alpha`` sets how much a single
    token reveals. Their ordering decides whether prediction tasks over this
    process are non-trivial. One observation multiplies its own state's belief
    coordinate by ``2*alpha/(1-alpha)``, while one transition step caps the ratio
    between any two coordinates at ``(1+2*L)/(1-L)`` for ``L = (3*stay-1)/2``.
    While ``stay <= alpha`` the second can never exceed the first, so the
    Bayes-optimal next-token guess is the last observed token at every step and
    belief tracking buys nothing.

    ``passive_model`` is this family at ``stay=0.90``.
    """

    transition_matrix = symmetric_transition_matrix(stay)
    if initial_distribution is None:
        initial_distribution = stationary_distribution(transition_matrix)
    return HMMModel(
        initial_distribution=initial_distribution,
        transition_matrix=transition_matrix,
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
