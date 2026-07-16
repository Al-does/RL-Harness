"""Unit tests for MESS3 model and exponential-tilt control math."""

import numpy as np
import pytest

from envs.hmm import stationary_distribution
from envs.mess3.model import (
    CONTROL_TRANSITION_MATRIX,
    PASSIVE_TRANSITION_MATRIX,
    emission_matrix,
)
from envs.mess3.tasks.occupancy_control import (
    kl_cost_per_state,
    kl_costs_batch,
    tilt_matrix,
    tilted_transition,
    tilted_transitions_batch,
)


def test_p0_stationary_distribution():
    pi = stationary_distribution(CONTROL_TRANSITION_MATRIX)
    np.testing.assert_allclose(pi, [0.45, 0.45, 0.10], atol=1e-12)


def test_zero_action_is_identity():
    w0 = np.zeros(2)
    np.testing.assert_allclose(
        tilted_transition(w0),
        CONTROL_TRANSITION_MATRIX,
        atol=1e-15,
    )
    np.testing.assert_allclose(kl_cost_per_state(w0), 0.0, atol=1e-15)


def test_rows_sum_to_one_across_box():
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = rng.uniform(-5, 5, size=2)
        U = tilted_transition(w)
        np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-12)
        assert (U > 0).all(), "tilted full-support rows must stay full-support"


def test_tilt_anchored_to_current_state():
    # w = (big, 0): each state boosts its d=+1 neighbor, not a fixed column.
    U = tilted_transition(np.array([3.0, 0.0]))
    for s in range(3):
        assert (
            U[s, (s + 1) % 3]
            > CONTROL_TRANSITION_MATRIX[s, (s + 1) % 3]
        )


def test_gauge_invariance_of_3d_parameterization():
    # Adding a constant c to ALL THREE tilts (self-loop included) leaves the
    # softmax row invariant — the flat direction the 2D gauge fixing removes.
    w = np.array([1.3, -0.7])
    for c in (0.5, -2.0, 3.1):
        T = tilt_matrix(w) + c
        G = CONTROL_TRANSITION_MATRIX * np.exp(T)
        U_shifted = G / G.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(U_shifted, tilted_transition(w), atol=1e-12)


def test_kl_cost_matches_direct_formula():
    rng = np.random.default_rng(1)
    for _ in range(50):
        w = rng.uniform(-5, 5, size=2)
        U = tilted_transition(w)
        direct = (U * np.log(U / CONTROL_TRANSITION_MATRIX)).sum(axis=1)
        np.testing.assert_allclose(kl_cost_per_state(w), direct, atol=1e-10)
        assert (kl_cost_per_state(w) >= -1e-12).all()


def test_batch_variants_match_scalar():
    rng = np.random.default_rng(2)
    W = rng.uniform(-5, 5, size=(40, 2))
    U_b = tilted_transitions_batch(W)
    kl_b = kl_costs_batch(W)
    for k, w in enumerate(W):
        np.testing.assert_allclose(U_b[k], tilted_transition(w), atol=1e-12)
        np.testing.assert_allclose(kl_b[k], kl_cost_per_state(w), atol=1e-12)


def test_emission_matrix():
    E = emission_matrix(0.85)
    np.testing.assert_allclose(E.sum(axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(np.diag(E), 0.85)
    assert E[0, 1] == pytest.approx(0.075)


def test_passive_matrix_stationary_uniform():
    pi = stationary_distribution(PASSIVE_TRANSITION_MATRIX)
    np.testing.assert_allclose(pi, 1.0 / 3.0, atol=1e-12)
