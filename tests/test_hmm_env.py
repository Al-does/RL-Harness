"""Focused tests for generic finite-HMM environment validation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from envs.hmm import HMMEnv


def validation_env(n_states: int = 2) -> HMMEnv:
    env = object.__new__(HMMEnv)
    env.model = SimpleNamespace(n_states=n_states)
    return env


def test_transition_validation_preserves_allclose_boundary():
    env = validation_env()
    tolerance = 0.99 * (1e-5 + 1e-12)
    within_tolerance = np.array(
        [[0.5, 0.5 + tolerance], [0.25, 0.75 - tolerance]]
    )

    validated = env._validate_transition(within_tolerance)

    np.testing.assert_array_equal(validated, within_tolerance)


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[0.5, 0.5 + 1.01e-5], [0.25, 0.75]]),
        np.array([[1.01, -0.01], [0.25, 0.75]]),
        np.array([[np.nan, np.nan], [0.25, 0.75]]),
    ],
)
def test_transition_validation_rejects_invalid_probabilities(matrix):
    with pytest.raises(ValueError, match="row-stochastic"):
        validation_env()._validate_transition(matrix)


def test_transition_validation_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        validation_env()._validate_transition(np.ones((2, 3)))
