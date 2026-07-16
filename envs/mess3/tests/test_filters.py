"""Belief correctness, observable operators, and empirical calibration."""

from __future__ import annotations

import numpy as np
import pytest

from envs.hmm import BeliefTracker, HMMEnv, measure, predict
from envs.mess3.model import (
    N_STATES,
    PASSIVE_TRANSITION_MATRIX,
    emission_matrix,
)
from envs.mess3.tasks.occupancy_control import tilted_transition


BELIEF_DIAGNOSTICS = {
    "state": True,
    "belief": True,
    "tokens": True,
}


def control_env(*, alpha: float = 0.85, delay: int = 1) -> HMMEnv:
    return HMMEnv(
        {
            "model": {
                "factory": "envs.mess3.model:control_model",
                "kwargs": {"alpha": alpha},
            },
            "task": {
                "class": (
                    "envs.mess3.tasks.occupancy_control:"
                    "OccupancyControlTask"
                ),
            },
            "delay": delay,
            "diagnostics": BELIEF_DIAGNOSTICS,
        }
    )


def observable_operator(
    likelihood: np.ndarray,
    transition_matrix: np.ndarray,
    observation: int,
) -> np.ndarray:
    """Unnormalized delay-one update operator for a row belief."""

    return np.diag(likelihood[:, observation]) @ transition_matrix


def test_belief_collapses_to_true_state_with_sharp_delay_zero_tokens():
    env = control_env(alpha=1.0, delay=0)
    _, info = env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(300):
        belief = info["belief_current"]
        state = info["state_current"]
        assert belief[state] == pytest.approx(1.0, abs=1e-12)
        _, _, _, _, info = env.step(rng.uniform(-2, 2, size=2))


def test_passive_environment_matches_observable_operator_products():
    env = HMMEnv(
        {
            "model": {"factory": "envs.mess3.model:passive_model"},
            "task": {"class": "envs.mess3.tasks.passive:PassiveTask"},
            "delay": 1,
            "diagnostics": BELIEF_DIAGNOSTICS,
        }
    )
    _, info = env.reset(seed=7)
    likelihood = emission_matrix(0.85)
    reference_belief = np.full(N_STATES, 1.0 / N_STATES)
    rng = np.random.default_rng(1)
    for _ in range(500):
        _, _, _, _, info = env.step(rng.uniform(-1, 1, size=2))
        token = info["visible_source_token"]
        reference_belief = reference_belief @ observable_operator(
            likelihood,
            PASSIVE_TRANSITION_MATRIX,
            token,
        )
        reference_belief /= reference_belief.sum()
        np.testing.assert_allclose(
            info["belief_current"],
            reference_belief,
            atol=1e-10,
        )


def assert_empirically_calibrated(
    *,
    delay: int,
    n_steps: int,
    reset_seed: int,
    action_seed: int,
) -> None:
    env = control_env(delay=delay)
    _, info = env.reset(seed=reset_seed)
    rng = np.random.default_rng(action_seed)
    probabilities = np.empty(n_steps)
    hits = np.empty(n_steps)
    for step in range(n_steps):
        probabilities[step] = info["belief_current"][2]
        hits[step] = float(info["state_current"] == 2)
        _, _, _, truncated, info = env.step(
            rng.uniform(-5, 5, size=2)
        )
        if truncated:
            _, info = env.reset()

    edges = np.quantile(probabilities, np.linspace(0, 1, 11))
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (
            (probabilities >= lower)
            & (probabilities <= upper)
        )
        if selected.sum() < 500:
            continue
        predicted = probabilities[selected].mean()
        observed = hits[selected].mean()
        assert abs(predicted - observed) < 0.02, (
            f"calibration bucket [{lower:.3f}, {upper:.3f}] off: "
            f"pred {predicted:.4f} vs freq {observed:.4f}"
        )


def test_belief_is_empirically_calibrated_delay_one():
    assert_empirically_calibrated(
        delay=1,
        n_steps=200_000,
        reset_seed=11,
        action_seed=2,
    )


def test_belief_is_empirically_calibrated_delay_zero():
    assert_empirically_calibrated(
        delay=0,
        n_steps=100_000,
        reset_seed=13,
        action_seed=4,
    )


def test_measure_and_predict_match_direct_updates():
    likelihood = emission_matrix(0.85)
    belief = np.array([0.2, 0.5, 0.3])
    posterior = measure(belief, likelihood, 1)
    assert posterior[1] > belief[1]
    assert posterior.sum() == pytest.approx(1.0)

    transition = tilted_transition(np.array([1.0, -1.0]))
    np.testing.assert_allclose(
        predict(belief, transition),
        belief @ transition,
        atol=1e-15,
    )


def test_belief_tracker_reset_restores_initial_distribution():
    tracker = BeliefTracker(np.array([0.45, 0.45, 0.10]))
    np.testing.assert_allclose(
        tracker.reset(),
        [0.45, 0.45, 0.10],
    )


def test_reset_measurement_requires_likelihood():
    tracker = BeliefTracker(np.full(3, 1 / 3))
    with pytest.raises(ValueError, match="likelihood"):
        tracker.reset(observation=2)
    belief = tracker.reset(
        observation=2,
        likelihood=emission_matrix(0.85),
    )
    assert np.argmax(belief) == 2
