"""Filter correctness: convention pins, one-hot collapse, passive-mode
observable-operator equivalence, and an empirical calibration check
(the filter must be the true posterior of the simulated process)."""

import numpy as np
import pytest

from envs.mess3.core import (
    MESS3_PASSIVE_M,
    N_STATES,
    N_TOKENS,
    P0,
    emission_matrix,
    stationary_distribution,
    tilted_transition,
)
from envs.mess3.env_continuous import Mess3ContinuousConfig, Mess3ContinuousEnv
from envs.mess3.filters import ExactFilter, measure, observable_operator, predict


def test_filter_collapses_to_one_hot_when_alpha_1_delay_0():
    # Required sanity from the spec: alpha=1.0, delay=0 -> belief is one-hot
    # on the true state at every decision time.
    env = Mess3ContinuousEnv(Mess3ContinuousConfig(alpha=1.0, delay=0, seed=3))
    obs, info = env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(300):
        b, s = info["belief"], info["state"]
        assert b[s] == pytest.approx(1.0, abs=1e-12)
        obs, *_ , info = env.step(rng.uniform(-2, 2, size=2))


def test_passive_mode_matches_observable_operators():
    # Passive-mode filter == normalized b @ T(o) products, T(o) = diag(E[:,o]) @ M.
    env = Mess3ContinuousEnv(Mess3ContinuousConfig(passive_mode=True, delay=1, seed=7))
    obs, info = env.reset(seed=7)
    E = emission_matrix(0.85)
    b_ref = np.full(N_STATES, 1.0 / N_STATES)
    rng = np.random.default_rng(1)
    for _ in range(500):
        obs, *_, info = env.step(rng.uniform(-1, 1, size=2))
        tok = info["obs_token"]  # the token the filter just consumed
        b_ref = b_ref @ observable_operator(E, MESS3_PASSIVE_M, tok)
        b_ref = b_ref / b_ref.sum()
        np.testing.assert_allclose(info["belief"], b_ref, atol=1e-10)


def test_filter_is_calibrated_delay1():
    # Empirical calibration: bucket steps by belief-in-state-2 and check the
    # realized frequency of s == 2 matches the belief (the defining property
    # of the exact posterior). Actions randomized to exercise conditioning
    # on the continuous executed action.
    env = Mess3ContinuousEnv(Mess3ContinuousConfig(beta=4.0, delay=1, seed=11))
    obs, info = env.reset(seed=11)
    rng = np.random.default_rng(2)
    n = 200_000
    probs = np.empty(n)
    hits = np.empty(n)
    for t in range(n):
        probs[t] = info["belief"][2]
        hits[t] = float(info["state"] == 2)
        obs, _, _, truncated, info = env.step(rng.uniform(-5, 5, size=2))
        if truncated:
            obs, info = env.reset()
    edges = np.quantile(probs, np.linspace(0, 1, 11))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs <= hi)
        if m.sum() < 500:
            continue
        assert abs(probs[m].mean() - hits[m].mean()) < 0.02, (
            f"calibration bucket [{lo:.3f}, {hi:.3f}] off: "
            f"pred {probs[m].mean():.4f} vs freq {hits[m].mean():.4f}"
        )


def test_filter_is_calibrated_delay0():
    env = Mess3ContinuousEnv(Mess3ContinuousConfig(beta=4.0, delay=0, seed=13))
    obs, info = env.reset(seed=13)
    rng = np.random.default_rng(4)
    n = 100_000
    probs = np.empty(n)
    hits = np.empty(n)
    for t in range(n):
        probs[t] = info["belief"][2]
        hits[t] = float(info["state"] == 2)
        obs, _, _, truncated, info = env.step(rng.uniform(-5, 5, size=2))
        if truncated:
            obs, info = env.reset()
    edges = np.quantile(probs, np.linspace(0, 1, 11))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs <= hi)
        if m.sum() < 500:
            continue
        assert abs(probs[m].mean() - hits[m].mean()) < 0.02


def test_measure_predict_basics():
    E = emission_matrix(0.85)
    b = np.array([0.2, 0.5, 0.3])
    post = measure(b, E, 1)
    assert post[1] > b[1] and post.sum() == pytest.approx(1.0)
    U = tilted_transition(np.array([1.0, -1.0]))
    np.testing.assert_allclose(predict(b, U), b @ U, atol=1e-15)


def test_delay1_reset_belief_is_initial_distribution():
    f = ExactFilter(emission_matrix(0.85), delay=1, init_belief=np.array([0.45, 0.45, 0.10]))
    np.testing.assert_allclose(f.reset(), [0.45, 0.45, 0.10])


def test_delay0_reset_requires_token():
    f = ExactFilter(emission_matrix(0.85), delay=0, init_belief=np.full(3, 1 / 3))
    with pytest.raises(ValueError):
        f.reset()
    b = f.reset(first_token=2)
    assert np.argmax(b) == 2
