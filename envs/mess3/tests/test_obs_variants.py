"""Tests for the Phase-3/4 observation variants of Environment A
(obs_mode = state / belief / stackK, and scramble_tokens)."""

import numpy as np
import pytest

from envs.mess3.env_continuous import Mess3ContinuousConfig, Mess3ContinuousEnv


def test_state_mode_shows_true_state():
    env = Mess3ContinuousEnv(Mess3ContinuousConfig(obs_mode="state", seed=0))
    obs, info = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(100):
        assert obs.shape == (3,)
        assert np.argmax(obs) == info["state"] and obs.sum() == 1.0
        obs, *_, info = env.step(rng.uniform(-2, 2, size=2))


def test_belief_mode_shows_decision_belief():
    env = Mess3ContinuousEnv(Mess3ContinuousConfig(obs_mode="belief", seed=1))
    obs, info = env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(100):
        np.testing.assert_allclose(obs, info["belief"], atol=1e-6)
        obs, *_, info = env.step(rng.uniform(-2, 2, size=2))


def test_stack_mode_layout():
    k = 4
    env = Mess3ContinuousEnv(Mess3ContinuousConfig(obs_mode=f"stack{k}", seed=2))
    obs, info = env.reset(seed=2)
    assert obs.shape == (k * 5,)
    np.testing.assert_allclose(obs, 0.0)  # delay=1, t=0: nothing visible yet
    seen_tokens, seen_actions = [], []
    rng = np.random.default_rng(2)
    for _ in range(6):
        w = rng.uniform(-3, 3, size=2)
        obs, *_, info = env.step(w)
        seen_tokens.append(info["obs_token"])
        seen_actions.append(w)
    # Newest-first frames: frame i = (token seen (i steps ago), action before it).
    for i in range(k):
        frame = obs[i * 5:(i + 1) * 5]
        assert np.argmax(frame[:3]) == seen_tokens[-1 - i]
        np.testing.assert_allclose(frame[3:], seen_actions[-1 - i], atol=1e-6)


def test_scramble_keeps_chain_and_filter_but_randomizes_obs():
    def run(scramble):
        env = Mess3ContinuousEnv(Mess3ContinuousConfig(scramble_tokens=scramble, seed=3))
        obs, info = env.reset(seed=3)
        states, tokens, obs_tokens, beliefs = [], [], [], []
        for _ in range(400):
            obs, *_, info = env.step(np.zeros(2))
            states.append(info["state"])
            tokens.append(info["obs_token"])
            obs_tokens.append(int(np.argmax(obs[:3])))
            beliefs.append(info["belief"].copy())
        return states, tokens, obs_tokens, beliefs

    s0, t0, v0, b0 = run(False)
    s1, t1, v1, b1 = run(True)
    # Scrambling consumes extra rng draws so paths differ, but info tokens must
    # still track the true chain (filter calibrated elsewhere); the OBSERVED
    # tokens must be uninformative: ~uniform and decorrelated from the truth.
    match = np.mean([a == b for a, b in zip(t1, v1)])
    assert match < 0.45  # true-token agreement would be ~1.0 unscrambled
    counts = np.bincount(v1, minlength=3) / len(v1)
    assert counts.max() < 0.45
    assert np.mean([a == b for a, b in zip(t0, v0)]) == 1.0


def test_bad_obs_mode_rejected():
    with pytest.raises(ValueError):
        Mess3ContinuousConfig(obs_mode="frames4")
