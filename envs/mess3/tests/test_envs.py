"""Environment-level behavior tests for both environments."""

import numpy as np
import pytest

from envs.mess3.env_continuous import OBS_DIM, Mess3ContinuousConfig, Mess3ContinuousEnv
from envs.mess3.env_stateguess import StateGuessConfig, StateGuessEnv


def rollout_states(env, policy, n, seed):
    obs, info = env.reset(seed=seed)
    states, rewards, costs = [], [], []
    for _ in range(n):
        obs, r, _, truncated, info = env.step(policy(obs, info))
        states.append(info["state"])
        rewards.append(r)
        costs.append(info["reward_control_cost"])
        if truncated:
            obs, info = env.reset()
    return np.array(states), np.array(rewards), np.array(costs)


def test_zero_action_reproduces_p0_stationary_and_zero_cost():
    env = Mess3ContinuousEnv(Mess3ContinuousConfig(seed=0))
    states, rewards, costs = rollout_states(env, lambda o, i: np.zeros(2), 300_000, seed=0)
    counts = np.bincount(states, minlength=3) / len(states)
    np.testing.assert_allclose(counts, [0.45, 0.45, 0.10], atol=0.005)
    assert np.all(costs == 0.0)
    # w=0 forever: reward is pure occupancy of state 2.
    assert rewards.mean() == pytest.approx(0.10, abs=0.005)


def test_reward_decomposition():
    env = Mess3ContinuousEnv(Mess3ContinuousConfig(beta=4.0, seed=1))
    obs, info = env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs, r, _, _, info = env.step(rng.uniform(-5, 5, size=2))
        assert r == pytest.approx(info["reward_occupancy"] - info["reward_control_cost"])
        assert info["reward_control_cost"] >= 0.0


def test_obs_layout_delay1():
    env = Mess3ContinuousEnv(Mess3ContinuousConfig(delay=1, seed=2))
    obs, info = env.reset(seed=2)
    assert obs.shape == (OBS_DIM,)
    # t=0: no token revealed yet, previous action zero.
    np.testing.assert_allclose(obs, 0.0)
    w = np.array([1.5, -2.5], dtype=np.float32)
    first_emitted = info["emitted_token"]
    obs, *_ , info = env.step(w)
    # Token slot now carries o_0; action slot carries w_0.
    assert obs[:3].sum() == 1.0 and np.argmax(obs[:3]) == first_emitted
    np.testing.assert_allclose(obs[3:], w, atol=1e-6)


def test_obs_layout_delay0():
    env = Mess3ContinuousEnv(Mess3ContinuousConfig(delay=0, seed=4))
    obs, info = env.reset(seed=4)
    assert obs[:3].sum() == 1.0 and np.argmax(obs[:3]) == info["obs_token"]


def test_action_clipping():
    env = Mess3ContinuousEnv(Mess3ContinuousConfig(w_max2=5.0, seed=5))
    env.reset(seed=5)
    _, r_big, *_ = env.step(np.array([100.0, -100.0]))
    env.reset(seed=5)
    _, r_edge, *_ = env.step(np.array([5.0, -5.0]))
    assert r_big == pytest.approx(r_edge)


def test_determinism_given_seed():
    def trace(env_cls, cfg, seed):
        env = env_cls(cfg)
        obs, info = env.reset(seed=seed)
        rng = np.random.default_rng(99)
        out = []
        for _ in range(200):
            a = rng.uniform(-1, 1, size=2) if hasattr(env.action_space, "low") else int(rng.integers(3))
            obs, r, _, _, info = env.step(a)
            out.append((info["state"], r, tuple(np.round(info["belief"], 12))))
        return out

    t1 = trace(Mess3ContinuousEnv, Mess3ContinuousConfig(), 123)
    t2 = trace(Mess3ContinuousEnv, Mess3ContinuousConfig(), 123)
    assert t1 == t2
    t3 = trace(StateGuessEnv, StateGuessConfig(), 123)
    t4 = trace(StateGuessEnv, StateGuessConfig(), 123)
    assert t3 == t4


def test_truncation_at_episode_length():
    env = Mess3ContinuousEnv(Mess3ContinuousConfig(episode_length=16, seed=6))
    env.reset(seed=6)
    for t in range(16):
        _, _, terminated, truncated, _ = env.step(np.zeros(2))
        assert not terminated
    assert truncated


def test_stateguess_actions_do_not_affect_dynamics():
    def states_under(policy_seed):
        env = StateGuessEnv(StateGuessConfig(seed=7))
        obs, info = env.reset(seed=7)
        rng = np.random.default_rng(policy_seed)
        out = []
        for _ in range(500):
            obs, _, _, _, info = env.step(int(rng.integers(3)))
            out.append(info["state"])
        return out

    assert states_under(0) == states_under(1)


def test_stateguess_reward_is_guess_accuracy():
    env = StateGuessEnv(StateGuessConfig(seed=8))
    obs, info = env.reset(seed=8)
    for _ in range(200):
        s = info["state"]
        _, r, _, _, info = env.step(s)
        assert r == 1.0


def test_stateguess_initial_state_from_stationary():
    counts = np.zeros(3)
    env = StateGuessEnv(StateGuessConfig())
    for seed in range(4000):
        _, info = env.reset(seed=seed)
        counts[info["state"]] += 1
    np.testing.assert_allclose(counts / counts.sum(), [0.45, 0.45, 0.10], atol=0.03)
