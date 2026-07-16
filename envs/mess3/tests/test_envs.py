"""Environment-level MESS3 timing, reward, and task checkpoints."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from envs.hmm import HMMEnv, stationary_distribution
from envs.mess3.model import CONTROL_TRANSITION_MATRIX


FULL_DIAGNOSTICS = {
    "state": True,
    "belief": True,
    "raw_belief": True,
    "tokens": True,
    "rewards": True,
    "transitions": True,
}


def occupancy_env(
    *,
    delay: int = 1,
    episode_length: int = 1024,
    seed: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
    task_kwargs: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    diagnostics: dict[str, bool] | None = None,
) -> HMMEnv:
    config: dict[str, Any] = {
        "model": {
            "factory": "envs.mess3.model:control_model",
            "kwargs": {} if model_kwargs is None else model_kwargs,
        },
        "task": {
            "class": (
                "envs.mess3.tasks.occupancy_control:"
                "OccupancyControlTask"
            ),
            "kwargs": {} if task_kwargs is None else task_kwargs,
        },
        "delay": delay,
        "episode_length": episode_length,
        "seed": seed,
    }
    if observation is not None:
        config["observation"] = observation
    if diagnostics is not None:
        config["diagnostics"] = diagnostics
    return HMMEnv(config)


def guess_env(
    *,
    future_horizon: int | None = None,
    episode_length: int = 1024,
    diagnostics: dict[str, bool] | None = None,
) -> HMMEnv:
    if future_horizon is None:
        task_class = "envs.mess3.tasks.state_guess:StateGuessTask"
        task_kwargs: dict[str, Any] = {}
    else:
        task_class = (
            "envs.mess3.tasks.future_state_guess:"
            "FutureStateGuessTask"
        )
        task_kwargs = {"horizon": future_horizon}
    config: dict[str, Any] = {
        "model": {"factory": "envs.mess3.model:state_guess_model"},
        "task": {"class": task_class, "kwargs": task_kwargs},
        "observation": {"action": None},
        "episode_length": episode_length,
    }
    if diagnostics is not None:
        config["diagnostics"] = diagnostics
    return HMMEnv(config)


def test_zero_action_reproduces_p0_stationary_and_zero_kl():
    n_steps = 300_000
    env = occupancy_env(
        episode_length=n_steps + 1,
        model_kwargs={
            "initial_distribution": stationary_distribution(
                CONTROL_TRANSITION_MATRIX
            ),
        },
        task_kwargs={"transition_kl_beta": 4.0},
        diagnostics=FULL_DIAGNOSTICS,
    )
    _, info = env.reset(seed=0)
    states = np.empty(n_steps, dtype=np.int64)
    rewards = np.empty(n_steps)
    transition_kl = np.empty(n_steps)
    for step in range(n_steps):
        _, reward, _, _, info = env.step(np.zeros(2))
        states[step] = info["state_after"]
        rewards[step] = reward
        transition_kl[step] = info["reward_components"]["transition_kl"]

    counts = np.bincount(states, minlength=3) / n_steps
    np.testing.assert_allclose(counts, [0.45, 0.45, 0.10], atol=0.005)
    np.testing.assert_allclose(transition_kl, 0.0, atol=1e-15)
    assert rewards.mean() == pytest.approx(0.10, abs=0.005)


def test_reward_decomposition_is_explicit():
    env = occupancy_env(
        task_kwargs={"transition_kl_beta": 4.0},
        diagnostics=FULL_DIAGNOSTICS,
    )
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(200):
        _, reward, _, _, info = env.step(rng.uniform(-5, 5, size=2))
        components = info["reward_components"]
        assert reward == pytest.approx(
            components["occupancy_reward"]
            + components["transition_kl_penalty"]
        )
        assert components["transition_kl_penalty"] == pytest.approx(
            -components["transition_kl"] / 4.0
        )
        assert components["transition_kl"] >= -1e-12


def test_occupancy_only_has_no_implicit_kl_penalty():
    env = occupancy_env(diagnostics=FULL_DIAGNOSTICS)
    env.reset(seed=1)
    _, reward, _, _, info = env.step(np.array([4.0, -4.0]))
    assert info["reward_components"] == {
        "occupancy_reward": reward,
    }


def test_observation_layout_delay_one():
    env = occupancy_env(delay=1, diagnostics=FULL_DIAGNOSTICS)
    observation, info = env.reset(seed=2)
    assert observation.shape == (5,)
    np.testing.assert_allclose(observation, 0.0)

    action = np.array([1.5, -2.5], dtype=np.float32)
    first_raw_token = info["raw_token_current"]
    observation, _, _, _, info = env.step(action)
    assert info["visible_source_token"] == first_raw_token
    assert info["visible_token_current"] == first_raw_token
    assert observation[:3].sum() == 1.0
    assert np.argmax(observation[:3]) == first_raw_token
    np.testing.assert_allclose(observation[3:], action, atol=1e-6)


def test_observation_layout_delay_zero():
    env = occupancy_env(delay=0, diagnostics=FULL_DIAGNOSTICS)
    observation, info = env.reset(seed=4)
    assert observation.shape == (5,)
    assert observation[:3].sum() == 1.0
    assert np.argmax(observation[:3]) == info["visible_token_current"]
    assert info["visible_source_token"] == info["raw_token_current"]
    np.testing.assert_allclose(observation[3:], 0.0)


def test_action_clipping_changes_requested_but_not_executed_transition():
    env = occupancy_env(
        task_kwargs={"action_limit": 5.0},
        diagnostics=FULL_DIAGNOSTICS,
    )
    env.reset(seed=5)
    _, large_reward, _, _, large_info = env.step(
        np.array([100.0, -100.0])
    )
    env.reset(seed=5)
    _, edge_reward, _, _, edge_info = env.step(np.array([5.0, -5.0]))

    np.testing.assert_allclose(large_info["requested_action"], [100.0, -100.0])
    np.testing.assert_allclose(large_info["executed_action"], [5.0, -5.0])
    np.testing.assert_allclose(
        large_info["executed_transition_matrix"],
        edge_info["executed_transition_matrix"],
    )
    assert large_info["state_after"] == edge_info["state_after"]
    assert large_reward == pytest.approx(edge_reward)


def test_truncation_occurs_exactly_at_episode_length():
    env = occupancy_env(episode_length=16)
    env.reset(seed=6)
    for step in range(16):
        _, _, terminated, truncated, _ = env.step(np.zeros(2))
        assert not terminated
        assert truncated is (step == 15)


def test_step_diagnostics_make_timing_and_transition_explicit():
    env = occupancy_env(diagnostics=FULL_DIAGNOSTICS)
    _, reset_info = env.reset(seed=11)
    state_at_decision = reset_info["state_current"]
    raw_token_at_decision = reset_info["raw_token_current"]
    _, _, _, _, step_info = env.step(np.zeros(2))

    assert step_info["transition_step"] == 0
    assert step_info["decision_step"] == 1
    assert step_info["state_before"] == state_at_decision
    assert step_info["state_after"] == step_info["state_current"]
    assert step_info["raw_token_before"] == raw_token_at_decision
    assert step_info["raw_token_after"] == step_info["raw_token_current"]
    occupancy = step_info["reward_components"]["occupancy_reward"]
    assert occupancy == float(state_at_decision == 2)
    np.testing.assert_allclose(
        step_info["original_transition_matrix"],
        CONTROL_TRANSITION_MATRIX,
    )
    np.testing.assert_allclose(
        step_info["executed_transition_matrix"],
        CONTROL_TRANSITION_MATRIX,
    )
    np.testing.assert_allclose(
        step_info["reference_transition_matrix"],
        CONTROL_TRANSITION_MATRIX,
    )


def test_state_guess_actions_do_not_affect_dynamics():
    def states_under(policy_seed: int) -> list[int]:
        env = guess_env(
            episode_length=501,
            diagnostics=FULL_DIAGNOSTICS,
        )
        _, info = env.reset(seed=7)
        rng = np.random.default_rng(policy_seed)
        states = []
        for _ in range(500):
            _, _, _, _, info = env.step(int(rng.integers(3)))
            states.append(info["state_current"])
        return states

    assert states_under(0) == states_under(1)


def test_state_guess_reward_scores_current_state():
    env = guess_env(diagnostics=FULL_DIAGNOSTICS)
    _, info = env.reset(seed=8)
    for _ in range(200):
        current_state = info["state_current"]
        _, reward, _, _, info = env.step(current_state)
        assert reward == 1.0
        assert info["state_before"] == current_state
        assert info["reward_components"]["state_guess_reward"] == 1.0
        assert info["reward_components"]["state_guess_valid"] == 1.0


def test_state_guess_initial_state_uses_stationary_distribution():
    counts = np.zeros(3)
    env = guess_env(diagnostics={"state": True})
    for seed in range(4000):
        _, info = env.reset(seed=seed)
        counts[info["state_current"]] += 1
    np.testing.assert_allclose(
        counts / counts.sum(),
        [0.45, 0.45, 0.10],
        atol=0.03,
    )


def test_future_state_guess_scores_when_horizon_matures():
    env = guess_env(
        future_horizon=2,
        diagnostics=FULL_DIAGNOSTICS,
    )
    env.reset(seed=9)
    first_guess = 1
    _, first_reward, _, _, first_info = env.step(first_guess)
    assert first_reward == 0.0
    assert first_info["reward_components"]["state_guess_valid"] == 0.0

    _, second_reward, _, _, second_info = env.step(0)
    assert second_reward == float(first_guess == second_info["state_after"])
    assert second_info["reward_components"]["state_guess_valid"] == 1.0


def test_future_predictions_are_discarded_at_truncation():
    env = guess_env(
        future_horizon=3,
        episode_length=1,
        diagnostics=FULL_DIAGNOSTICS,
    )
    env.reset(seed=10)
    _, reward, _, truncated, info = env.step(0)
    assert reward == 0.0
    assert truncated
    assert info["reward_components"]["pending_predictions"] == 1.0
    assert env.task.pending_predictions == 0
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)

    env.reset()
    _, _, _, truncated, _ = env.step(0)
    assert truncated
