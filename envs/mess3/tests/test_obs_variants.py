"""Policy-observation layout and presentation-scrambling tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from envs.hmm import HMMEnv


FULL_DIAGNOSTICS = {
    "state": True,
    "belief": True,
    "raw_belief": True,
    "tokens": True,
    "rewards": True,
    "transitions": True,
}


def occupancy_env(
    observation: dict[str, Any],
    *,
    delay: int = 1,
    action_limit: float = 5.0,
    diagnostics: dict[str, bool] | None = None,
) -> HMMEnv:
    config: dict[str, Any] = {
        "model": {"factory": "envs.mess3.model:control_model"},
        "task": {
            "class": (
                "envs.mess3.tasks.occupancy_control:"
                "OccupancyControlTask"
            ),
            "kwargs": {"action_limit": action_limit},
        },
        "observation": observation,
        "delay": delay,
        "episode_length": 512,
    }
    if diagnostics is not None:
        config["diagnostics"] = diagnostics
    return HMMEnv(config)


def test_hidden_state_observation_shows_true_state():
    env = occupancy_env(
        {
            "token": None,
            "action": None,
            "hidden_state": True,
        },
        diagnostics=FULL_DIAGNOSTICS,
    )
    observation, info = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(100):
        assert observation.shape == (3,)
        assert observation.sum() == 1.0
        assert np.argmax(observation) == info["state_current"]
        observation, _, _, _, info = env.step(
            rng.uniform(-2, 2, size=2)
        )


def test_belief_observation_shows_decision_belief():
    env = occupancy_env(
        {
            "token": None,
            "action": None,
            "belief": True,
        },
        diagnostics=FULL_DIAGNOSTICS,
    )
    observation, info = env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(100):
        np.testing.assert_allclose(
            observation,
            info["belief_current"],
            atol=1e-6,
        )
        observation, _, _, _, info = env.step(
            rng.uniform(-2, 2, size=2)
        )


def test_history_is_grouped_newest_first_tokens_then_actions():
    depth = 4
    env = occupancy_env(
        {
            "token": {"offset": 0, "depth": depth},
            "action": {"offset": 0, "depth": depth},
        },
        diagnostics=FULL_DIAGNOSTICS,
    )
    observation, _ = env.reset(seed=2)
    assert observation.shape == (5 * depth,)
    np.testing.assert_allclose(observation, 0.0)

    seen_tokens: list[int] = []
    seen_actions: list[np.ndarray] = []
    rng = np.random.default_rng(2)
    for _ in range(6):
        action = rng.uniform(-3, 3, size=2)
        observation, _, _, _, info = env.step(action)
        seen_tokens.append(info["visible_token_current"])
        seen_actions.append(info["executed_action"])

    token_width = 3 * depth
    for index in range(depth):
        token_block = observation[index * 3 : (index + 1) * 3]
        action_start = token_width + index * 2
        action_block = observation[action_start : action_start + 2]
        assert np.argmax(token_block) == seen_tokens[-1 - index]
        np.testing.assert_allclose(
            action_block,
            seen_actions[-1 - index],
            atol=1e-6,
        )


def test_history_offsets_select_older_decision_records():
    env = occupancy_env(
        {
            "token": {"offset": 1, "depth": 1},
            "action": {"offset": 1, "depth": 1},
        },
        delay=0,
        diagnostics=FULL_DIAGNOSTICS,
    )
    observation, reset_info = env.reset(seed=12)
    np.testing.assert_allclose(observation, 0.0)

    first_action = np.array([1.0, -1.0])
    observation, _, _, _, first_info = env.step(first_action)
    assert np.argmax(observation[:3]) == reset_info["visible_token_current"]
    np.testing.assert_allclose(observation[3:], 0.0)

    observation, _, _, _, _ = env.step(np.array([-2.0, 2.0]))
    assert np.argmax(observation[:3]) == first_info["visible_token_current"]
    np.testing.assert_allclose(observation[3:], first_action)


def test_uniform_token_scrambling_preserves_raw_path_but_changes_presentation():
    def run(mode: str):
        env = occupancy_env(
            {
                "token": {"offset": 0, "depth": 1},
                "action": None,
                "belief": True,
                "token_scrambling": mode,
            },
            delay=0,
            diagnostics=FULL_DIAGNOSTICS,
        )
        _, info = env.reset(seed=3)
        states = []
        raw_tokens = []
        source_tokens = []
        visible_tokens = []
        beliefs = []
        raw_beliefs = []
        for _ in range(400):
            states.append(info["state_current"])
            raw_tokens.append(info["raw_token_current"])
            source_tokens.append(info["visible_source_token"])
            visible_tokens.append(info["visible_token_current"])
            beliefs.append(info["belief_current"].copy())
            raw_beliefs.append(info["raw_belief_current"].copy())
            _, _, _, _, info = env.step(np.zeros(2))
        return (
            states,
            raw_tokens,
            source_tokens,
            visible_tokens,
            beliefs,
            raw_beliefs,
        )

    plain = run("none")
    scrambled = run("uniform")
    assert plain[0] == scrambled[0]
    assert plain[1] == scrambled[1]
    assert plain[2] == scrambled[2]
    np.testing.assert_allclose(plain[5], scrambled[5])
    assert plain[3] == plain[2]
    assert plain[3] != scrambled[3]

    scrambled_match_rate = np.mean(
        np.asarray(scrambled[3]) == np.asarray(scrambled[2])
    )
    assert scrambled_match_rate < 0.45
    visible_frequencies = np.bincount(scrambled[3], minlength=3) / 400
    assert visible_frequencies.max() < 0.45
    np.testing.assert_allclose(plain[4], plain[5])
    assert np.mean(
        np.abs(np.asarray(scrambled[4]) - np.asarray(scrambled[5]))
    ) > 0.01


def test_uniform_action_scrambling_changes_only_policy_features():
    def run(mode: str):
        env = occupancy_env(
            {
                "token": None,
                "action": {"offset": 0, "depth": 1},
                "action_scrambling": mode,
            },
            diagnostics=FULL_DIAGNOSTICS,
        )
        env.reset(seed=13)
        observations, latent = [], []
        action = np.array([0.75, -0.5])
        for _ in range(50):
            observation, _, _, _, info = env.step(action)
            observations.append(observation.copy())
            latent.append(
                (
                    info["state_current"],
                    info["raw_token_current"],
                    tuple(info["executed_action"]),
                )
            )
        return np.asarray(observations), latent

    plain_observations, plain_latent = run("none")
    scrambled_observations, scrambled_latent = run("uniform")
    assert plain_latent == scrambled_latent
    np.testing.assert_allclose(
        plain_observations,
        np.tile([0.75, -0.5], (50, 1)),
    )
    assert not np.allclose(scrambled_observations, plain_observations)


def test_invalid_scrambling_mode_is_rejected():
    with pytest.raises(ValueError, match="token_scrambling"):
        occupancy_env(
            {
                "token": {"offset": 0, "depth": 1},
                "action": None,
                "token_scrambling": "permutation",
            }
        )


@pytest.mark.parametrize(
    "observation_config",
    [
        {},
        {"token": None, "action": None, "hidden_state": True},
        {"token": None, "action": None, "belief": True},
        {
            "token": {"offset": 0, "depth": 3},
            "action": {"offset": 0, "depth": 3},
        },
    ],
)
def test_observation_space_contains_every_variant(observation_config):
    env = occupancy_env(observation_config, action_limit=0.25)
    observation, _ = env.reset(seed=4)
    assert env.observation_space.contains(observation)
    for _ in range(20):
        observation, *_ = env.step(np.array([0.2, -0.2]))
        assert env.observation_space.contains(observation)
