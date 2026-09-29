"""Benchmark semantics and Gymnasium checks for RockSample."""

from __future__ import annotations

from math import log

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from envs.rocksample import (
    CANONICAL_INSTANCES,
    Action,
    Observation,
    RockSampleConfig,
    RockSampleEnv,
    action_names,
    check_action,
    instance_definition,
    sensor_efficiency,
)


EXPECTED_INSTANCES = {
    (4, 4): ((0, 2), ((3, 1), (2, 1), (1, 3), (1, 0)), log(2.0)),
    (5, 5): (
        (0, 2),
        ((2, 4), (0, 4), (3, 3), (2, 2), (4, 1)),
        4.0,
    ),
    (5, 7): (
        (0, 2),
        ((1, 0), (2, 1), (1, 2), (2, 2), (4, 2), (0, 3), (3, 4)),
        20.0,
    ),
    (7, 8): (
        (0, 3),
        (
            (2, 0),
            (0, 1),
            (3, 1),
            (6, 3),
            (2, 4),
            (3, 4),
            (5, 5),
            (1, 6),
        ),
        20.0,
    ),
    (10, 10): (
        (0, 5),
        (
            (0, 3),
            (0, 7),
            (1, 8),
            (3, 3),
            (3, 8),
            (4, 3),
            (5, 8),
            (6, 1),
            (9, 3),
            (9, 9),
        ),
        20.0,
    ),
    (11, 11): (
        (0, 5),
        (
            (0, 3),
            (0, 7),
            (1, 8),
            (2, 4),
            (3, 3),
            (3, 8),
            (4, 3),
            (5, 8),
            (6, 1),
            (9, 3),
            (9, 9),
        ),
        20.0,
    ),
}


def move_to(env: RockSampleEnv, target: tuple[int, int]) -> None:
    x, y = env.rover_position
    target_x, target_y = target
    horizontal = Action.EAST if target_x > x else Action.WEST
    vertical = Action.NORTH if target_y > y else Action.SOUTH
    for _ in range(abs(target_x - x)):
        env.step(horizontal)
    for _ in range(abs(target_y - y)):
        env.step(vertical)


def observation_symbol_features(
    env: RockSampleEnv,
    observation: np.ndarray,
) -> np.ndarray:
    offset = 2 + 2 * env.config.k
    return observation[offset : offset + len(Observation)]


def previous_action_features(
    env: RockSampleEnv,
    observation: np.ndarray,
) -> np.ndarray:
    return observation[-env.action_space.n :]


def test_canonical_instance_definitions_match_zmdp_and_despot():
    assert set(CANONICAL_INSTANCES) == set(EXPECTED_INSTANCES)
    for size, (start, rocks, d0) in EXPECTED_INSTANCES.items():
        instance = CANONICAL_INSTANCES[size]
        assert instance.start == start
        assert instance.rocks == rocks
        assert instance.sensor_half_efficiency_distance == pytest.approx(d0)


def test_default_observation_contains_layout_symbol_and_no_previous_action():
    env = RockSampleEnv({"evaluation": True})
    observation, info = env.reset(seed=3)

    assert env.action_space.n == 9
    assert observation.shape == (22,)
    assert env.observation_space.contains(observation)
    np.testing.assert_allclose(observation[:2], [0.0, 2.0 / 3.0])
    np.testing.assert_allclose(
        observation[2:10],
        np.asarray(EXPECTED_INSTANCES[(4, 4)][1]).reshape(-1) / 3.0,
    )
    np.testing.assert_array_equal(
        observation_symbol_features(env, observation),
        [0.0, 0.0, 1.0],
    )
    np.testing.assert_array_equal(
        previous_action_features(env, observation),
        np.zeros(env.action_space.n),
    )
    assert info["observation_symbol"] == Observation.NONE


def test_observation_contains_one_hot_previous_action():
    env = RockSampleEnv({"evaluation": True})
    observation, _ = env.reset(seed=3)
    np.testing.assert_array_equal(
        previous_action_features(env, observation),
        np.zeros(env.action_space.n),
    )

    for action in (Action.WEST, check_action(2), Action.EAST):
        observation, _, _, _, _ = env.step(action)
        expected = np.zeros(env.action_space.n)
        expected[int(action)] = 1.0
        np.testing.assert_array_equal(
            previous_action_features(env, observation),
            expected,
        )


@pytest.mark.parametrize(
    "config",
    [
        {"evaluation": True},
        {"randomize_train_layout": False},
    ],
)
def test_evaluation_and_fixed_training_use_canonical_layout(config):
    env = RockSampleEnv(config)
    layouts = []
    for seed in range(4):
        env.reset(seed=seed)
        layouts.append(env.rock_positions)
    assert layouts == [EXPECTED_INSTANCES[(4, 4)][1]] * 4


def test_randomized_training_layouts_are_distinct_and_hold_out_evaluation():
    env = RockSampleEnv()
    evaluation_cells = frozenset(EXPECTED_INSTANCES[(4, 4)][1])
    layouts = []
    for _ in range(20):
        env.reset()
        layout = env.rock_positions
        layouts.append(layout)
        assert len(layout) == len(set(layout)) == 4
        assert env.instance.start not in layout
        assert frozenset(layout) != evaluation_cells
    assert len(set(layouts)) > 1


def test_nonstandard_evaluation_layout_is_seeded_and_excludes_start():
    first = instance_definition(6, 5, layout_seed=17)
    repeated = instance_definition(6, 5, layout_seed=17)
    different = instance_definition(6, 5, layout_seed=18)

    assert first == repeated
    assert first != different
    assert first.start == (0, 3)
    assert first.layout_seed == 17
    assert len(first.rocks) == len(set(first.rocks)) == 5
    assert first.start not in first.rocks
    assert first.sensor_half_efficiency_distance == 20.0


def test_illegal_actions_are_noop_and_accumulate_episode_rate():
    env = RockSampleEnv({"evaluation": True})
    env.reset(seed=5)

    observation, reward, terminated, truncated, info = env.step(Action.WEST)
    assert env.rover_position == (0, 2)
    assert reward == 0.0
    assert not terminated and not truncated
    assert info["illegal_action"]
    assert info["illegal_action_count"] == 1
    assert info["illegal_action_rate"] == 1.0
    np.testing.assert_array_equal(
        observation_symbol_features(env, observation),
        [0.0, 0.0, 1.0],
    )

    _, reward, _, _, info = env.step(Action.SAMPLE)
    assert reward == 0.0
    assert info["illegal_action"]
    assert info["illegal_action_count"] == 2
    assert info["illegal_action_rate"] == 1.0

    _, _, _, _, info = env.step(Action.EAST)
    assert not info["illegal_action"]
    assert info["illegal_action_rate"] == pytest.approx(2.0 / 3.0)


def test_exit_east_terminates_with_reward():
    env = RockSampleEnv({"evaluation": True})
    env.reset(seed=7)
    for _ in range(3):
        _, reward, terminated, truncated, _ = env.step(Action.EAST)
        assert reward == 0.0
        assert not terminated and not truncated
    _, reward, terminated, truncated, _ = env.step(Action.EAST)
    assert reward == 10.0
    assert terminated and not truncated


def test_sampling_good_or_bad_rock_then_resampling_is_bad():
    env = RockSampleEnv({"evaluation": True})
    env.reset(seed=11)
    rock_index = 2
    initially_good = bool(env.rock_qualities[rock_index])
    move_to(env, env.rock_positions[rock_index])

    observation, reward, terminated, truncated, info = env.step(Action.SAMPLE)
    assert reward == (10.0 if initially_good else -10.0)
    assert not terminated and not truncated
    assert not info["illegal_action"]
    assert not env.rock_qualities[rock_index]
    np.testing.assert_array_equal(
        observation_symbol_features(env, observation),
        [0.0, 0.0, 1.0],
    )

    _, reward, _, _, info = env.step(Action.SAMPLE)
    assert reward == -10.0
    assert not info["illegal_action"]


def test_check_is_perfect_on_rock_and_uses_good_bad_symbols():
    env = RockSampleEnv({"evaluation": True})
    env.reset(seed=19)
    rock_index = 2
    initially_good = bool(env.rock_qualities[rock_index])
    move_to(env, env.rock_positions[rock_index])

    observation, reward, terminated, truncated, info = env.step(
        check_action(rock_index)
    )
    expected = Observation.GOOD if initially_good else Observation.BAD
    assert reward == 0.0
    assert not terminated and not truncated
    assert info["observation_symbol"] == expected
    expected_one_hot = np.zeros(3)
    expected_one_hot[expected] = 1.0
    np.testing.assert_array_equal(
        observation_symbol_features(env, observation),
        expected_one_hot,
    )


def test_sensor_efficiency_has_expected_half_distance_and_limits():
    assert sensor_efficiency(0.0, 4.0) == 1.0
    assert sensor_efficiency(4.0, 4.0) == pytest.approx(0.5)
    assert sensor_efficiency(8.0, 4.0) == pytest.approx(0.25)


def test_seed_reproduces_layout_qualities_and_sensor_trajectory():
    actions = [
        check_action(0),
        Action.EAST,
        check_action(1),
        Action.NORTH,
        check_action(2),
    ]

    def trajectory(seed: int):
        env = RockSampleEnv()
        env.reset(seed=seed)
        records = [(env.rock_positions, tuple(env.rock_qualities))]
        for action in actions:
            _, reward, terminated, truncated, info = env.step(action)
            records.append(
                (
                    reward,
                    terminated,
                    truncated,
                    info["observation_symbol"],
                    env.rover_position,
                )
            )
        return records

    assert trajectory(23) == trajectory(23)
    assert trajectory(23) != trajectory(24)


def test_diagnostics_control_hidden_quality_info():
    _, ordinary = RockSampleEnv({"evaluation": True}).reset(seed=2)
    _, diagnostic = RockSampleEnv(
        {"evaluation": True, "diagnostics": True}
    ).reset(seed=2)
    assert "rock_qualities" not in ordinary
    assert "rock_qualities" in diagnostic


def test_truncation_requires_reset():
    env = RockSampleEnv({"episode_length": 2})
    env.reset(seed=5)
    _, _, terminated, truncated, _ = env.step(check_action(0))
    assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(check_action(0))
    assert not terminated and truncated
    with pytest.raises(RuntimeError, match="reset"):
        env.step(check_action(0))


@pytest.mark.parametrize(
    "config",
    [
        {"evaluation": True},
        {"randomize_train_layout": True},
        {"n": 6, "k": 5, "evaluation": True},
    ],
)
def test_environment_passes_gymnasium_checker(config):
    check_env(RockSampleEnv(config), skip_render_check=True)


def test_rllib_env_runner_integration():
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner

    config = (
        PPOConfig()
        .environment(
            RockSampleEnv,
            env_config={"episode_length": 8},
        )
        .env_runners(num_env_runners=0, rollout_fragment_length=16)
        .rl_module(model_config={"fcnet_hiddens": [8]})
    )
    runner = SingleAgentEnvRunner(config=config)
    try:
        episodes = runner.sample(num_timesteps=16)
        assert sum(len(episode) for episode in episodes) == 16
        assert all(
            episode.observations[0].shape == (22,)
            for episode in episodes
        )
    finally:
        runner.stop()


def test_action_helpers_follow_documented_order():
    assert action_names(3) == (
        "north",
        "south",
        "east",
        "west",
        "sample",
        "check_0",
        "check_1",
        "check_2",
    )
    assert check_action(0) == 5
    assert check_action(3) == 8


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        ({"n": 1}, ValueError, "n"),
        ({"n": True}, ValueError, "n"),
        ({"k": 0}, ValueError, "k"),
        ({"n": 2, "k": 4}, ValueError, "non-start"),
        ({"randomize_train_layout": 1}, TypeError, "randomize"),
        ({"evaluation": 1}, TypeError, "evaluation"),
        ({"episode_length": 0}, ValueError, "episode_length"),
        ({"eval_layout_seed": None}, TypeError, "eval_layout_seed"),
        ({"diagnostics": 1}, TypeError, "diagnostics"),
        ({"seed": 1.5}, TypeError, "seed"),
        ({"n": 2, "k": 3}, ValueError, "distinct"),
        ({"unknown": True}, TypeError, "unknown"),
    ],
)
def test_invalid_config_is_rejected(config, error, message):
    with pytest.raises(error, match=message):
        RockSampleConfig.from_value(config)
