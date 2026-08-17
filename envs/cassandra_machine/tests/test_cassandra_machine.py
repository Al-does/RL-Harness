"""Canonical model and Gymnasium checks for machine maintenance."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from envs.cassandra_machine import (
    N_COMPONENTS,
    N_CONDITIONS,
    N_OBSERVATIONS,
    N_STATES,
    Action,
    CassandraMachineConfig,
    CassandraMachineEnv,
    Condition,
    decode_observation,
    decode_state,
    encode_observation,
    encode_state,
    observation_matrix,
    reward_vector,
    transition_matrix,
)


def test_canonical_cardinalities_and_encodings_round_trip():
    assert N_COMPONENTS == 4
    assert N_CONDITIONS == 4
    assert N_STATES == 256
    assert N_OBSERVATIONS == 16

    for state in range(N_STATES):
        assert encode_state(decode_state(state)) == state
    for observation in range(N_OBSERVATIONS):
        assert encode_observation(decode_observation(observation)) == observation


def test_canonical_joint_transition_probabilities():
    operate = transition_matrix(Action.OPERATE)
    inspect = transition_matrix(Action.INSPECT)
    repair = transition_matrix(Action.REPAIR)
    replace = transition_matrix(Action.REPLACE)

    for matrix in (operate, inspect, repair, replace):
        assert matrix.shape == (N_STATES, N_STATES)
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-15)
        assert (matrix >= 0.0).all()

    assert operate[1, 0] == pytest.approx(0.03)
    assert operate[1, 1] == pytest.approx(0.97)
    assert operate[255, 255] == pytest.approx(0.885293, abs=5e-7)
    np.testing.assert_array_equal(inspect, np.eye(N_STATES))
    assert repair[85, 170] == pytest.approx(0.4096)
    np.testing.assert_array_equal(replace[:, 255], np.ones(N_STATES))
    assert replace.sum() == N_STATES


def test_canonical_observation_probabilities_are_action_conditioned():
    inspect = observation_matrix(Action.INSPECT)
    assert inspect.shape == (N_STATES, N_OBSERVATIONS)
    np.testing.assert_allclose(inspect.sum(axis=1), 1.0, atol=1e-15)
    assert inspect[0, 0] == pytest.approx(0.922368, abs=5e-7)
    assert inspect[1, 0] == pytest.approx(0.894132, abs=5e-7)
    assert inspect[2, 1] == pytest.approx(0.752954, abs=5e-7)
    assert inspect[255, 15] == pytest.approx(0.885293, abs=5e-7)

    for action in (Action.OPERATE, Action.REPAIR, Action.REPLACE):
        matrix = observation_matrix(action)
        np.testing.assert_array_equal(matrix[:, 0], np.ones(N_STATES))
        np.testing.assert_array_equal(matrix[:, 1:], 0.0)


def test_canonical_rewards_match_expanded_source_values():
    operate = reward_vector(Action.OPERATE)
    assert operate[0] == 0.0
    assert operate[85] == pytest.approx(0.280112, abs=5e-7)
    assert operate[86] == pytest.approx(0.363472, abs=5e-7)
    assert operate[170] == pytest.approx(0.794123, abs=5e-7)
    assert operate[255] == pytest.approx(0.994013, abs=5e-7)
    np.testing.assert_array_equal(reward_vector(Action.INSPECT), -1.0)
    np.testing.assert_array_equal(reward_vector(Action.REPAIR), -3.0)
    np.testing.assert_array_equal(reward_vector(Action.REPLACE), -15.0)


def test_environment_passes_gymnasium_checker():
    check_env(
        CassandraMachineEnv(
            {
                "episode_length": 8,
                "observation_mode": "factored_belief",
            }
        ),
        skip_render_check=True,
    )


@pytest.mark.parametrize("observation_mode", ["symbol", "factored_belief"])
def test_observation_modes_expose_canonical_information(observation_mode):
    env = CassandraMachineEnv(
        {
            "observation_mode": observation_mode,
            "diagnostics": True,
        }
    )
    observation, info = env.reset(seed=4)

    assert env.observation_space.contains(observation)
    assert info["state_current"] == 255
    np.testing.assert_array_equal(
        info["components_current"],
        np.full(N_COMPONENTS, Condition.GOOD),
    )
    if observation_mode == "symbol":
        assert observation == 0
    else:
        expected = np.zeros((N_COMPONENTS, N_CONDITIONS))
        expected[:, Condition.GOOD] = 1.0
        np.testing.assert_array_equal(
            observation.reshape(N_COMPONENTS, N_CONDITIONS),
            expected,
        )


def test_immediate_reward_uses_pre_transition_condition():
    env = CassandraMachineEnv({"diagnostics": True})
    env.reset(seed=0)
    _, reward, _, _, info = env.step(Action.OPERATE)

    assert reward == pytest.approx(0.9985**4)
    assert info["state_before"] == 255
    assert info["reward_components"] == {"production_reward": reward}


def test_inspection_holds_state_and_replacement_restores_all_components():
    env = CassandraMachineEnv({"diagnostics": True})
    env.reset(seed=9)
    _, inspect_reward, _, _, inspect_info = env.step(Action.INSPECT)
    assert inspect_info["state_before"] == inspect_info["state_after"] == 255
    assert inspect_reward == -1.0

    for _ in range(300):
        env.step(Action.OPERATE)
    assert (env.component_states < Condition.GOOD).any()

    _, replace_reward, _, _, replace_info = env.step(Action.REPLACE)
    assert replace_reward == -15.0
    assert replace_info["state_after"] == 255
    np.testing.assert_array_equal(
        env.component_states,
        np.full(N_COMPONENTS, Condition.GOOD),
    )


def test_factored_belief_matches_dense_bayes_filter():
    env = CassandraMachineEnv(
        {
            "episode_length": 32,
            "observation_mode": "factored_belief",
            "diagnostics": True,
        }
    )
    env.reset(seed=17)
    dense_belief = np.zeros(N_STATES)
    dense_belief[255] = 1.0
    states = np.stack([decode_state(state) for state in range(N_STATES)])
    actions = (
        Action.OPERATE,
        Action.OPERATE,
        Action.INSPECT,
        Action.REPAIR,
        Action.INSPECT,
        Action.OPERATE,
        Action.REPLACE,
        Action.INSPECT,
    )

    for action in actions:
        observation, _, _, _, info = env.step(action)
        dense_belief = dense_belief @ transition_matrix(action)
        dense_belief *= observation_matrix(action)[:, info["observation_symbol"]]
        dense_belief /= dense_belief.sum()
        expected_marginals = np.stack(
            [
                [
                    dense_belief[states[:, component] == condition].sum()
                    for condition in range(N_CONDITIONS)
                ]
                for component in range(N_COMPONENTS)
            ]
        )
        np.testing.assert_allclose(
            observation.reshape(N_COMPONENTS, N_CONDITIONS),
            expected_marginals,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            env.factored_belief,
            expected_marginals,
            atol=1e-12,
        )


def test_seeded_trajectories_are_reproducible():
    actions = np.random.default_rng(2).integers(0, len(Action), size=100)

    def trajectory(seed: int) -> list[tuple[int, float, int]]:
        env = CassandraMachineEnv(
            {"episode_length": len(actions) + 1, "diagnostics": True}
        )
        env.reset(seed=seed)
        records = []
        for action in actions:
            _, reward, _, _, info = env.step(int(action))
            records.append(
                (info["state_current"], reward, info["observation_symbol"])
            )
        return records

    assert trajectory(23) == trajectory(23)
    assert trajectory(23) != trajectory(24)


def test_truncation_requires_reset():
    env = CassandraMachineEnv({"episode_length": 2})
    env.reset(seed=5)
    _, _, terminated, truncated, _ = env.step(Action.OPERATE)
    assert not terminated
    assert not truncated
    _, _, terminated, truncated, _ = env.step(Action.OPERATE)
    assert not terminated
    assert truncated
    with pytest.raises(RuntimeError, match="reset"):
        env.step(Action.OPERATE)


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        ({"episode_length": 0}, ValueError, "episode_length"),
        ({"observation_mode": "state"}, ValueError, "observation_mode"),
        ({"diagnostics": 1}, TypeError, "diagnostics"),
        ({"unknown": True}, TypeError, "unknown"),
    ],
)
def test_config_validation(config, error, message):
    with pytest.raises(error, match=message):
        CassandraMachineConfig.from_value(config)
