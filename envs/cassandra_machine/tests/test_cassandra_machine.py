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
    GlobalAliasAction,
    TargetedAction,
    action_names,
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


def test_targeted_transition_probabilities_affect_only_selected_component():
    before = np.array(
        [Condition.BAD, Condition.FAIR, Condition.BROKEN, Condition.GOOD]
    )
    before_state = encode_state(before)

    repair = transition_matrix(
        TargetedAction.REPAIR_COMPONENT_0,
        action_scope="targeted",
    )
    unchanged = before.copy()
    improved = before.copy()
    improved[0] = Condition.FAIR
    assert repair[before_state, encode_state(unchanged)] == pytest.approx(0.2)
    assert repair[before_state, encode_state(improved)] == pytest.approx(0.8)
    assert np.count_nonzero(repair[before_state]) == 2

    broken_repair = transition_matrix(
        TargetedAction.REPAIR_COMPONENT_2,
        action_scope="targeted",
    )
    assert broken_repair[before_state, before_state] == 1.0

    replace = transition_matrix(
        TargetedAction.REPLACE_COMPONENT_2,
        action_scope="targeted",
    )
    replaced = before.copy()
    replaced[2] = Condition.GOOD
    assert replace[before_state, encode_state(replaced)] == 1.0
    assert np.count_nonzero(replace[before_state]) == 1


def test_global_alias_actions_exactly_duplicate_canonical_maintenance():
    assert len(action_names("global_aliases")) == len(TargetedAction) == 10
    for component in range(N_COMPONENTS):
        repair = GlobalAliasAction.REPAIR_ALIAS_0 + component
        replace = GlobalAliasAction.REPLACE_ALIAS_0 + component
        np.testing.assert_array_equal(
            transition_matrix(repair, action_scope="global_aliases"),
            transition_matrix(Action.REPAIR),
        )
        np.testing.assert_array_equal(
            observation_matrix(repair, action_scope="global_aliases"),
            observation_matrix(Action.REPAIR),
        )
        np.testing.assert_array_equal(
            reward_vector(repair, action_scope="global_aliases"),
            reward_vector(Action.REPAIR),
        )
        np.testing.assert_array_equal(
            transition_matrix(replace, action_scope="global_aliases"),
            transition_matrix(Action.REPLACE),
        )
        np.testing.assert_array_equal(
            observation_matrix(replace, action_scope="global_aliases"),
            observation_matrix(Action.REPLACE),
        )
        np.testing.assert_array_equal(
            reward_vector(replace, action_scope="global_aliases"),
            reward_vector(Action.REPLACE),
        )


def test_canonical_observation_probabilities_are_action_conditioned():
    inspect = observation_matrix(Action.INSPECT)
    assert inspect.shape == (N_STATES, N_OBSERVATIONS)
    np.testing.assert_allclose(inspect.sum(axis=1), 1.0, atol=1e-15)
    assert inspect[0, 0] == pytest.approx(0.922368, abs=5e-7)
    assert inspect[1, 0] == pytest.approx(0.894132, abs=5e-7)
    assert inspect[2, 1] == pytest.approx(0.752954, abs=5e-7)
    assert inspect[255, 15] == pytest.approx(0.885293, abs=5e-7)

    operate = observation_matrix(Action.OPERATE)
    assert operate[0, 0] == 1.0
    assert operate[181, 15] == pytest.approx(0.534375)
    assert operate[255, 15] == 1.0
    np.testing.assert_array_equal(operate[:, 1:15], 0.0)

    for action in (Action.REPAIR, Action.REPLACE):
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


def test_targeted_action_costs_are_state_independent():
    for component in range(N_COMPONENTS):
        repair = TargetedAction.REPAIR_COMPONENT_0 + component
        replace = TargetedAction.REPLACE_COMPONENT_0 + component
        np.testing.assert_array_equal(
            reward_vector(repair, action_scope="targeted"),
            -0.75,
        )
        np.testing.assert_array_equal(
            reward_vector(replace, action_scope="targeted"),
            -3.75,
        )


@pytest.mark.parametrize(
    "action_scope",
    ["global", "global_aliases", "targeted"],
)
def test_environment_passes_gymnasium_checker(action_scope):
    check_env(
        CassandraMachineEnv(
            {
                "episode_length": 8,
                "observation_mode": "factored_belief",
                "action_scope": action_scope,
            }
        ),
        skip_render_check=True,
    )


@pytest.mark.parametrize(
    "observation_mode",
    ["symbol", "state", "belief", "factored_belief"],
)
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
    elif observation_mode == "state":
        expected = np.zeros(N_STATES)
        expected[255] = 1.0
        np.testing.assert_array_equal(observation, expected)
    elif observation_mode == "belief":
        expected = np.zeros(N_STATES)
        expected[255] = 1.0
        np.testing.assert_array_equal(observation, expected)
    else:
        expected = np.zeros((N_COMPONENTS, N_CONDITIONS))
        expected[:, Condition.GOOD] = 1.0
        np.testing.assert_array_equal(
            observation.reshape(N_COMPONENTS, N_CONDITIONS),
            expected,
        )


@pytest.mark.parametrize(
    ("action_scope", "replacement"),
    [
        ("global_aliases", GlobalAliasAction.REPLACE_ALIAS_2),
        ("targeted", TargetedAction.REPLACE_COMPONENT_2),
    ],
)
def test_fully_observable_action_variants_return_current_state(
    action_scope,
    replacement,
):
    env = CassandraMachineEnv(
        {
            "action_scope": action_scope,
            "observation_mode": "state",
            "initial_state_distribution": "uniform",
            "diagnostics": True,
        }
    )
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=7)

    assert env.observation_space.shape == (N_STATES,)
    assert np.argmax(observation) == info["state_current"]
    assert np.argmax(observation) == encode_state(env.component_states)
    assert observation.sum() == 1.0

    observation, _, _, _, info = env.step(replacement)

    assert np.argmax(observation) == info["state_current"] == info["state_after"]
    assert np.argmax(observation) == encode_state(env.component_states)
    assert observation.sum() == 1.0
    assert env.observation_space.contains(observation)


def test_state_observations_can_disable_unused_belief_tracking():
    env = CassandraMachineEnv(
        {
            "observation_mode": "state",
            "track_belief": False,
            "diagnostics": False,
        }
    )
    observation, _ = env.reset(seed=7)
    env._advance_belief = lambda *args: pytest.fail(
        "disabled belief tracking advanced the filter"
    )

    observation, _, _, _, _ = env.step(Action.OPERATE)

    assert env.observation_space.contains(observation)
    with pytest.raises(RuntimeError, match="disabled"):
        _ = env.belief
    with pytest.raises(RuntimeError, match="disabled"):
        _ = env.factored_belief


def test_uniform_initial_distribution_is_seeded_and_matches_prior_belief():
    env = CassandraMachineEnv(
        {
            "initial_state_distribution": "uniform",
            "observation_mode": "belief",
            "diagnostics": True,
        }
    )
    first_observation, first_info = env.reset(seed=42)
    second_observation, second_info = env.reset(seed=42)

    np.testing.assert_array_equal(
        first_info["components_current"],
        second_info["components_current"],
    )
    np.testing.assert_allclose(first_observation, 1.0 / N_STATES)
    np.testing.assert_allclose(second_observation, 1.0 / N_STATES)
    np.testing.assert_allclose(
        first_info["factored_belief_current"],
        1.0 / N_CONDITIONS,
    )
    states = {
        tuple(env.reset(seed=seed)[1]["components_current"])
        for seed in range(32)
    }
    assert len(states) > 1


def test_global_alias_environment_applies_global_repair_and_replace():
    env = CassandraMachineEnv(
        {
            "action_scope": "global_aliases",
            "initial_state_distribution": "uniform",
            "diagnostics": True,
        }
    )
    env.reset(seed=7)
    _, repair_reward, _, _, repair_info = env.step(
        GlobalAliasAction.REPAIR_ALIAS_3
    )
    assert repair_reward == -3.0
    assert repair_info["action_name"] == "repair_alias_3"

    _, replace_reward, _, _, replace_info = env.step(
        GlobalAliasAction.REPLACE_ALIAS_1
    )
    assert replace_reward == -15.0
    assert replace_info["action_name"] == "replace_alias_1"
    np.testing.assert_array_equal(
        replace_info["components_after"],
        np.full(N_COMPONENTS, Condition.GOOD),
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


def test_targeted_environment_has_no_global_maintenance_actions():
    env = CassandraMachineEnv(
        {"action_scope": "targeted", "diagnostics": True}
    )
    env.reset(seed=9)

    assert env.action_space.n == 10
    _, repair_reward, _, _, repair_info = env.step(
        TargetedAction.REPAIR_COMPONENT_0
    )
    assert repair_reward == -0.75
    assert repair_info["action_name"] == "repair_component_0"
    np.testing.assert_array_equal(
        repair_info["components_after"],
        np.full(N_COMPONENTS, Condition.GOOD),
    )

    for _ in range(300):
        env.step(TargetedAction.OPERATE)
    before = env.component_states
    assert (before < Condition.GOOD).all()

    _, replace_reward, _, _, replace_info = env.step(
        TargetedAction.REPLACE_COMPONENT_2
    )
    expected = before.copy()
    expected[2] = Condition.GOOD
    assert replace_reward == -3.75
    assert replace_info["action_name"] == "replace_component_2"
    np.testing.assert_array_equal(env.component_states, expected)


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


def test_targeted_factored_belief_matches_dense_bayes_filter():
    env = CassandraMachineEnv(
        {
            "action_scope": "targeted",
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
        TargetedAction.OPERATE,
        TargetedAction.INSPECT,
        TargetedAction.REPAIR_COMPONENT_0,
        TargetedAction.OPERATE,
        TargetedAction.REPLACE_COMPONENT_3,
        TargetedAction.INSPECT,
    )

    for action in actions:
        observation, _, _, _, info = env.step(action)
        dense_belief = dense_belief @ transition_matrix(
            action,
            action_scope="targeted",
        )
        dense_belief *= observation_matrix(
            action,
            action_scope="targeted",
        )[:, info["observation_symbol"]]
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
        ({"observation_mode": "latent"}, ValueError, "observation_mode"),
        ({"action_scope": "local"}, ValueError, "action_scope"),
        (
            {"initial_state_distribution": "stationary"},
            ValueError,
            "initial_state_distribution",
        ),
        ({"diagnostics": 1}, TypeError, "diagnostics"),
        ({"track_belief": 1}, TypeError, "track_belief"),
        (
            {"observation_mode": "belief", "track_belief": False},
            ValueError,
            "track_belief",
        ),
        (
            {"diagnostics": True, "track_belief": False},
            ValueError,
            "track_belief",
        ),
        ({"unknown": True}, TypeError, "unknown"),
    ],
)
def test_config_validation(config, error, message):
    with pytest.raises(error, match=message):
        CassandraMachineConfig.from_value(config)
