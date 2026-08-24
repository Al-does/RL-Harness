"""Behavioral checkpoints for the public finite-HMM API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from envs.hmm import (
    ActionDecision,
    BeliefTracker,
    FactorCoupling,
    HMMEnv,
    HMMModel,
    TransitionEvent,
    compose_hmm_factors,
    factor_marginals,
    factored_model,
    measure,
    predict,
    product_distribution,
)


FULL_DIAGNOSTICS = {
    "state": True,
    "belief": True,
    "raw_belief": True,
    "tokens": True,
    "rewards": True,
    "transitions": True,
}


def tiny_model_factory() -> HMMModel:
    """Top-level factory used to exercise import-path construction."""

    return HMMModel(
        initial_distribution=np.array([0.75, 0.25]),
        transition_matrix=np.array([[0.8, 0.2], [0.1, 0.9]]),
        emission_matrix=np.array([[0.9, 0.1], [0.2, 0.8]]),
    )


def other_tiny_model_factory() -> HMMModel:
    """A distinct factor for composition and import-path tests."""

    return HMMModel(
        initial_distribution=np.array([0.4, 0.6]),
        transition_matrix=np.array([[0.6, 0.4], [0.3, 0.7]]),
        emission_matrix=np.array([[0.7, 0.3], [0.1, 0.9]]),
    )


class InlineGuessTask:
    """Minimal top-level task used by the generic environment integration."""

    requires_belief = False

    def __init__(self, *, model: HMMModel) -> None:
        self.action_space = gym.spaces.Discrete(model.n_states)
        self.action_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(model.n_states,),
            dtype=np.float32,
        )

    def reset(self) -> None:
        pass

    def resolve_action(
        self,
        action: int,
        state: int,
        model: HMMModel,
    ) -> ActionDecision:
        del state
        guess = int(action)
        if not self.action_space.contains(guess):
            raise ValueError("guess is outside the action space")
        return ActionDecision(
            requested_action=guess,
            executed_action=guess,
            transition_matrix=model.transition_matrix,
        )

    def reward(
        self,
        event: TransitionEvent,
        decision: ActionDecision,
    ) -> tuple[float, dict[str, float]]:
        reward = float(decision.executed_action == event.state_before)
        return reward, {"pre_transition_accuracy": reward}

    def encode_action(self, executed_action: int) -> np.ndarray:
        encoded = np.zeros(self.action_space.n, dtype=np.float32)
        encoded[int(executed_action)] = 1.0
        return encoded


@pytest.fixture
def make_env() -> Callable[..., HMMEnv]:
    """Construct the one inline HMM integration used by this module."""

    def make(
        *,
        delay: int = 1,
        observation: dict[str, Any] | None = None,
        diagnostics: dict[str, bool] | None = None,
        episode_length: int = 256,
        randomize_first_episode_length: bool = False,
        seed: int | None = None,
    ) -> HMMEnv:
        config: dict[str, Any] = {
            "model": {"factory": f"{__name__}:tiny_model_factory"},
            "task": {"class": f"{__name__}:InlineGuessTask"},
            "delay": delay,
            "episode_length": episode_length,
            "randomize_first_episode_length": randomize_first_episode_length,
            "seed": seed,
        }
        if observation is not None:
            config["observation"] = observation
        if diagnostics is not None:
            config["diagnostics"] = diagnostics
        return HMMEnv(config)

    return make


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("initial_distribution", np.array([0.4, 0.4])),
        (
            "transition_matrix",
            np.array([[0.8, 0.3], [0.1, 0.9]]),
        ),
        (
            "emission_matrix",
            np.array([[0.9, 0.2], [0.2, 0.8]]),
        ),
    ],
)
def test_model_rejects_non_stochastic_probabilities(field, value):
    values = {
        "initial_distribution": np.array([0.5, 0.5]),
        "transition_matrix": np.array([[0.8, 0.2], [0.1, 0.9]]),
        "emission_matrix": np.array([[0.9, 0.1], [0.2, 0.8]]),
    }
    values[field] = value
    with pytest.raises(ValueError, match=field):
        HMMModel(**values)


def test_model_owns_immutable_probability_copies():
    initial = np.array([1.0, 0.0])
    transition = np.eye(2)
    emission = np.eye(2)
    model = HMMModel(
        initial_distribution=initial,
        transition_matrix=transition,
        emission_matrix=emission,
    )

    initial[:] = [0.0, 1.0]
    transition[0] = [0.0, 1.0]
    emission[0] = [0.0, 1.0]
    np.testing.assert_array_equal(model.initial_distribution, [1.0, 0.0])
    np.testing.assert_array_equal(model.transition_matrix, np.eye(2))
    np.testing.assert_array_equal(model.emission_matrix, np.eye(2))

    for probabilities in (
        model.initial_distribution,
        model.transition_matrix,
        model.emission_matrix,
    ):
        with pytest.raises(ValueError):
            probabilities.flat[0] = 0.0


def test_independent_factor_factory_is_exact_cartesian_product():
    first = tiny_model_factory()
    second = other_tiny_model_factory()
    model = compose_hmm_factors([first, second])

    assert model.n_states == 4
    assert model.n_tokens == 4
    np.testing.assert_allclose(
        model.initial_distribution,
        np.kron(first.initial_distribution, second.initial_distribution),
    )
    np.testing.assert_allclose(
        model.transition_matrix,
        np.kron(first.transition_matrix, second.transition_matrix),
    )
    np.testing.assert_allclose(
        model.emission_matrix,
        np.kron(first.emission_matrix, second.emission_matrix),
    )


def test_factor_factory_can_merge_subtoken_tuples_into_one_token():
    first = HMMModel(
        initial_distribution=np.array([1.0]),
        transition_matrix=np.ones((1, 1)),
        emission_matrix=np.array([[0.25, 0.75]]),
    )
    second = HMMModel(
        initial_distribution=np.array([1.0]),
        transition_matrix=np.ones((1, 1)),
        emission_matrix=np.array([[0.6, 0.4]]),
    )
    # XOR-like deterministic composition: equal subtokens map to zero.
    model = compose_hmm_factors(
        [first, second],
        token_map=[[0, 1], [1, 0]],
    )

    np.testing.assert_allclose(
        model.emission_matrix,
        [[0.25 * 0.6 + 0.75 * 0.4, 0.25 * 0.4 + 0.75 * 0.6]],
    )


def test_directional_coupling_interpolates_child_dynamics():
    parent = tiny_model_factory()
    child = other_tiny_model_factory()
    conditional = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    independent = compose_hmm_factors(
        [parent, child],
        couplings=[
            FactorCoupling(
                parent=0,
                child=1,
                transition_matrices=conditional,
                strength=0.0,
            )
        ],
    )
    coupled = compose_hmm_factors(
        [parent, child],
        couplings=[
            {
                "parent": 0,
                "child": 1,
                "transition_matrices": conditional,
                "strength": 1.0,
            }
        ],
    )
    halfway = compose_hmm_factors(
        [parent, child],
        couplings=[
            {
                "parent": 0,
                "child": 1,
                "transition_matrices": conditional,
                "strength": 0.5,
            }
        ],
    )

    np.testing.assert_allclose(
        independent.transition_matrix,
        np.kron(parent.transition_matrix, child.transition_matrix),
    )
    # Source (parent=0, child=0), destination (parent=1, child=1).
    expected_coupled = parent.transition_matrix[0, 1] * conditional[1, 0, 1]
    assert coupled.transition_matrix[0, 3] == pytest.approx(expected_coupled)
    np.testing.assert_allclose(
        halfway.transition_matrix,
        0.5 * independent.transition_matrix + 0.5 * coupled.transition_matrix,
    )
    np.testing.assert_allclose(coupled.transition_matrix.sum(axis=1), 1.0)


def test_factored_import_path_factory_and_probability_helpers():
    model = factored_model(
        factors=[
            {"factory": f"{__name__}:tiny_model_factory"},
            {"factory": f"{__name__}:other_tiny_model_factory"},
        ]
    )
    joint = np.stack(
        [
            model.initial_distribution,
            np.array([0.1, 0.2, 0.3, 0.4]),
        ]
    )
    marginals = factor_marginals(joint, (2, 2))

    np.testing.assert_allclose(marginals[0][1], [0.3, 0.7])
    np.testing.assert_allclose(marginals[1][1], [0.4, 0.6])
    np.testing.assert_allclose(
        product_distribution(marginals)[0],
        model.initial_distribution,
    )


def test_factored_factory_rejects_invalid_graph_and_mapping():
    factors = [tiny_model_factory(), other_tiny_model_factory()]
    conditional = np.repeat(np.eye(2)[None, :, :], 2, axis=0)

    with pytest.raises(TypeError, match="parent must be an integer"):
        FactorCoupling(
            parent=0.5,
            child=1,
            transition_matrices=conditional,
        )
    with pytest.raises(ValueError, match="parent index < child index"):
        compose_hmm_factors(
            factors,
            couplings=[
                {
                    "parent": 1,
                    "child": 0,
                    "transition_matrices": conditional,
                }
            ],
        )
    with pytest.raises(ValueError, match="contiguous"):
        compose_hmm_factors(factors, token_map=[[0, 2], [2, 0]])


def test_belief_tracker_delay_zero_order_is_predict_then_measure():
    model = tiny_model_factory()
    tracker = BeliefTracker(model.initial_distribution)
    tracker.reset(0, likelihood=model.emission_matrix)
    tracker.predict(model.transition_matrix)
    tracker.measure(1, model.emission_matrix)

    expected = measure(
        predict(
            measure(
                model.initial_distribution,
                model.emission_matrix,
                0,
            ),
            model.transition_matrix,
        ),
        model.emission_matrix,
        1,
    )
    np.testing.assert_allclose(tracker.belief, expected)


def test_belief_tracker_delay_one_order_is_measure_then_predict():
    model = tiny_model_factory()
    tracker = BeliefTracker(model.initial_distribution)
    tracker.reset()
    tracker.measure(0, model.emission_matrix)
    tracker.predict(model.transition_matrix)

    expected = predict(
        measure(
            model.initial_distribution,
            model.emission_matrix,
            0,
        ),
        model.transition_matrix,
    )
    np.testing.assert_allclose(tracker.belief, expected)


def test_env_reset_and_step_have_explicit_timing(make_env):
    env = make_env(diagnostics=FULL_DIAGNOSTICS)
    observation, reset_info = env.reset(seed=17)
    np.testing.assert_array_equal(observation, np.zeros(4, dtype=np.float32))
    assert reset_info["decision_step"] == 0
    assert reset_info["visible_token_current"] is None

    state_before = reset_info["state_current"]
    raw_token_before = reset_info["raw_token_current"]
    observation, reward, terminated, truncated, info = env.step(state_before)

    assert not terminated and not truncated
    assert reward == 1.0
    assert info["reward_components"] == {"pre_transition_accuracy": 1.0}
    assert info["transition_step"] == 0
    assert info["decision_step"] == 1
    assert info["state_before"] == state_before
    assert info["state_after"] == info["state_current"]
    assert info["raw_token_before"] == raw_token_before
    assert info["raw_token_after"] == info["raw_token_current"]
    assert info["visible_source_token"] == raw_token_before
    assert np.argmax(observation[:2]) == info["visible_token_current"]
    assert np.argmax(observation[2:]) == state_before
    np.testing.assert_allclose(
        info["original_transition_matrix"],
        tiny_model_factory().transition_matrix,
    )
    np.testing.assert_allclose(
        info["executed_transition_matrix"],
        info["original_transition_matrix"],
    )


def test_env_is_deterministic_given_reset_seed(make_env):
    def trace() -> list[tuple[int, int, int | None, float]]:
        env = make_env(diagnostics=FULL_DIAGNOSTICS)
        _, info = env.reset(seed=29)
        output = []
        for step in range(100):
            _, reward, _, _, info = env.step(step % 2)
            output.append(
                (
                    info["state_current"],
                    info["raw_token_current"],
                    info["visible_token_current"],
                    reward,
                )
            )
        return output

    assert trace() == trace()


def test_only_first_episode_length_can_be_randomized(make_env):
    episode_length = 31
    env = make_env(
        episode_length=episode_length,
        randomize_first_episode_length=True,
    )

    def run_episode(*, seed: int | None = None) -> int:
        env.reset(seed=seed)
        for length in range(1, episode_length + 1):
            _, _, _, truncated, _ = env.step(0)
            if truncated:
                return length
        raise AssertionError("episode did not truncate")

    # RLlib may reset once to apply a worker seed before sampling starts.
    env.reset(seed=17)
    first_length = run_episode()
    assert 1 <= first_length <= episode_length
    assert first_length != episode_length
    assert run_episode() == episode_length
    assert run_episode() == episode_length


def test_first_episode_length_randomization_is_seeded_and_rng_isolated(make_env):
    def trace(randomize: bool):
        env = make_env(
            episode_length=97,
            randomize_first_episode_length=randomize,
            diagnostics=FULL_DIAGNOSTICS,
        )
        _, info = env.reset(seed=23)
        output = []
        for step in range(97):
            action = step % 2
            _, reward, _, truncated, info = env.step(action)
            output.append(
                (
                    info["state_current"],
                    info["raw_token_current"],
                    info["visible_token_current"],
                    reward,
                )
            )
            if truncated:
                break
        return output

    randomized = trace(True)
    assert randomized == trace(True)
    assert randomized == trace(False)[: len(randomized)]


def test_first_episode_length_randomization_requires_bool(make_env):
    with pytest.raises(TypeError, match="randomize_first_episode_length"):
        make_env(randomize_first_episode_length=1)


def test_presentation_scrambling_does_not_change_latent_path(make_env):
    def run(mode: str):
        env = make_env(
            delay=0,
            observation={
                "token": {"offset": 0, "depth": 1},
                "action": None,
                "token_scrambling": mode,
            },
            diagnostics=FULL_DIAGNOSTICS,
        )
        _, info = env.reset(seed=31)
        latent, visible, raw_beliefs = [], [], []
        for step in range(200):
            latent.append(
                (
                    info["state_current"],
                    info["raw_token_current"],
                    info["visible_source_token"],
                )
            )
            visible.append(info["visible_token_current"])
            raw_beliefs.append(info["raw_belief_current"])
            _, _, _, _, info = env.step(step % 2)
        return latent, visible, raw_beliefs

    plain_latent, plain_visible, plain_raw_beliefs = run("none")
    scrambled_latent, scrambled_visible, scrambled_raw_beliefs = run("uniform")
    assert plain_latent == scrambled_latent
    np.testing.assert_allclose(plain_raw_beliefs, scrambled_raw_beliefs)
    assert plain_visible != scrambled_visible
    assert plain_visible == [source for _, _, source in plain_latent]


def test_diagnostics_are_opt_in(make_env):
    env = make_env()
    _, reset_info = env.reset(seed=41)
    assert reset_info == {"decision_step": 0}
    _, _, _, _, step_info = env.step(0)
    assert step_info == {"decision_step": 1}
