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
    HMMEnv,
    HMMModel,
    TransitionEvent,
    measure,
    predict,
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
