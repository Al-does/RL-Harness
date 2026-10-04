from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from envs.hmm import ActionDecision, HMMEnv, HMMModel, TransitionEvent  # noqa: E402
from envs.hmm.jax_env import JaxHMMEnv  # noqa: E402

EPISODE_LENGTH = 6


def token_guess_reward(action, raw_token_before):
    return action == raw_token_before


def make_jax_env(model, *, episode_length=EPISODE_LENGTH, delay=1):
    return JaxHMMEnv.from_model(
        model,
        episode_length=episode_length,
        reward_fn=token_guess_reward,
        delay=delay,
    )


def asymmetric_edge_model() -> HMMModel:
    rng = np.random.default_rng(3)
    edges = rng.dirichlet(np.ones(2 * 3), size=3).reshape(3, 2, 3)
    edges = edges.transpose(1, 0, 2)  # (token, source, destination)
    transition = edges.sum(axis=0)
    emission = edges.sum(axis=2).T
    return HMMModel(
        initial_distribution=np.array([0.5, 0.3, 0.2]),
        transition_matrix=transition,
        emission_matrix=emission,
        edge_transition_matrices=edges,
    )


class TokenGuessTask:
    requires_belief = False

    def __init__(self, *, model: HMMModel) -> None:
        self.action_space = gym.spaces.Discrete(model.n_tokens)
        self.action_observation_space = gym.spaces.Box(
            0.0, 1.0, (model.n_tokens,), np.float32
        )
        self._transition = model.transition_matrix

    def reset(self) -> None:
        pass

    def resolve_action(self, action, state, model) -> ActionDecision:
        return ActionDecision(int(action), int(action), self._transition)

    def reward(self, event: TransitionEvent, decision: ActionDecision):
        correct = float(decision.executed_action == event.raw_token_before)
        return correct, {}

    def encode_action(self, executed_action: int) -> np.ndarray:
        return np.zeros(self.action_space.n, np.float32)


def numpy_env(delay: int) -> HMMEnv:
    return HMMEnv(
        {
            "model": {"factory": f"{__name__}:asymmetric_edge_model"},
            "task": {"class": f"{__name__}:TokenGuessTask"},
            "observation": {"token": {"depth": 1}, "action": None},
            "delay": delay,
            "episode_length": EPISODE_LENGTH,
            "randomize_first_episode_length": False,
        }
    )


def exact_token_marginals(model: HMMModel, length: int) -> np.ndarray:
    """P(x_t = k) for the raw token stream."""

    state = model.initial_distribution
    rows = []
    for _ in range(length):
        rows.append(state @ model.emission_matrix)
        state = state @ model.transition_matrix
    return np.array(rows)


def jax_rollout(env: JaxHMMEnv, episodes: int, actions: np.ndarray):
    def episode(key):
        reset_key, key = jax.random.split(key)
        obs, state = env.reset(reset_key)

        def body(carry, inputs):
            state = carry
            step_key, action = inputs
            out = env.step(step_key, state, action)
            return out.state, (out.obs, out.reward, out.truncated, out.raw_token_before)

        _, traj = jax.lax.scan(
            body,
            state,
            (jax.random.split(key, EPISODE_LENGTH), jnp.asarray(actions)),
        )
        return obs, traj

    return jax.jit(jax.vmap(episode))(jax.random.split(jax.random.key(0), episodes))


@pytest.mark.parametrize("delay", [0, 1])
def test_observation_and_reward_timing_matches_hmm_env(delay):
    model = asymmetric_edge_model()
    env = make_jax_env(model, delay=delay)
    actions = np.array([0, 1, 0, 1, 1, 0])
    first_obs, (obs, reward, truncated, raw) = jax_rollout(env, 64, actions)
    first_obs, obs, reward, truncated, raw = map(
        np.asarray, (first_obs, obs, reward, truncated, raw)
    )
    np.testing.assert_array_equal(reward, (actions[None] == raw).astype(np.float32))
    np.testing.assert_array_equal(truncated[:, -1], True)
    np.testing.assert_array_equal(truncated[:, :-1], False)
    if delay == 1:
        np.testing.assert_array_equal(first_obs, 0.0)
        np.testing.assert_array_equal(obs.argmax(-1), raw)
    else:
        np.testing.assert_array_equal(first_obs.argmax(-1), raw[:, 0])
        np.testing.assert_array_equal(obs[:, :-1].argmax(-1), raw[:, 1:])

    reference = numpy_env(delay)
    reference_obs, _ = reference.reset(seed=0)
    assert reference_obs.shape == first_obs.shape[1:]
    if delay == 1:
        np.testing.assert_array_equal(reference_obs, 0.0)
    for action in actions:
        _, _, terminated, step_truncated, _ = reference.step(int(action))
        assert not terminated
    assert step_truncated


def test_token_statistics_match_numpy_env_and_exact_marginals():
    model = asymmetric_edge_model()
    env = make_jax_env(model)
    episodes = 20_000
    zeros = np.zeros(EPISODE_LENGTH, np.int32)
    _, (_, reward, _, _) = jax_rollout(env, episodes, zeros)
    jax_rate = np.asarray(reward).mean(0)

    reference = numpy_env(1)
    rewards = np.empty((4_000, EPISODE_LENGTH))
    for episode in range(len(rewards)):
        reference.reset(seed=episode)
        for t in range(EPISODE_LENGTH):
            rewards[episode, t] = reference.step(0)[1]
    numpy_rate = rewards.mean(0)

    exact = exact_token_marginals(model, EPISODE_LENGTH)[:, 0]
    np.testing.assert_allclose(jax_rate, exact, atol=4 * 0.5 / np.sqrt(episodes))
    np.testing.assert_allclose(numpy_rate, exact, atol=4 * 0.5 / np.sqrt(len(rewards)))


def test_sample_tokens_matches_episode_token_law():
    model = asymmetric_edge_model()
    env = make_jax_env(model)
    tokens = np.asarray(env.sample_tokens(jax.random.key(1), 20_000, EPISODE_LENGTH))
    exact = exact_token_marginals(model, EPISODE_LENGTH)[:, 1]
    np.testing.assert_allclose(
        (tokens == 1).mean(0), exact, atol=4 * 0.5 / np.sqrt(len(tokens))
    )


def test_predictive_distributions_match_numpy_filter():
    model = asymmetric_edge_model()
    env = make_jax_env(model)
    tokens = np.asarray(env.sample_tokens(jax.random.key(2), 16, 10))
    predicted = np.asarray(env.predictive_distributions(jnp.asarray(tokens)))
    edges = model.edge_transition_matrices
    for row, sequence in enumerate(tokens):
        belief = model.initial_distribution
        for t, token in enumerate(sequence):
            np.testing.assert_allclose(
                predicted[row, t], belief @ model.emission_matrix, atol=1e-5
            )
            belief = belief @ edges[token]
            belief = belief / belief.sum()


def test_step_autoreset_restarts_after_truncation():
    model = asymmetric_edge_model()
    env = make_jax_env(model, episode_length=2)
    _, state = env.reset(jax.random.key(0))
    out = env.step_autoreset(jax.random.key(1), state, jnp.int32(0))
    assert not bool(out.truncated) and int(out.state.step) == 1
    out = env.step_autoreset(jax.random.key(2), out.state, jnp.int32(0))
    assert bool(out.truncated) and int(out.state.step) == 0
    np.testing.assert_array_equal(np.asarray(out.obs), 0.0)
    assert float(np.asarray(out.final_obs).sum()) == 1.0
