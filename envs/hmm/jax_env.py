"""Pure-JAX :class:`envs.hmm.HMMEnv` for edge-emitting, action-independent HMMs.

Covers the configuration where actions never change the hidden dynamics (for
example next-token guessing): token observation depth 1, no action or reward
features, no token scrambling, fixed episode length, delay 0 or 1. The task's
reward is a pure ``reward_fn(action, raw_token_before)`` supplied by the
experiment. Semantics match ``HMMEnv``:

- ``reset`` samples ``s0`` from the initial distribution and the first edge
  ``(x0, s1)``. With ``delay=1`` the first observation is all zeros (BOS);
  with ``delay=0`` it is ``one_hot(x0)``.
- ``step(action)`` scores ``action`` with ``reward_fn`` against the pending
  raw token ``raw_token_before``, samples the next edge, and returns the newly visible
  token: ``raw_token_before`` for ``delay=1``, the new token for ``delay=0``.
- Episode caps are truncations (``terminated`` is always False).

All functions are pure in a PRNG key and vmap/jit cleanly. ``sample_tokens``
and ``predictive_distributions`` give the matching passive token stream and
its exact Bayesian next-token distributions for supervised training.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from envs.hmm.model import HMMModel


class EnvState(NamedTuple):
    hidden_state: jax.Array  # () int32, state after the pending edge
    raw_token: jax.Array  # () int32, pending token the next action scores
    step: jax.Array  # () int32


class StepOutput(NamedTuple):
    obs: jax.Array
    state: EnvState
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    final_obs: jax.Array  # observation before autoreset (== obs otherwise)
    raw_token_before: jax.Array


@dataclass(frozen=True, eq=False)
class JaxHMMEnv:
    """Static description of an action-independent edge-emitting HMM env."""

    initial_distribution: jax.Array  # (S,)
    edge_transition_matrices: jax.Array  # (K, S, S): token, source, dest
    episode_length: int
    reward_fn: Callable[[jax.Array, jax.Array], jax.Array]
    delay: int = 1

    def __post_init__(self) -> None:
        if self.delay not in (0, 1):
            raise ValueError("delay must be 0 or 1")
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if self.edge_transition_matrices.ndim != 3:
            raise ValueError("edge_transition_matrices must be (K, S, S)")

    @classmethod
    def from_model(
        cls,
        model: HMMModel,
        *,
        episode_length: int,
        reward_fn: Callable[[jax.Array, jax.Array], jax.Array],
        delay: int = 1,
    ) -> JaxHMMEnv:
        if model.edge_transition_matrices is None:
            raise ValueError("JaxHMMEnv requires an edge-emitting HMMModel")
        return cls(
            initial_distribution=jnp.asarray(
                model.initial_distribution, jnp.float32
            ),
            edge_transition_matrices=jnp.asarray(
                model.edge_transition_matrices, jnp.float32
            ),
            episode_length=int(episode_length),
            reward_fn=reward_fn,
            delay=int(delay),
        )

    @property
    def n_tokens(self) -> int:
        return int(self.edge_transition_matrices.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.edge_transition_matrices.shape[1])

    @property
    def obs_dim(self) -> int:
        return self.n_tokens

    @property
    def num_actions(self) -> int:
        return self.n_tokens

    # -- dynamics ---------------------------------------------------------

    def _sample_edge(
        self, key: jax.Array, state: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Sample ``(token, next_state)`` from the edges leaving ``state``."""

        probabilities = self.edge_transition_matrices[:, state, :].reshape(-1)
        index = jax.random.categorical(key, jnp.log(probabilities))
        return (index // self.n_states).astype(jnp.int32), (
            index % self.n_states
        ).astype(jnp.int32)

    def _one_hot(self, token: jax.Array) -> jax.Array:
        return jax.nn.one_hot(token, self.n_tokens, dtype=jnp.float32)

    def reset(self, key: jax.Array) -> tuple[jax.Array, EnvState]:
        state_key, edge_key = jax.random.split(key)
        s0 = jax.random.categorical(
            state_key, jnp.log(self.initial_distribution)
        ).astype(jnp.int32)
        token, s1 = self._sample_edge(edge_key, s0)
        obs = (
            jnp.zeros(self.n_tokens, jnp.float32)
            if self.delay == 1
            else self._one_hot(token)
        )
        return obs, EnvState(s1, token, jnp.int32(0))

    def step(
        self, key: jax.Array, state: EnvState, action: jax.Array
    ) -> StepOutput:
        raw_token_before = state.raw_token
        reward = jnp.asarray(
            self.reward_fn(action, raw_token_before), jnp.float32
        )
        token, next_state = self._sample_edge(key, state.hidden_state)
        visible = raw_token_before if self.delay == 1 else token
        obs = self._one_hot(visible)
        step = state.step + 1
        truncated = step >= self.episode_length
        return StepOutput(
            obs=obs,
            state=EnvState(next_state, token, step),
            reward=reward,
            terminated=jnp.bool_(False),
            truncated=truncated,
            final_obs=obs,
            raw_token_before=raw_token_before,
        )

    def step_autoreset(
        self, key: jax.Array, state: EnvState, action: jax.Array
    ) -> StepOutput:
        step_key, reset_key = jax.random.split(key)
        out = self.step(step_key, state, action)
        reset_obs, reset_state = self.reset(reset_key)
        done = out.truncated
        return out._replace(
            obs=jnp.where(done, reset_obs, out.obs),
            state=jax.tree.map(
                lambda fresh, cont: jnp.where(done, fresh, cont),
                reset_state,
                out.state,
            ),
        )

    # -- passive token stream ---------------------------------------------

    def sample_tokens(
        self, key: jax.Array, batch_size: int, length: int
    ) -> jax.Array:
        """Raw token streams ``x_0 .. x_{length-1}``, shape ``(batch, length)``.

        Same law as the ``raw_token_before`` sequence of one episode.
        """

        def one(sequence_key):
            state_key, edge_key = jax.random.split(sequence_key)
            s0 = jax.random.categorical(
                state_key, jnp.log(self.initial_distribution)
            ).astype(jnp.int32)

            def body(state, step_key):
                token, next_state = self._sample_edge(step_key, state)
                return next_state, token

            _, tokens = jax.lax.scan(
                body, s0, jax.random.split(edge_key, length)
            )
            return tokens

        return jax.vmap(one)(jax.random.split(key, batch_size))

    def predictive_distributions(self, tokens: jax.Array) -> jax.Array:
        """Exact ``P(x_t | x_<t)`` for ``tokens`` (..., L); shape (..., L, K)."""

        edges = self.edge_transition_matrices
        token_given_state = edges.sum(-1).T  # (S, K)

        def one(sequence):
            def body(belief, token):
                predictive = belief @ token_given_state
                belief = belief @ edges[token]
                return belief / belief.sum(), predictive

            _, predictive = jax.lax.scan(
                body, self.initial_distribution, sequence
            )
            return predictive

        flat = tokens.reshape(-1, tokens.shape[-1])
        return jax.vmap(one)(flat).reshape(*tokens.shape, self.n_tokens)
