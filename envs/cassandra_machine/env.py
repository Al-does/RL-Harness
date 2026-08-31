"""Gymnasium environment for Cassandra's factored machine-maintenance POMDP."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from envs.cassandra_machine.model import (
    ACTION_COST,
    COMPONENT_TRANSITIONS,
    GLOBAL_ALIAS_ACTION_COST,
    GLOBAL_ALIAS_COMPONENT_TRANSITIONS,
    INSPECTION_POSITIVE_PROBABILITY,
    N_COMPONENTS,
    N_CONDITIONS,
    N_OBSERVATIONS,
    N_STATES,
    OPERATE_COMPONENT_PASS_PROBABILITY,
    Action,
    Condition,
    decode_observation,
    decode_state,
    encode_observation,
    encode_state,
    OPERATE_COMPONENT_REWARD,
    TARGETED_ACTION_COST,
    TARGETED_COMPONENT_TRANSITIONS,
    action_names,
)


_RNG_STREAM_KEYS = {
    "transition": (0,),
    "observation": (1,),
    "initial_state": (2,),
}
_OBSERVATION_MODES = {"symbol", "state", "belief", "factored_belief"}
_INITIAL_STATE_DISTRIBUTIONS = {"all_good", "uniform"}
_STATE_COMPONENTS = np.stack(
    [decode_state(state) for state in range(N_STATES)]
)


@dataclass(frozen=True, slots=True)
class CassandraMachineConfig:
    """Validated simulation options for :class:`CassandraMachineEnv`."""

    episode_length: int = 1000
    observation_mode: str = "symbol"
    action_scope: str = "global"
    initial_state_distribution: str = "all_good"
    diagnostics: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if self.observation_mode not in _OBSERVATION_MODES:
            raise ValueError(
                "observation_mode must be 'symbol', 'state', 'belief', or "
                "'factored_belief'"
            )
        if self.action_scope not in {
            "global",
            "global_aliases",
            "targeted",
        }:
            raise ValueError(
                "action_scope must be 'global', 'global_aliases', or "
                "'targeted'"
            )
        if self.initial_state_distribution not in _INITIAL_STATE_DISTRIBUTIONS:
            raise ValueError(
                "initial_state_distribution must be 'all_good' or 'uniform'"
            )
        if not isinstance(self.diagnostics, bool):
            raise TypeError("diagnostics must be a bool")

    @classmethod
    def from_value(
        cls,
        value: Mapping[str, Any] | CassandraMachineConfig | None,
    ) -> CassandraMachineConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(**dict(value))


class CassandraMachineEnv(gym.Env):
    """The canonical four-component machine maintenance benchmark.

    The hidden state consists of four independent component conditions. The
    ``operate`` action degrades them and earns a condition-dependent product
    reward; ``inspect`` emits four noisy binary readings; ``repair`` improves
    non-broken components; and ``replace`` restores every component to good.

    ``action_scope="global"`` preserves the canonical four actions.
    ``action_scope="global_aliases"`` exposes four exact aliases of global
    repair and four exact aliases of global replacement.
    ``action_scope="targeted"`` replaces global repair and replacement with
    four component-addressable repair and four component-addressable
    replacement actions.

    ``observation_mode="symbol"`` returns the original 16-valued POMDP symbol.
    ``"state"`` returns the fully observable current 256-valued joint state.
    ``"belief"`` returns the exact 256-state Bayesian belief.
    ``"factored_belief"`` returns its exact four-by-four component marginals,
    flattened component-major to 16 values.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Mapping[str, Any] | CassandraMachineConfig | None = None,
    ) -> None:
        self.config = CassandraMachineConfig.from_value(config)
        self._action_names = action_names(self.config.action_scope)
        if self.config.action_scope == "global":
            self._action_costs = ACTION_COST
            self._component_transitions = np.broadcast_to(
                COMPONENT_TRANSITIONS[:, None, :, :],
                (
                    len(Action),
                    N_COMPONENTS,
                    N_CONDITIONS,
                    N_CONDITIONS,
                ),
            )
        elif self.config.action_scope == "global_aliases":
            self._action_costs = GLOBAL_ALIAS_ACTION_COST
            self._component_transitions = GLOBAL_ALIAS_COMPONENT_TRANSITIONS
        else:
            self._action_costs = TARGETED_ACTION_COST
            self._component_transitions = TARGETED_COMPONENT_TRANSITIONS
        self.action_space = gym.spaces.Discrete(len(self._action_names))
        if self.config.observation_mode == "symbol":
            self.observation_space = gym.spaces.Discrete(N_OBSERVATIONS)
        elif self.config.observation_mode == "state":
            self.observation_space = gym.spaces.Discrete(N_STATES)
        elif self.config.observation_mode == "belief":
            self.observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(N_STATES,),
                dtype=np.float32,
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(N_COMPONENTS * N_CONDITIONS,),
                dtype=np.float32,
            )

        self._transition_rng: np.random.Generator
        self._observation_rng: np.random.Generator
        self._initial_state_rng: np.random.Generator
        self._seed(self.config.seed)
        self._components = np.full(
            N_COMPONENTS,
            int(Condition.GOOD),
            dtype=np.int8,
        )
        self._belief = np.zeros(N_STATES, dtype=np.float64)
        self._belief[N_STATES - 1] = 1.0
        self._factored_belief = np.zeros(
            (N_COMPONENTS, N_CONDITIONS),
            dtype=np.float64,
        )
        self._factored_belief[:, Condition.GOOD] = 1.0
        self._observation_symbol = 0
        self._step = 0
        self._initialized = False
        self._needs_reset = True

    def _seed(self, seed: int | None) -> None:
        root = np.random.SeedSequence(seed)
        streams = {
            name: np.random.SeedSequence(
                root.entropy,
                spawn_key=(*root.spawn_key, *key),
                pool_size=root.pool_size,
            )
            for name, key in _RNG_STREAM_KEYS.items()
        }
        self._transition_rng = np.random.default_rng(streams["transition"])
        self._observation_rng = np.random.default_rng(streams["observation"])
        self._initial_state_rng = np.random.default_rng(
            streams["initial_state"]
        )

    @property
    def component_states(self) -> np.ndarray:
        """Return a copy of the current privileged component conditions."""

        return self._components.copy()

    @property
    def factored_belief(self) -> np.ndarray:
        """Return a copy of the exact agent-conditioned component marginals."""

        return self._factored_belief.copy()

    @property
    def belief(self) -> np.ndarray:
        """Return a copy of the exact 256-state Bayesian belief."""

        return self._belief.copy()

    def _policy_observation(self) -> int | np.ndarray:
        if self.config.observation_mode == "symbol":
            return int(self._observation_symbol)
        if self.config.observation_mode == "state":
            return encode_state(self._components)
        if self.config.observation_mode == "belief":
            return self._belief.astype(np.float32, copy=True)
        return self._factored_belief.astype(np.float32, copy=True).reshape(-1)

    def _info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "decision_step": self._step,
            "observation_symbol": self._observation_symbol,
        }
        if self.config.diagnostics:
            info.update(
                {
                    "state_current": encode_state(self._components),
                    "components_current": self._components.copy(),
                    "belief_current": self._belief.copy(),
                    "factored_belief_current": self._factored_belief.copy(),
                }
            )
        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[int | np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            self._seed(seed)
            self.action_space.seed(seed)

        if self.config.initial_state_distribution == "uniform":
            self._components = self._initial_state_rng.integers(
                0,
                N_CONDITIONS,
                size=N_COMPONENTS,
                dtype=np.int8,
            )
            self._belief.fill(1.0 / N_STATES)
            self._factored_belief.fill(1.0 / N_CONDITIONS)
        else:
            self._components.fill(int(Condition.GOOD))
            self._belief.fill(0.0)
            self._belief[N_STATES - 1] = 1.0
            self._factored_belief.fill(0.0)
            self._factored_belief[:, Condition.GOOD] = 1.0
        self._observation_symbol = 0
        self._step = 0
        self._initialized = True
        self._needs_reset = False
        return self._policy_observation(), self._info()

    def _validate_action(self, action: Any) -> int:
        if isinstance(action, np.ndarray):
            if action.shape != ():
                raise ValueError("action must be a scalar")
            action = action.item()
        if not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        action_index = int(action)
        if not 0 <= action_index < self.action_space.n:
            raise ValueError("invalid machine-maintenance action")
        return action_index

    def _sample_transition(self, action: int) -> None:
        rows = self._component_transitions[action][
            np.arange(N_COMPONENTS),
            self._components,
        ]
        cumulative = np.cumsum(rows, axis=1)
        draws = self._transition_rng.random(N_COMPONENTS)
        self._components = np.minimum(
            (draws[:, None] > cumulative).sum(axis=1),
            N_CONDITIONS - 1,
        ).astype(np.int8)

    def _sample_observation(self, action: int) -> int:
        if action == Action.OPERATE:
            pass_probability = float(
                np.prod(
                    OPERATE_COMPONENT_PASS_PROBABILITY[self._components]
                )
            )
            passed = self._observation_rng.random() < pass_probability
            return (N_OBSERVATIONS - 1) if passed else 0
        if action == Action.INSPECT:
            probabilities = INSPECTION_POSITIVE_PROBABILITY[self._components]
            bits = self._observation_rng.random(N_COMPONENTS) < probabilities
            return encode_observation(bits)
        return 0

    def _advance_belief(self, action: int, observation: int) -> None:
        transitions = self._component_transitions[action]
        prior = self._belief.reshape((N_CONDITIONS,) * N_COMPONENTS)
        for component in range(N_COMPONENTS):
            axis = N_COMPONENTS - 1 - component
            moved = np.moveaxis(prior, axis, -1)
            moved = moved @ transitions[component]
            prior = np.moveaxis(moved, -1, axis)
        prior = prior.reshape(-1)

        if action == Action.OPERATE:
            pass_probability = np.prod(
                OPERATE_COMPONENT_PASS_PROBABILITY[_STATE_COMPONENTS],
                axis=1,
            )
            likelihood = (
                pass_probability
                if observation == N_OBSERVATIONS - 1
                else 1.0 - pass_probability
            )
        elif action == Action.INSPECT:
            bits = decode_observation(observation)
            per_component = np.where(
                bits[None, :] == 1,
                INSPECTION_POSITIVE_PROBABILITY[
                    _STATE_COMPONENTS
                ],
                1.0
                - INSPECTION_POSITIVE_PROBABILITY[
                    _STATE_COMPONENTS
                ],
            )
            likelihood = np.prod(per_component, axis=1)
        else:
            likelihood = 1.0

        posterior = prior * likelihood
        self._belief = posterior / posterior.sum()
        self._factored_belief = np.stack(
            [
                [
                    self._belief[
                        _STATE_COMPONENTS[:, component] == condition
                    ].sum()
                    for condition in range(N_CONDITIONS)
                ]
                for component in range(N_COMPONENTS)
            ]
        )

    def step(
        self,
        action: Any,
    ) -> tuple[int | np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._initialized:
            raise RuntimeError("reset must be called before step")
        if self._needs_reset:
            raise RuntimeError("reset must be called after episode truncation")
        action_index = self._validate_action(action)

        state_before = encode_state(self._components)
        components_before = self._components.copy()
        if action_index == Action.OPERATE:
            reward = float(
                np.prod(OPERATE_COMPONENT_REWARD[components_before])
            )
        else:
            reward = float(self._action_costs[action_index])

        self._sample_transition(action_index)
        self._observation_symbol = self._sample_observation(action_index)
        self._advance_belief(action_index, self._observation_symbol)
        self._step += 1
        truncated = self._step >= self.config.episode_length
        self._needs_reset = truncated

        info = self._info()
        info.update(
            {
                "action": action_index,
                "action_name": self._action_names[action_index],
            }
        )
        if self.config.diagnostics:
            info.update(
                {
                    "state_before": state_before,
                    "components_before": components_before,
                    "state_after": encode_state(self._components),
                    "components_after": self._components.copy(),
                    "reward_components": {
                        (
                            "production_reward"
                            if action_index == Action.OPERATE
                            else "maintenance_cost"
                        ): reward,
                    },
                }
            )
        return self._policy_observation(), reward, False, truncated, info
