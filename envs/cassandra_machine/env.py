"""Gymnasium environment for Cassandra's factored machine-maintenance POMDP."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from envs.cassandra_machine.model import (
    ACTION_COST,
    ACTION_NAMES,
    COMPONENT_TRANSITIONS,
    INSPECTION_POSITIVE_PROBABILITY,
    N_COMPONENTS,
    N_CONDITIONS,
    N_OBSERVATIONS,
    Action,
    Condition,
    decode_observation,
    encode_observation,
    encode_state,
    OPERATE_COMPONENT_REWARD,
)


_RNG_STREAM_KEYS = {
    "transition": (0,),
    "observation": (1,),
}
_OBSERVATION_MODES = {"symbol", "factored_belief"}


@dataclass(frozen=True, slots=True)
class CassandraMachineConfig:
    """Validated simulation options for :class:`CassandraMachineEnv`."""

    episode_length: int = 1000
    observation_mode: str = "symbol"
    diagnostics: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if self.observation_mode not in _OBSERVATION_MODES:
            raise ValueError(
                "observation_mode must be 'symbol' or 'factored_belief'"
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

    ``observation_mode="symbol"`` returns the original 16-valued POMDP symbol.
    ``"factored_belief"`` returns the exact four-by-four marginal belief used
    by factored-belief agents, flattened component-major to 16 values.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Mapping[str, Any] | CassandraMachineConfig | None = None,
    ) -> None:
        self.config = CassandraMachineConfig.from_value(config)
        self.action_space = gym.spaces.Discrete(len(Action))
        if self.config.observation_mode == "symbol":
            self.observation_space = gym.spaces.Discrete(N_OBSERVATIONS)
        else:
            self.observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(N_COMPONENTS * N_CONDITIONS,),
                dtype=np.float32,
            )

        self._transition_rng: np.random.Generator
        self._observation_rng: np.random.Generator
        self._seed(self.config.seed)
        self._components = np.full(
            N_COMPONENTS,
            int(Condition.GOOD),
            dtype=np.int8,
        )
        self._belief = np.zeros((N_COMPONENTS, N_CONDITIONS), dtype=np.float64)
        self._belief[:, Condition.GOOD] = 1.0
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

    @property
    def component_states(self) -> np.ndarray:
        """Return a copy of the current privileged component conditions."""

        return self._components.copy()

    @property
    def factored_belief(self) -> np.ndarray:
        """Return a copy of the exact agent-conditioned component marginals."""

        return self._belief.copy()

    def _policy_observation(self) -> int | np.ndarray:
        if self.config.observation_mode == "symbol":
            return int(self._observation_symbol)
        return self._belief.astype(np.float32, copy=True).reshape(-1)

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
                    "factored_belief_current": self._belief.copy(),
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

        self._components.fill(int(Condition.GOOD))
        self._belief.fill(0.0)
        self._belief[:, Condition.GOOD] = 1.0
        self._observation_symbol = 0
        self._step = 0
        self._initialized = True
        self._needs_reset = False
        return self._policy_observation(), self._info()

    @staticmethod
    def _validate_action(action: Any) -> int:
        if isinstance(action, np.ndarray):
            if action.shape != ():
                raise ValueError("action must be a scalar")
            action = action.item()
        if not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        action_index = int(action)
        if not 0 <= action_index < len(Action):
            raise ValueError("invalid machine-maintenance action")
        return action_index

    def _sample_transition(self, action: int) -> None:
        rows = COMPONENT_TRANSITIONS[action, self._components]
        cumulative = np.cumsum(rows, axis=1)
        draws = self._transition_rng.random(N_COMPONENTS)
        self._components = np.minimum(
            (draws[:, None] > cumulative).sum(axis=1),
            N_CONDITIONS - 1,
        ).astype(np.int8)

    def _sample_observation(self, action: int) -> int:
        if action != Action.INSPECT:
            return 0
        probabilities = INSPECTION_POSITIVE_PROBABILITY[self._components]
        bits = self._observation_rng.random(N_COMPONENTS) < probabilities
        return encode_observation(bits)

    def _advance_belief(self, action: int, observation: int) -> None:
        transition = COMPONENT_TRANSITIONS[action]
        prior = self._belief @ transition
        if action != Action.INSPECT:
            self._belief = prior
            return

        bits = decode_observation(observation)
        likelihood = np.where(
            bits[:, None] == 1,
            INSPECTION_POSITIVE_PROBABILITY[None, :],
            1.0 - INSPECTION_POSITIVE_PROBABILITY[None, :],
        )
        posterior = prior * likelihood
        self._belief = posterior / posterior.sum(axis=1, keepdims=True)

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
            reward = float(ACTION_COST[action_index])

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
                "action_name": ACTION_NAMES[action_index],
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
