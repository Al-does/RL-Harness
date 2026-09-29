"""Gymnasium environment for the RockSample POMDP."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import comb, hypot

import gymnasium as gym
import numpy as np

from envs.rocksample.model import (
    BAD_ROCK_REWARD,
    EXIT_REWARD,
    GOOD_ROCK_REWARD,
    Action,
    InstanceDefinition,
    Observation,
    instance_definition,
    sensor_efficiency,
)


_RNG_STREAM_KEYS = {
    "layout": (0,),
    "qualities": (1,),
    "sensor": (2,),
}


@dataclass(frozen=True, slots=True)
class RockSampleConfig:
    """Validated simulation options for :class:`RockSampleEnv`."""

    n: int = 4
    k: int = 4
    randomize_train_layout: bool = True
    evaluation: bool = False
    episode_length: int = 100
    eval_layout_seed: int = 0
    diagnostics: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if type(self.n) is not int or self.n < 2:
            raise ValueError("n must be an integer of at least 2")
        if type(self.k) is not int or self.k <= 0:
            raise ValueError("k must be a positive integer")
        if self.k > self.n * self.n - 1:
            raise ValueError("k cannot exceed the non-start grid cells")
        for name in ("randomize_train_layout", "evaluation", "diagnostics"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if type(self.episode_length) is not int or self.episode_length <= 0:
            raise ValueError("episode_length must be a positive integer")
        if type(self.eval_layout_seed) is not int:
            raise TypeError("eval_layout_seed must be an int")
        if self.seed is not None and type(self.seed) is not int:
            raise TypeError("seed must be an int or None")
        if (
            self.randomize_train_layout
            and not self.evaluation
            and comb(self.n * self.n - 1, self.k) <= 1
        ):
            raise ValueError(
                "randomized training requires a layout distinct from evaluation"
            )

    @classmethod
    def from_value(
        cls,
        value: Mapping[str, object] | RockSampleConfig | None,
    ) -> RockSampleConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(**dict(value))


class RockSampleEnv(gym.Env[np.ndarray, int]):
    """Configurable RockSample with held-out randomized training layouts.

    The flat policy observation contains the normalized rover position,
    normalized rock positions in rock-index order, and a one-hot encoding of
    the latest ``Good``, ``Bad``, or ``None`` observation symbol followed by
    the previous action. The action vector is all zeros immediately after
    reset.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Mapping[str, object] | RockSampleConfig | None = None,
    ) -> None:
        self.config = RockSampleConfig.from_value(config)
        self.instance: InstanceDefinition = instance_definition(
            self.config.n,
            self.config.k,
            layout_seed=self.config.eval_layout_seed,
        )
        self.action_space = gym.spaces.Discrete(len(Action) + self.config.k)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(
                2
                + 2 * self.config.k
                + len(Observation)
                + self.action_space.n,
            ),
            dtype=np.float32,
        )

        self._layout_rng: np.random.Generator
        self._quality_rng: np.random.Generator
        self._sensor_rng: np.random.Generator
        self._seed(self.config.seed)

        self._rover = self.instance.start
        self._rocks = np.asarray(self.instance.rocks, dtype=np.int64)
        self._rock_by_position: dict[tuple[int, int], int] = {}
        self._qualities = np.zeros(self.config.k, dtype=np.bool_)
        self._observation_symbol = Observation.NONE
        self._previous_action: int | None = None
        self._step = 0
        self._illegal_action_count = 0
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
        self._layout_rng = np.random.default_rng(streams["layout"])
        self._quality_rng = np.random.default_rng(streams["qualities"])
        self._sensor_rng = np.random.default_rng(streams["sensor"])

    @property
    def rover_position(self) -> tuple[int, int]:
        """Return the rover's current known position."""

        return self._rover

    @property
    def rock_positions(self) -> tuple[tuple[int, int], ...]:
        """Return current rock positions in check-action index order."""

        return tuple((int(x), int(y)) for x, y in self._rocks)

    @property
    def rock_qualities(self) -> np.ndarray:
        """Return a copy of the privileged current rock qualities."""

        return self._qualities.copy()

    def _training_layout(self) -> np.ndarray:
        start = self.instance.start
        candidates = np.array(
            [
                (x, y)
                for x in range(self.config.n)
                for y in range(self.config.n)
                if (x, y) != start
            ],
            dtype=np.int64,
        )
        evaluation_cells = frozenset(self.instance.rocks)
        while True:
            indices = self._layout_rng.choice(
                len(candidates),
                size=self.config.k,
                replace=False,
            )
            rocks = candidates[indices]
            if frozenset((int(x), int(y)) for x, y in rocks) != evaluation_cells:
                return rocks

    def _policy_observation(self) -> np.ndarray:
        scale = float(self.config.n - 1)
        observation = np.zeros(
            self.observation_space.shape,
            dtype=np.float32,
        )
        observation[:2] = np.asarray(self._rover, dtype=np.float32) / scale
        observation[2 : 2 + 2 * self.config.k] = (
            self._rocks.astype(np.float32).reshape(-1) / scale
        )
        symbol_offset = 2 + 2 * self.config.k
        observation[symbol_offset + int(self._observation_symbol)] = 1.0
        if self._previous_action is not None:
            action_offset = symbol_offset + len(Observation)
            observation[action_offset + self._previous_action] = 1.0
        return observation

    def _info(self, *, illegal_action: bool) -> dict[str, object]:
        info: dict[str, object] = {
            "decision_step": self._step,
            "observation_symbol": int(self._observation_symbol),
            "rover_position": np.asarray(self._rover, dtype=np.int64),
            "rock_positions": self._rocks.copy(),
            "illegal_action": illegal_action,
            "illegal_action_count": self._illegal_action_count,
            "illegal_action_rate": (
                self._illegal_action_count / self._step if self._step else 0.0
            ),
        }
        if self.config.diagnostics:
            info["rock_qualities"] = self._qualities.copy()
        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            self._seed(seed)
            self.action_space.seed(seed)

        if self.config.evaluation or not self.config.randomize_train_layout:
            self._rocks = np.asarray(self.instance.rocks, dtype=np.int64)
        else:
            self._rocks = self._training_layout()
        self._rock_by_position = {
            (int(x), int(y)): index
            for index, (x, y) in enumerate(self._rocks)
        }
        self._qualities = self._quality_rng.integers(
            0,
            2,
            size=self.config.k,
            dtype=np.int8,
        ).astype(np.bool_)
        self._rover = self.instance.start
        self._observation_symbol = Observation.NONE
        self._previous_action = None
        self._step = 0
        self._illegal_action_count = 0
        self._needs_reset = False
        return self._policy_observation(), self._info(illegal_action=False)

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self._needs_reset:
            raise RuntimeError("reset must be called before step")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")

        action = int(action)
        x, y = self._rover
        reward = 0.0
        terminated = False
        illegal_action = False
        self._observation_symbol = Observation.NONE

        if action == Action.NORTH:
            if y == self.config.n - 1:
                illegal_action = True
            else:
                self._rover = (x, y + 1)
        elif action == Action.SOUTH:
            if y == 0:
                illegal_action = True
            else:
                self._rover = (x, y - 1)
        elif action == Action.EAST:
            if x == self.config.n - 1:
                reward = EXIT_REWARD
                terminated = True
            else:
                self._rover = (x + 1, y)
        elif action == Action.WEST:
            if x == 0:
                illegal_action = True
            else:
                self._rover = (x - 1, y)
        elif action == Action.SAMPLE:
            rock_index = self._rock_by_position.get(self._rover)
            if rock_index is None:
                illegal_action = True
            else:
                reward = (
                    GOOD_ROCK_REWARD
                    if self._qualities[rock_index]
                    else BAD_ROCK_REWARD
                )
                self._qualities[rock_index] = False
        else:
            rock_index = action - len(Action)
            rock_x, rock_y = self._rocks[rock_index]
            distance = hypot(x - int(rock_x), y - int(rock_y))
            efficiency = sensor_efficiency(
                distance,
                self.instance.sensor_half_efficiency_distance,
            )
            correct = self._sensor_rng.random() < (1.0 + efficiency) / 2.0
            reports_good = bool(self._qualities[rock_index]) == correct
            self._observation_symbol = (
                Observation.GOOD if reports_good else Observation.BAD
            )

        self._previous_action = action
        self._step += 1
        if illegal_action:
            self._illegal_action_count += 1
        truncated = self._step >= self.config.episode_length and not terminated
        if terminated or truncated:
            self._needs_reset = True
        return (
            self._policy_observation(),
            reward,
            terminated,
            truncated,
            self._info(illegal_action=illegal_action),
        )
