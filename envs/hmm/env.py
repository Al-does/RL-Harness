"""One reusable Gymnasium environment for finite discrete HMM tasks."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
import importlib
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from envs.hmm.belief import BeliefTracker
from envs.hmm.model import HMMModel


_ENVIRONMENT_STREAM_KEYS = {
    "state": (0,),
    "emission": (1,),
    "presentation": (2,),
    "episode_length": (3,),
}
# Explicit spawn keys are order-independent; never renumber or reuse a key.
# Keys 0..3 match the historical SeedSequence.spawn(4) children.


@dataclass(frozen=True, slots=True)
class HistoryWindow:
    """A contiguous newest-first window over decision records."""

    offset: int = 0
    depth: int = 1

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("history offset must be non-negative")
        if self.depth <= 0:
            raise ValueError("history depth must be positive")

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | HistoryWindow) -> HistoryWindow:
        if isinstance(value, cls):
            return value
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    """Select flat policy features and their history windows.

    Token offset zero is the currently visible token. Action offset zero is
    the action executed immediately before the current decision. Each window
    is emitted newest first; token features precede action features.
    """

    token: HistoryWindow | None = field(default_factory=HistoryWindow)
    action: HistoryWindow | None = field(default_factory=HistoryWindow)
    belief: bool = False
    hidden_state: bool = False
    token_scrambling: str = "none"
    action_scrambling: str = "none"

    def __post_init__(self) -> None:
        if self.token_scrambling not in {"none", "uniform"}:
            raise ValueError(
                "token_scrambling must be either 'none' or 'uniform'"
            )
        if self.action_scrambling not in {"none", "uniform"}:
            raise ValueError(
                "action_scrambling must be either 'none' or 'uniform'"
            )

    @classmethod
    def from_value(
        cls,
        value: Mapping[str, Any] | ObservationConfig | None,
    ) -> ObservationConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        config = dict(value)
        token = config.pop("token", {})
        action = config.pop("action", {})
        return cls(
            token=None if token is None else HistoryWindow.from_value(token),
            action=None if action is None else HistoryWindow.from_value(action),
            **config,
        )


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    """Select privileged values copied into Gymnasium ``info``."""

    state: bool = False
    belief: bool = False
    raw_belief: bool = False
    tokens: bool = False
    rewards: bool = False
    transitions: bool = False

    @classmethod
    def from_value(
        cls,
        value: Mapping[str, Any] | DiagnosticsConfig | None,
    ) -> DiagnosticsConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(**dict(value))

    @classmethod
    def full(cls) -> DiagnosticsConfig:
        return cls(
            state=True,
            belief=True,
            raw_belief=True,
            tokens=True,
            rewards=True,
            transitions=True,
        )


@dataclass(frozen=True, slots=True)
class HMMEnvConfig:
    """Validated top-level construction data for :class:`HMMEnv`."""

    model: Mapping[str, Any]
    task: Mapping[str, Any]
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    delay: int = 0
    episode_length: int = 1024
    randomize_first_episode_length: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.delay not in (0, 1):
            raise ValueError("delay must be 0 or 1")
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if not isinstance(self.randomize_first_episode_length, bool):
            raise TypeError("randomize_first_episode_length must be a bool")
        if not isinstance(self.model, Mapping):
            raise TypeError("model must be a component configuration")
        if not isinstance(self.task, Mapping):
            raise TypeError("task must be a component configuration")

    @classmethod
    def from_value(
        cls,
        value: Mapping[str, Any] | HMMEnvConfig,
    ) -> HMMEnvConfig:
        if isinstance(value, cls):
            return value
        config = dict(value)
        observation = ObservationConfig.from_value(
            config.pop("observation", None)
        )
        diagnostics = DiagnosticsConfig.from_value(
            config.pop("diagnostics", None)
        )
        return cls(
            observation=observation,
            diagnostics=diagnostics,
            **config,
        )


@dataclass(frozen=True, slots=True)
class ActionDecision:
    """A requested action and the transition law it actually executes."""

    requested_action: Any
    executed_action: Any
    transition_matrix: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransitionEvent:
    """One completed transition with explicit before/after timing."""

    step: int
    state_before: int
    state_after: int
    raw_token_before: int
    raw_token_after: int
    belief_before: np.ndarray | None = None
    belief_after: np.ndarray | None = None


class HMMTask(Protocol):
    """Structural task contract consumed by :class:`HMMEnv`."""

    action_space: gym.Space
    action_observation_space: gym.spaces.Box
    requires_belief: bool

    def reset(self) -> None: ...

    def resolve_action(
        self,
        action: Any,
        state: int,
        model: HMMModel,
    ) -> ActionDecision: ...

    def reward(
        self,
        event: TransitionEvent,
        decision: ActionDecision,
    ) -> tuple[float, dict[str, float]]: ...

    def encode_action(self, executed_action: Any) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class _DecisionRecord:
    visible_token: int | None
    action_features: np.ndarray | None
    reward: float | None


def _load_symbol(path: str) -> Any:
    if not isinstance(path, str) or ":" not in path:
        raise ValueError("component paths must use 'package.module:Symbol'")
    module_name, qualified_name = path.split(":", 1)
    if not module_name or not qualified_name:
        raise ValueError("component paths must use 'package.module:Symbol'")
    value: Any = importlib.import_module(module_name)
    for part in qualified_name.split("."):
        value = getattr(value, part)
    return value


def _component(
    config: Mapping[str, Any],
    *,
    path_key: str,
) -> tuple[Any, dict[str, Any]]:
    values = dict(config)
    try:
        path = values.pop(path_key)
    except KeyError as error:
        raise ValueError(f"component config requires {path_key!r}") from error
    kwargs = dict(values.pop("kwargs", {}))
    if values:
        raise ValueError(
            f"unknown component config fields: {sorted(values)}"
        )
    return _load_symbol(path), kwargs


def _one_hot(index: int | None, size: int) -> np.ndarray:
    output = np.zeros(size, dtype=np.float32)
    if index is not None:
        output[index] = 1.0
    return output


def _diagnostic_action(action: Any) -> Any:
    if isinstance(action, np.ndarray):
        return action.copy()
    if isinstance(action, np.generic):
        return action.item()
    return action


class HMMEnv(gym.Env):
    """Simulate one finite HMM with one attached action/reward task."""

    metadata = {"render_modes": []}

    def __init__(self, config: Mapping[str, Any] | HMMEnvConfig):
        self.config = HMMEnvConfig.from_value(config)

        model_factory, model_kwargs = _component(
            self.config.model,
            path_key="factory",
        )
        model = model_factory(**model_kwargs)
        if not isinstance(model, HMMModel):
            raise TypeError("the configured model factory must return HMMModel")
        self.model = model

        task_class, task_kwargs = _component(
            self.config.task,
            path_key="class",
        )
        self.task: HMMTask = task_class(model=self.model, **task_kwargs)
        self._validate_task()
        self._task_requires_belief = bool(
            getattr(self.task, "requires_belief", False)
        )
        self.action_space = self.task.action_space

        self._token_confusion = self._token_confusion_matrix()
        self._visible_likelihood = (
            self.model.emission_matrix @ self._token_confusion
        )

        track_belief = (
            self.config.observation.belief
            or self.config.diagnostics.belief
            or self._task_requires_belief
        )
        self._belief_tracker = (
            BeliefTracker(self.model.initial_distribution)
            if track_belief
            else None
        )
        self._raw_belief_tracker = (
            BeliefTracker(self.model.initial_distribution)
            if self.config.diagnostics.raw_belief
            else None
        )

        self._action_feature_space = self._get_action_feature_space()
        self.observation_space = self._build_observation_space()
        self._history = deque[_DecisionRecord](
            maxlen=self._required_history_length()
        )
        self._raw_token_history = deque[int](
            maxlen=self.config.delay + 1
        )

        self._state_rng: np.random.Generator
        self._emission_rng: np.random.Generator
        self._presentation_rng: np.random.Generator
        self._episode_rng: np.random.Generator
        self._seed(self.config.seed)

        self._state = 0
        self._raw_token = 0
        self._visible_token: int | None = None
        self._visible_source_token: int | None = None
        self._step = 0
        self._current_episode_length = self.config.episode_length
        self._first_episode_pending = True
        self._initialized = False
        self._needs_reset = True

    def _validate_task(self) -> None:
        if not isinstance(self.task.action_space, gym.Space):
            raise TypeError("task.action_space must be a Gymnasium space")
        requires_belief = getattr(self.task, "requires_belief", False)
        if not isinstance(requires_belief, bool):
            raise TypeError("task.requires_belief must be a bool when defined")
        for method_name in (
            "reset",
            "resolve_action",
            "reward",
            "encode_action",
        ):
            if not callable(getattr(self.task, method_name, None)):
                raise TypeError(f"task must define {method_name}()")

    def _get_action_feature_space(self) -> gym.spaces.Box | None:
        if self.config.observation.action is None:
            return None
        space = getattr(self.task, "action_observation_space", None)
        if not isinstance(space, gym.spaces.Box):
            raise TypeError(
                "tasks used with action observations must expose a Box "
                "action_observation_space"
            )
        if len(space.shape) != 1:
            raise ValueError("action_observation_space must be one-dimensional")
        return space

    def _token_confusion_matrix(self) -> np.ndarray:
        n_tokens = self.model.n_tokens
        if self.config.observation.token_scrambling == "none":
            return np.eye(n_tokens)
        return np.full((n_tokens, n_tokens), 1.0 / n_tokens)

    def _build_observation_space(self) -> gym.spaces.Box:
        low: list[np.ndarray] = []
        high: list[np.ndarray] = []
        observation = self.config.observation
        if observation.token is not None:
            token_size = observation.token.depth * self.model.n_tokens
            low.append(np.zeros(token_size))
            high.append(np.ones(token_size))
        if observation.action is not None:
            assert self._action_feature_space is not None
            low.append(
                np.tile(
                    self._action_feature_space.low,
                    observation.action.depth,
                )
            )
            high.append(
                np.tile(
                    self._action_feature_space.high,
                    observation.action.depth,
                )
            )
        if observation.belief:
            low.append(np.zeros(self.model.n_states))
            high.append(np.ones(self.model.n_states))
        if observation.hidden_state:
            low.append(np.zeros(self.model.n_states))
            high.append(np.ones(self.model.n_states))
        if not low:
            raise ValueError("policy observation must contain at least one feature")
        return gym.spaces.Box(
            low=np.concatenate(low).astype(np.float32),
            high=np.concatenate(high).astype(np.float32),
            dtype=np.float32,
        )

    def _required_history_length(self) -> int:
        lengths = [1]
        for window in (
            self.config.observation.token,
            self.config.observation.action,
        ):
            if window is not None:
                lengths.append(window.offset + window.depth)
        return max(lengths)

    def _seed(self, seed: int | None) -> None:
        root = np.random.SeedSequence(seed)
        streams = {
            name: np.random.SeedSequence(
                root.entropy,
                spawn_key=(*root.spawn_key, *key),
                pool_size=root.pool_size,
            )
            for name, key in _ENVIRONMENT_STREAM_KEYS.items()
        }
        self._state_rng = np.random.default_rng(streams["state"])
        self._emission_rng = np.random.default_rng(streams["emission"])
        self._presentation_rng = np.random.default_rng(
            streams["presentation"]
        )
        self._episode_rng = np.random.default_rng(streams["episode_length"])

    def _reset_episode_length(self) -> None:
        if (
            not self._first_episode_pending
            or not self.config.randomize_first_episode_length
        ):
            self._current_episode_length = self.config.episode_length
            return
        self._current_episode_length = int(
            self._episode_rng.integers(1, self.config.episode_length + 1)
        )

    @staticmethod
    def _sample(
        rng: np.random.Generator,
        probabilities: np.ndarray,
    ) -> int:
        index = int(
            np.searchsorted(
                np.cumsum(probabilities),
                rng.random(),
                side="right",
            )
        )
        return min(index, len(probabilities) - 1)

    def _present_token(self, raw_token: int) -> int:
        if self.config.observation.token_scrambling == "none":
            return int(raw_token)
        return int(self._presentation_rng.integers(self.model.n_tokens))

    def _reset_token_delivery(self) -> None:
        self._raw_token_history.clear()
        self._raw_token_history.append(self._raw_token)
        if self.config.delay == 0:
            self._visible_source_token = self._raw_token
            self._visible_token = self._present_token(self._raw_token)
        else:
            self._visible_source_token = None
            self._visible_token = None

    def _advance_token_delivery(self, raw_token_after: int) -> None:
        self._raw_token_history.append(raw_token_after)
        if len(self._raw_token_history) < self.config.delay + 1:
            self._visible_source_token = None
            self._visible_token = None
            return
        self._visible_source_token = self._raw_token_history[0]
        self._visible_token = self._present_token(
            self._visible_source_token
        )

    def _reset_beliefs(self) -> None:
        if self._belief_tracker is not None:
            if self.config.delay == 0:
                assert self._visible_token is not None
                self._belief_tracker.reset(
                    self._visible_token,
                    likelihood=self._visible_likelihood,
                )
            else:
                self._belief_tracker.reset()
        if self._raw_belief_tracker is not None:
            if self.config.delay == 0:
                self._raw_belief_tracker.reset(
                    self._raw_token,
                    likelihood=self.model.emission_matrix,
                )
            else:
                self._raw_belief_tracker.reset()

    def _advance_beliefs(
        self,
        transition_matrix: np.ndarray,
        *,
        raw_token_before: int,
        raw_token_after: int,
    ) -> None:
        if self._belief_tracker is not None:
            if self._visible_token is None:
                raise RuntimeError("a post-step decision must have a visible token")
            self._belief_tracker.advance(
                self._visible_token,
                self._visible_likelihood,
                transition_matrix,
                delay=self.config.delay,
            )

        if self._raw_belief_tracker is not None:
            observation = (
                raw_token_before
                if self.config.delay == 1
                else raw_token_after
            )
            self._raw_belief_tracker.advance(
                observation,
                self.model.emission_matrix,
                transition_matrix,
                delay=self.config.delay,
            )

    def _validate_transition(self, transition_matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(transition_matrix, dtype=np.float64)
        expected = (self.model.n_states, self.model.n_states)
        if matrix.shape != expected:
            raise ValueError(
                f"task transition_matrix must have shape {expected}"
            )
        if (
            not np.isfinite(matrix).all()
            or (matrix < 0.0).any()
            or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12)
        ):
            raise ValueError("task transition_matrix must be row-stochastic")
        return matrix

    def _encode_action(self, action: Any) -> np.ndarray:
        assert self._action_feature_space is not None
        encoded = np.asarray(
            self.task.encode_action(action),
            dtype=np.float32,
        )
        if encoded.shape != self._action_feature_space.shape:
            raise ValueError(
                "task.encode_action returned a shape inconsistent with "
                "action_observation_space"
            )
        return encoded

    def _uniform_action(self) -> Any:
        space = self.action_space
        if isinstance(space, gym.spaces.Discrete):
            return int(
                self._presentation_rng.integers(
                    space.start,
                    space.start + space.n,
                )
            )
        if isinstance(space, gym.spaces.Box):
            if not np.isfinite(space.low).all() or not np.isfinite(space.high).all():
                raise ValueError(
                    "uniform action scrambling requires finite Box bounds"
                )
            return self._presentation_rng.uniform(
                space.low,
                space.high,
            ).astype(space.dtype)
        raise TypeError(
            "uniform action scrambling supports Discrete and Box action spaces"
        )

    def _present_action(self, executed_action: Any) -> np.ndarray:
        if self.config.observation.action_scrambling == "uniform":
            return self._encode_action(self._uniform_action())
        return self._encode_action(executed_action)

    def _history_record(self, offset: int) -> _DecisionRecord | None:
        if offset >= len(self._history):
            return None
        return self._history[-1 - offset]

    def _build_observation(self) -> np.ndarray:
        features: list[np.ndarray] = []
        observation = self.config.observation
        if observation.token is not None:
            for index in range(observation.token.depth):
                record = self._history_record(
                    observation.token.offset + index
                )
                token = None if record is None else record.visible_token
                features.append(_one_hot(token, self.model.n_tokens))
        if observation.action is not None:
            assert self._action_feature_space is not None
            for index in range(observation.action.depth):
                record = self._history_record(
                    observation.action.offset + index
                )
                if record is None or record.action_features is None:
                    features.append(
                        np.zeros(
                            self._action_feature_space.shape,
                            dtype=np.float32,
                        )
                    )
                else:
                    features.append(record.action_features)
        if observation.belief:
            if self._belief_tracker is None:
                raise RuntimeError("belief observation requires belief tracking")
            features.append(
                np.asarray(self._belief_tracker.belief, dtype=np.float32)
            )
        if observation.hidden_state:
            features.append(_one_hot(self._state, self.model.n_states))
        return np.concatenate(features).astype(np.float32, copy=False)

    def _current_info(self) -> dict[str, Any]:
        diagnostics = self.config.diagnostics
        info: dict[str, Any] = {"decision_step": self._step}
        if diagnostics.state:
            info["state_current"] = self._state
        if diagnostics.belief:
            if self._belief_tracker is None:
                raise RuntimeError("belief diagnostics require belief tracking")
            info["belief_current"] = self._belief_tracker.belief.copy()
        if diagnostics.raw_belief:
            if self._raw_belief_tracker is None:
                raise RuntimeError(
                    "raw-belief diagnostics require raw-belief tracking"
                )
            info["raw_belief_current"] = (
                self._raw_belief_tracker.belief.copy()
            )
        if diagnostics.tokens:
            info["raw_token_current"] = self._raw_token
            info["visible_token_current"] = self._visible_token
            info["visible_source_token"] = self._visible_source_token
        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            self._seed(seed)

        self._reset_episode_length()
        self.task.reset()
        self._step = 0
        self._state = self._sample(
            self._state_rng,
            self.model.initial_distribution,
        )
        self._raw_token = self._sample(
            self._emission_rng,
            self.model.emission_matrix[self._state],
        )
        self._reset_token_delivery()
        self._reset_beliefs()
        self._history.clear()
        self._history.append(
            _DecisionRecord(
                visible_token=self._visible_token,
                action_features=None,
                reward=None,
            )
        )
        self._initialized = True
        self._needs_reset = False
        return self._build_observation(), self._current_info()

    def step(
        self,
        action: Any,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._initialized:
            raise RuntimeError("reset must be called before step")
        if self._needs_reset:
            raise RuntimeError("reset must be called after episode truncation")

        state_before = self._state
        raw_token_before = self._raw_token
        belief_before = (
            None
            if not self._task_requires_belief
            else self._belief_tracker.belief.copy()
        )
        decision = self.task.resolve_action(
            action,
            state_before,
            self.model,
        )
        if not isinstance(decision, ActionDecision):
            raise TypeError("task.resolve_action must return ActionDecision")
        transition_matrix = self._validate_transition(
            decision.transition_matrix
        )

        self._state = self._sample(
            self._state_rng,
            transition_matrix[state_before],
        )
        self._raw_token = self._sample(
            self._emission_rng,
            self.model.emission_matrix[self._state],
        )
        self._advance_token_delivery(self._raw_token)
        self._advance_beliefs(
            transition_matrix,
            raw_token_before=raw_token_before,
            raw_token_after=self._raw_token,
        )
        belief_after = (
            None
            if not self._task_requires_belief
            else self._belief_tracker.belief.copy()
        )
        event = TransitionEvent(
            step=self._step,
            state_before=state_before,
            state_after=self._state,
            raw_token_before=raw_token_before,
            raw_token_after=self._raw_token,
            belief_before=belief_before,
            belief_after=belief_after,
        )
        reward, reward_components = self.task.reward(event, decision)
        reward = float(reward)
        reward_components = {
            str(name): float(value)
            for name, value in reward_components.items()
        }

        action_features = (
            None
            if self.config.observation.action is None
            else self._present_action(decision.executed_action)
        )
        self._history.append(
            _DecisionRecord(
                visible_token=self._visible_token,
                action_features=action_features,
                reward=reward,
            )
        )
        self._step += 1
        truncated = self._step >= self._current_episode_length
        if truncated:
            self._first_episode_pending = False
            self._needs_reset = True
            on_truncation = getattr(self.task, "on_truncation", None)
            if callable(on_truncation):
                on_truncation()

        observation = self._build_observation()
        info = self._current_info()
        diagnostics = self.config.diagnostics
        if diagnostics.rewards:
            info["reward_components"] = reward_components
        if diagnostics.transitions:
            info.update(
                {
                    "transition_step": event.step,
                    "state_before": event.state_before,
                    "state_after": event.state_after,
                    "raw_token_before": event.raw_token_before,
                    "raw_token_after": event.raw_token_after,
                    "original_transition_matrix": (
                        self.model.transition_matrix.copy()
                    ),
                    "executed_transition_matrix": transition_matrix.copy(),
                    "requested_action": _diagnostic_action(
                        decision.requested_action
                    ),
                    "executed_action": _diagnostic_action(
                        decision.executed_action
                    ),
                }
            )
            reference_matrix = decision.metadata.get(
                "reference_transition_matrix"
            )
            if reference_matrix is not None:
                info["reference_transition_matrix"] = np.asarray(
                    reference_matrix,
                    dtype=np.float64,
                ).copy()
        return observation, reward, False, truncated, info
