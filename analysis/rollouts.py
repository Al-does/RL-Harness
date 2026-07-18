"""Generic rollout collection driven by experiment-supplied callables."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from harness.seeding import (
    SeedSource,
    child_seed_sequence,
    named_seed_sequences,
    seed_sequence_to_int,
)


_ROLLOUT_STREAM_KEYS = {
    "episode_seeds": (0,),
    "action_spaces": (1,),
    "policy_sampling": (2,),
}
# Explicit spawn keys are order-independent; never renumber or reuse a key.


@dataclass(frozen=True, slots=True)
class RolloutData:
    representations: np.ndarray
    targets: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray


@dataclass(frozen=True, slots=True)
class BatchedRolloutData:
    """Aligned data collected from device-batched environment interaction."""

    representations: np.ndarray
    targets: dict[str, np.ndarray]
    actions: np.ndarray
    rewards: np.ndarray


@dataclass(frozen=True, slots=True)
class PolicyRandomness:
    """Local policy RNG state, independent from environments and action spaces."""

    seed_sequence: np.random.SeedSequence
    numpy: np.random.Generator


def _policy_randomness(
    seed: np.random.SeedSequence,
) -> PolicyRandomness:
    return PolicyRandomness(
        seed_sequence=seed,
        numpy=np.random.default_rng(seed),
    )


def _episode_seed(
    stream: np.random.SeedSequence,
    environment_index: int,
    episode_index: int,
) -> int:
    return seed_sequence_to_int(
        child_seed_sequence(stream, (environment_index, episode_index))
    )


def collect_rollout_data(
    env_factory: Callable[[], Any],
    step_adapter: Callable[
        [np.ndarray, Any, PolicyRandomness],
        tuple[Any, Any, np.ndarray],
    ],
    target_adapter: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
    *,
    n_steps: int,
    seed: SeedSource,
    initial_state: Callable[[], Any] | None = None,
    warmup: int = 0,
) -> RolloutData:
    """Collect on-demand representations and public environment targets.

    ``step_adapter`` returns ``(action, next_model_state, representation)``.
    The experiment owns all model, distribution, and target semantics.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    streams = named_seed_sequences(seed, _ROLLOUT_STREAM_KEYS)
    randomness = _policy_randomness(streams["policy_sampling"])
    env = env_factory()
    representations, targets, actions, rewards = [], [], [], []
    episode_step = 0
    episode_index = 0
    model_state = initial_state() if initial_state is not None else None
    try:
        env.action_space.seed(
            _episode_seed(streams["action_spaces"], 0, episode_index)
        )
        observation, info = env.reset(
            seed=_episode_seed(streams["episode_seeds"], 0, episode_index)
        )
        while len(representations) < n_steps:
            action, model_state, representation = step_adapter(
                observation,
                model_state,
                randomness,
            )
            target = target_adapter(observation, info)
            next_observation, reward, terminated, truncated, next_info = (
                env.step(action)
            )
            if episode_step >= warmup:
                representations.append(np.asarray(representation))
                targets.append(np.asarray(target))
                actions.append(np.atleast_1d(np.asarray(action)))
                rewards.append(reward)

            episode_step += 1
            if terminated or truncated:
                episode_index += 1
                env.action_space.seed(
                    _episode_seed(
                        streams["action_spaces"],
                        0,
                        episode_index,
                    )
                )
                observation, info = env.reset(
                    seed=_episode_seed(
                        streams["episode_seeds"],
                        0,
                        episode_index,
                    )
                )
                episode_step = 0
                model_state = (
                    initial_state() if initial_state is not None else None
                )
            else:
                observation, info = next_observation, next_info
    finally:
        env.close()

    return RolloutData(
        representations=np.asarray(representations),
        targets=np.asarray(targets),
        actions=np.asarray(actions),
        rewards=np.asarray(rewards),
    )


def collect_batched_rollout_data(
    env_factory: Callable[[], Any],
    step_adapter: Callable[
        [
            np.ndarray,
            Any,
            PolicyRandomness,
            Sequence[Any],
        ],
        tuple[Sequence[Any], Any, np.ndarray],
    ],
    target_adapter: Callable[
        [
            np.ndarray,
            Sequence[Mapping[str, Any]],
            np.ndarray,
        ],
        Mapping[str, Any],
    ],
    *,
    n_steps: int,
    seed: SeedSource,
    n_envs: int,
    initial_state: Callable[[int], Any] | None = None,
    reset_state: Callable[[Any, np.ndarray], Any] | None = None,
    warmup: int = 0,
) -> BatchedRolloutData:
    """Collect aligned rollouts while batching model inference across envs.

    The generic collector owns environment lifecycle, seeding, warmup, and
    episode resets. ``step_adapter`` owns model inference and action semantics;
    it receives stacked observations, local policy randomness, and the seeded
    action spaces. The target adapter returns named arrays whose leading
    dimension is ``n_envs``.

    Stateful adapters must provide both ``initial_state`` and ``reset_state``.
    The latter resets only the environment indices whose episodes ended.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if (initial_state is None) != (reset_state is None):
        raise ValueError(
            "initial_state and reset_state must be provided together"
        )

    streams = named_seed_sequences(seed, _ROLLOUT_STREAM_KEYS)
    randomness = _policy_randomness(streams["policy_sampling"])
    envs = []
    action_spaces = []
    observations: list[np.ndarray] = []
    infos: list[Mapping[str, Any]] = []
    episode_steps = np.zeros(n_envs, dtype=np.int64)
    episode_indices = np.zeros(n_envs, dtype=np.int64)
    model_state = None
    representations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[Any] = []
    targets: dict[str, list[np.ndarray]] = {}
    target_keys: set[str] | None = None

    try:
        for _ in range(n_envs):
            envs.append(env_factory())
        action_spaces = [env.action_space for env in envs]
        model_state = (
            initial_state(n_envs) if initial_state is not None else None
        )
        for index, (env, action_space) in enumerate(
            zip(envs, action_spaces)
        ):
            action_space.seed(
                _episode_seed(streams["action_spaces"], index, 0)
            )
            observation, info = env.reset(
                seed=_episode_seed(streams["episode_seeds"], index, 0)
            )
            observations.append(np.asarray(observation))
            infos.append(info)

        while len(representations) < n_steps:
            observation_batch = np.stack(observations)
            env_actions, model_state, representation_batch = step_adapter(
                observation_batch,
                model_state,
                randomness,
                action_spaces,
            )
            if len(env_actions) != n_envs:
                raise ValueError(
                    "step_adapter must return one action per environment"
                )
            representation_batch = np.asarray(representation_batch)
            if (
                representation_batch.ndim == 0
                or representation_batch.shape[0] != n_envs
            ):
                raise ValueError(
                    "step_adapter representations must lead with n_envs"
                )

            target_batch = {
                str(key): np.asarray(value)
                for key, value in target_adapter(
                    observation_batch,
                    infos,
                    episode_steps.copy(),
                ).items()
            }
            current_keys = set(target_batch)
            if target_keys is None:
                target_keys = current_keys
                targets = {key: [] for key in target_batch}
            elif current_keys != target_keys:
                raise ValueError(
                    "target_adapter must return stable target keys"
                )
            for key, value in target_batch.items():
                if value.ndim == 0 or value.shape[0] != n_envs:
                    raise ValueError(
                        f"target {key!r} must lead with n_envs"
                    )

            reset_indices = []
            for index, env in enumerate(envs):
                next_observation, reward, terminated, truncated, next_info = (
                    env.step(env_actions[index])
                )
                if (
                    episode_steps[index] >= warmup
                    and len(representations) < n_steps
                ):
                    representations.append(
                        np.asarray(representation_batch[index])
                    )
                    actions.append(
                        np.atleast_1d(np.asarray(env_actions[index]))
                    )
                    rewards.append(reward)
                    for key, value in target_batch.items():
                        targets[key].append(np.asarray(value[index]))

                episode_steps[index] += 1
                if terminated or truncated:
                    episode_indices[index] += 1
                    action_spaces[index].seed(
                        _episode_seed(
                            streams["action_spaces"],
                            index,
                            int(episode_indices[index]),
                        )
                    )
                    next_observation, next_info = env.reset(
                        seed=_episode_seed(
                            streams["episode_seeds"],
                            index,
                            int(episode_indices[index]),
                        )
                    )
                    episode_steps[index] = 0
                    reset_indices.append(index)
                observations[index] = np.asarray(next_observation)
                infos[index] = next_info

            if reset_indices and reset_state is not None:
                model_state = reset_state(
                    model_state,
                    np.asarray(reset_indices, dtype=np.int64),
                )
    finally:
        for env in envs:
            env.close()

    return BatchedRolloutData(
        representations=np.asarray(representations),
        targets={
            key: np.asarray(values) for key, values in targets.items()
        },
        actions=np.asarray(actions),
        rewards=np.asarray(rewards),
    )
