"""Generic rollout collection driven by experiment-supplied callables."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RolloutData:
    representations: np.ndarray
    targets: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray


def collect_rollout_data(
    env_factory: Callable[[], Any],
    step_adapter: Callable[
        [np.ndarray, Any, np.random.Generator],
        tuple[Any, Any, np.ndarray],
    ],
    target_adapter: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
    *,
    n_steps: int,
    seed: int,
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

    rng = np.random.default_rng(seed)
    env = env_factory()
    representations, targets, actions, rewards = [], [], [], []
    episode_step = 0
    model_state = initial_state() if initial_state is not None else None
    try:
        episode_seed = int(rng.integers(2**31 - 1))
        env.action_space.seed(episode_seed)
        observation, info = env.reset(seed=episode_seed)
        while len(representations) < n_steps:
            action, model_state, representation = step_adapter(
                observation,
                model_state,
                rng,
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
                episode_seed = int(rng.integers(2**31 - 1))
                env.action_space.seed(episode_seed)
                observation, info = env.reset(seed=episode_seed)
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
