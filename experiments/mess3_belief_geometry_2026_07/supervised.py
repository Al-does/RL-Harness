"""Study-local supervised workflow for the two non-RL MESS3 controls."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES


TargetName = Literal["state", "next_token"]


def _training_device(context: RunContext) -> torch.device:
    profile = context.hardware or PROFILES["cpu"]
    if profile.learner_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profile selected but CUDA is unavailable")
        return torch.device("cuda")
    if (
        profile.learner_device == "mps"
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def rollout_episodes(
    env_factory: Callable[[], gym.Env],
    *,
    n_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect random-action MESS3 episodes and decision-time states."""
    rng = np.random.default_rng(seed)
    observations, states = [], []
    env = env_factory()
    try:
        for _ in range(n_episodes):
            episode_seed = int(rng.integers(2**31 - 1))
            env.action_space.seed(episode_seed)
            observation, info = env.reset(seed=episode_seed)
            episode_observations = [observation]
            episode_states = [info["state"]]
            done = False
            while not done:
                observation, _, terminated, truncated, info = env.step(
                    env.action_space.sample()
                )
                done = terminated or truncated
                if not done:
                    episode_observations.append(observation)
                    episode_states.append(info["state"])
            observations.append(np.stack(episode_observations))
            states.append(np.asarray(episode_states, dtype=np.int64))
    finally:
        env.close()
    return np.stack(observations), np.stack(states)


def _make_targets(
    observations: torch.Tensor,
    states: torch.Tensor,
    target: TargetName,
    *,
    num_token_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, slice]:
    if target == "state":
        return (
            states,
            torch.ones_like(states, dtype=torch.bool),
            slice(None),
        )
    next_tokens = observations[:, 1:, :num_token_classes]
    targets = next_tokens.argmax(dim=-1)
    valid = next_tokens.sum(dim=-1) > 0.5
    return targets, valid, slice(None, -1)


def train_supervised(
    context: RunContext,
    *,
    env_factory: Callable[[], gym.Env],
    module_class: type,
    model_config: dict[str, Any],
    logits_from_embeddings: Callable[[Any, torch.Tensor], torch.Tensor],
    target: TargetName,
    total_steps: int,
    num_classes: int,
    batch_episodes: int = 8,
    learning_rate: float = 3e-4,
    fresh_data_episodes: int = 512,
    log_every: int = 25,
):
    """Train one study-local classification control and return its module."""
    if context.seed is None:
        raise ValueError("supervised runs require a resolved integer seed")
    torch.manual_seed(context.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(context.seed)
    device = _training_device(context)

    probe_env = env_factory()
    try:
        observation_space = probe_env.observation_space
        action_space = probe_env.action_space
    finally:
        probe_env.close()
    module = module_class(
        observation_space=observation_space,
        action_space=action_space,
        model_config=model_config,
    ).to(device)
    if not hasattr(module, "encode_chunks"):
        raise TypeError(
            f"{module_class.__name__} lacks supervised sequence encoding"
        )
    optimizer = torch.optim.Adam(
        module.parameters(),
        lr=learning_rate,
    )

    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    checkpoints = outputs.checkpoints_dir
    checkpoints.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(tag: str, env_steps: int) -> Path:
        path = checkpoints / f"module_state_{tag}.pt"
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in module.state_dict().items()
                },
                "env_steps": env_steps,
            },
            path,
        )
        return path

    save_checkpoint("00000000", 0)
    env_steps = 0
    optimizer_step = 0
    next_checkpoint = 1
    data_seed = context.seed * 1000
    started_at = time.monotonic()

    with outputs.progress_path.open("a") as progress:
        while env_steps < total_steps:
            data_seed += 1
            observation_array, state_array = rollout_episodes(
                env_factory,
                n_episodes=fresh_data_episodes,
                seed=data_seed,
            )
            all_observations = torch.from_numpy(observation_array).float()
            all_states = torch.from_numpy(state_array)
            episode_count = all_observations.shape[0]
            permutation = torch.randperm(episode_count)
            for start in range(0, episode_count, batch_episodes):
                indices = permutation[start : start + batch_episodes]
                observations = all_observations[indices].to(device)
                states = all_states[indices].to(device)
                batch_size, sequence_length, _ = observations.shape
                context_window = torch.zeros(
                    batch_size,
                    module.sequence_lookback,
                    observations.shape[-1],
                    device=device,
                )
                context_lengths = torch.zeros(
                    batch_size,
                    device=device,
                )
                embeddings = module.encode_chunks(
                    context_window,
                    context_lengths,
                    observations,
                )
                logits = logits_from_embeddings(module, embeddings)
                targets, valid, logits_slice = _make_targets(
                    observations,
                    states,
                    target,
                    num_token_classes=num_classes,
                )
                logits = logits[:, logits_slice, :]
                loss = F.cross_entropy(logits[valid], targets[valid])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                optimizer_step += 1
                env_steps += batch_size * sequence_length
                if optimizer_step % log_every == 0:
                    with torch.no_grad():
                        accuracy = (
                            logits[valid].argmax(dim=-1) == targets[valid]
                        ).float().mean()
                    record = {
                        "optimizer_step": optimizer_step,
                        "env_steps": env_steps,
                        "cross_entropy": float(loss.detach()),
                        "accuracy": float(accuracy),
                        "wall_seconds": round(
                            time.monotonic() - started_at,
                            1,
                        ),
                    }
                    progress.write(json.dumps(record) + "\n")
                    progress.flush()
                if optimizer_step >= next_checkpoint:
                    save_checkpoint(f"{env_steps:08d}", env_steps)
                    next_checkpoint *= 2
                if env_steps >= total_steps:
                    break

    final_checkpoint = save_checkpoint("final", env_steps)
    (context.results_dir / "summary.json").write_text(
        json.dumps(
            {
                "env_steps": env_steps,
                "optimizer_steps": optimizer_step,
                "final_checkpoint": str(final_checkpoint),
                "target": target,
            },
            indent=2,
        )
        + "\n"
    )
    return module.cpu()
