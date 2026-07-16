"""Small domain adapters and operational helpers shared by this study."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from ray.rllib.core.columns import Columns

from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners


OPERATING_POINT = {
    "transition_kl_beta": 4.0,
    "action_limit": 5.0,
    "delay": 1,
}
CONTINUOUS_ENV_BASE = {
    "model": {
        "factory": "envs.mess3.model:control_model",
        "kwargs": {"alpha": 0.85},
    },
    "task": {
        "class": (
            "envs.mess3.tasks.occupancy_control:"
            "OccupancyControlTask"
        ),
        "kwargs": {
            "transition_kl_beta": OPERATING_POINT["transition_kl_beta"],
            "action_limit": OPERATING_POINT["action_limit"],
        },
    },
    "delay": OPERATING_POINT["delay"],
    "episode_length": 1024,
}
STATE_GUESS_ENV_BASE = {
    "model": {
        "factory": "envs.mess3.model:state_guess_model",
        "kwargs": {"alpha": 0.85},
    },
    "task": {
        "class": "envs.mess3.tasks.state_guess:StateGuessTask",
    },
    "observation": {"action": None},
    "delay": 1,
    "episode_length": 1024,
}
SMOKE_ENV_STEPS = 4096


def apply_runtime_resources(
    config: Any,
    context: RunContext,
    *,
    default_env_runners: int,
) -> Any:
    """Apply only operational runner/Learner resources to an RLlib config."""
    profile = context.hardware or PROFILES["cpu"]
    return config.env_runners(
        num_env_runners=(
            0
            if context.smoke
            else resolve_env_runners(profile, default_env_runners)
        ),
        num_envs_per_env_runner=(
            1 if context.smoke else profile.num_envs_per_env_runner
        ),
        num_gpus_per_env_runner=(
            0 if context.smoke else profile.num_gpus_per_env_runner
        ),
        sample_timeout_s=600.0,
    ).learners(
        num_gpus_per_learner=(
            1 if profile.learner_device == "cuda" else 0
        )
    )


def next_visible_token_targets(
    batch: Mapping[str, Any],
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adapt MESS3's observation layout to next-token classification."""
    observations = batch[Columns.OBS]
    if observations.ndim != 3 or logits.ndim != 3:
        raise ValueError("MESS3 next-token training expects (B, T, D) tensors")
    num_classes = logits.shape[-1]
    if num_classes > observations.shape[-1]:
        raise ValueError("token classes must fit in the observation feature axis")

    mask = batch.get(Columns.LOSS_MASK)
    if mask is None:
        mask = torch.ones(
            observations.shape[:2],
            dtype=torch.bool,
            device=observations.device,
        )
    else:
        mask = mask.to(dtype=torch.bool)

    next_tokens = observations[:, 1:, :num_classes]
    targets = next_tokens.argmax(dim=-1)
    populated = next_tokens.sum(dim=-1) > 0.5
    valid = mask[:, :-1] & mask[:, 1:] & populated
    return logits[:, :-1, :], targets, valid
