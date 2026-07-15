"""Device-native target extraction for MESS3 observation layouts."""

from __future__ import annotations

import torch


def next_token_targets(
    observations: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_token_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return next-visible-token labels and valid transition positions."""
    if observations.ndim != 3:
        raise ValueError(
            "MESS3 token targets require observations shaped (B, T, D)"
        )
    if tuple(mask.shape) != tuple(observations.shape[:2]):
        raise ValueError("mask must match the observation batch and time axes")
    if (
        num_token_classes <= 0
        or num_token_classes > observations.shape[-1]
    ):
        raise ValueError(
            "num_token_classes must fit in the observation feature axis"
        )

    next_tokens = observations[:, 1:, :num_token_classes]
    targets = next_tokens.argmax(dim=-1)
    populated = next_tokens.sum(dim=-1) > 0.5
    valid_steps = mask.to(dtype=torch.bool)
    valid = valid_steps[:, :-1] & valid_steps[:, 1:] & populated
    return targets, valid
