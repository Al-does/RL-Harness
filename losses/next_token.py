"""Algorithm-agnostic next-token auxiliary loss for RLlib Learners."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ray.rllib.core.columns import Columns

NAMESPACE = "next_token_aux"
LAMBDA_KEY = f"{NAMESPACE}/lambda"
FWD_KEY = f"{NAMESPACE}/logits"


def next_token_targets(
    obs: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_token_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return next-token labels and valid transition positions."""
    if obs.ndim != 3:
        raise ValueError("next-token targets require observations shaped (B, T, D)")
    if tuple(mask.shape) != tuple(obs.shape[:2]):
        raise ValueError("loss mask must match the observation batch and time axes")
    if num_token_classes <= 0 or num_token_classes > obs.shape[-1]:
        raise ValueError("num_token_classes must fit in the observation feature axis")

    next_tokens = obs[:, 1:, :num_token_classes]
    targets = next_tokens.argmax(dim=-1)
    populated = next_tokens.sum(dim=-1) > 0.5
    valid_steps = mask.to(dtype=torch.bool)
    valid = valid_steps[:, :-1] & valid_steps[:, 1:] & populated
    return targets, valid


class NextTokenAuxLossMixin:
    """Add namespaced next-token cross-entropy to any base Learner loss."""

    def compute_loss_for_module(
        self, *, module_id, config, batch, fwd_out
    ):
        total = super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )
        weight = float(config.learner_config_dict.get(LAMBDA_KEY, 0.0))
        if weight <= 0.0 or FWD_KEY not in fwd_out:
            return total

        obs = batch[Columns.OBS]
        logits = fwd_out[FWD_KEY]
        if obs.ndim != 3 or logits.ndim != 3 or logits.shape[1] < 2:
            return total

        mask = batch.get(Columns.LOSS_MASK)
        if mask is None:
            mask = torch.ones(
                obs.shape[:2], dtype=torch.bool, device=obs.device
            )

        token_logits = logits[:, :-1, :]
        targets, valid = next_token_targets(
            obs,
            mask,
            num_token_classes=token_logits.shape[-1],
        )
        flat_logits = token_logits.reshape(-1, token_logits.shape[-1])
        flat_targets = targets.reshape(-1)
        flat_valid = valid.reshape(-1).to(dtype=flat_logits.dtype)
        valid_count = flat_valid.sum().clamp_min(1.0)

        per_token_ce = F.cross_entropy(
            flat_logits, flat_targets, reduction="none"
        )
        cross_entropy = (per_token_ce * flat_valid).sum() / valid_count
        accuracy = (
            (flat_logits.argmax(dim=-1) == flat_targets).to(flat_logits.dtype)
            * flat_valid
        ).sum() / valid_count

        self.metrics.log_dict(
            {
                f"{NAMESPACE}/ce": cross_entropy,
                f"{NAMESPACE}/accuracy": accuracy,
            },
            key=module_id,
            window=1,
        )
        return total + weight * cross_entropy
