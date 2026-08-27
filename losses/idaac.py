"""Device-native objective primitives for DAAC and IDAAC."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average ``values`` over valid elements without leaving their device."""

    if mask is None:
        return values.mean()
    weights = mask.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def ppo_surrogate(
    logp_ratio: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_param: float,
) -> torch.Tensor:
    """Return PPO's pointwise clipped policy-gain surrogate."""

    unclipped = logp_ratio * advantages
    clipped = (
        torch.clamp(logp_ratio, 1.0 - clip_param, 1.0 + clip_param)
        * advantages
    )
    return torch.minimum(unclipped, clipped)


def advantage_prediction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error for the policy network's action-conditioned GAE head."""

    return masked_mean((predictions - targets).square(), mask)


def invariance_losses(
    discriminator_logits: torch.Tensor,
    encoder_logits: torch.Tensor,
    order_targets: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return discriminator classification and encoder-confusion losses.

    Gradient isolation is the caller's responsibility: discriminator logits
    should use detached policy features, while encoder logits should use
    detached discriminator parameters.
    """

    return (
        discriminator_order_loss(
            discriminator_logits,
            order_targets,
            mask=mask,
        ),
        encoder_confusion_loss(encoder_logits, mask=mask),
    )


def discriminator_order_loss(
    logits: torch.Tensor,
    order_targets: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary temporal-order classification loss."""

    targets = order_targets.to(device=logits.device, dtype=logits.dtype)
    pointwise = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    return masked_mean(pointwise, mask)


def encoder_confusion_loss(
    logits: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Push discriminator predictions toward maximum binary uncertainty."""

    confusion_targets = torch.full_like(logits, 0.5)
    pointwise = F.binary_cross_entropy_with_logits(
        logits,
        confusion_targets,
        reduction="none",
    )
    return masked_mean(pointwise, mask)


def clipped_value_loss(
    predictions: torch.Tensor,
    old_predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    clip_param: float,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return PPO-style clipped half-MSE and its unclipped diagnostic."""

    error_squared = (predictions - targets).square()
    clipped_predictions = old_predictions + torch.clamp(
        predictions - old_predictions,
        -clip_param,
        clip_param,
    )
    clipped_error_squared = (clipped_predictions - targets).square()
    loss = 0.5 * masked_mean(
        torch.maximum(error_squared, clipped_error_squared),
        mask,
    )
    return loss, 0.5 * masked_mean(error_squared, mask)


__all__ = [
    "advantage_prediction_loss",
    "clipped_value_loss",
    "discriminator_order_loss",
    "encoder_confusion_loss",
    "invariance_losses",
    "masked_mean",
    "ppo_surrogate",
]
