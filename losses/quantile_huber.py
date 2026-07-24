"""Pure tensor loss for implicit quantile regression."""

from __future__ import annotations

import torch


def quantile_huber_loss(
    quantiles: torch.Tensor,
    taus: torch.Tensor,
    targets: torch.Tensor,
    *,
    kappa: float = 1.0,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Regress sampled quantiles against scalar return samples."""

    if kappa <= 0.0:
        raise ValueError("quantile Huber kappa must be positive")
    if quantiles.shape != taus.shape:
        raise ValueError("quantiles and taus must have matching shapes")
    if quantiles.shape[:-1] != targets.shape:
        raise ValueError("targets must match quantile leading dimensions")

    errors = targets.unsqueeze(-1) - quantiles
    absolute_errors = errors.abs()
    huber = torch.where(
        absolute_errors <= kappa,
        0.5 * errors.square(),
        kappa * (absolute_errors - 0.5 * kappa),
    )
    weights = (
        taus - (errors.detach() < 0.0).to(dtype=taus.dtype)
    ).abs()
    per_item = (weights * huber / kappa).mean(dim=-1)
    if valid is None:
        return per_item.mean()

    valid_float = valid.to(device=per_item.device, dtype=per_item.dtype)
    return (per_item * valid_float).sum() / valid_float.sum().clamp_min(1.0)
