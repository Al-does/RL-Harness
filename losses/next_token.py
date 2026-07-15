"""Algorithm-agnostic next-token auxiliary loss for RLlib Learners."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F

NAMESPACE = "next_token_aux"
LAMBDA_KEY = f"{NAMESPACE}/lambda"
FWD_KEY = f"{NAMESPACE}/logits"
TARGET_EXTRACTOR_KEY = f"{NAMESPACE}/target_extractor"

TargetExtractor = Callable[
    [Mapping[str, Any], torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


def masked_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return device-native cross-entropy and accuracy over valid positions."""
    if logits.ndim < 2:
        raise ValueError("classification logits must include a class axis")
    if tuple(targets.shape) != tuple(logits.shape[:-1]):
        raise ValueError("target shape must match the non-class logits axes")
    if tuple(valid.shape) != tuple(targets.shape):
        raise ValueError("valid mask shape must match targets")

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1).to(dtype=torch.long)
    flat_valid = valid.reshape(-1).to(
        device=flat_logits.device, dtype=flat_logits.dtype
    )
    valid_count = flat_valid.sum().clamp_min(1.0)
    per_item_ce = F.cross_entropy(
        flat_logits, flat_targets, reduction="none"
    )
    cross_entropy = (per_item_ce * flat_valid).sum() / valid_count
    accuracy = (
        (flat_logits.argmax(dim=-1) == flat_targets).to(flat_logits.dtype)
        * flat_valid
    ).sum() / valid_count
    return cross_entropy, accuracy


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

        logits = fwd_out[FWD_KEY]
        extractor: TargetExtractor | None = config.learner_config_dict.get(
            TARGET_EXTRACTOR_KEY
        )
        if not callable(extractor):
            raise ValueError(
                f"active {NAMESPACE} loss requires callable "
                f"{TARGET_EXTRACTOR_KEY!r}"
            )
        aligned_logits, targets, valid = extractor(batch, logits)
        cross_entropy, accuracy = masked_classification_metrics(
            aligned_logits,
            targets,
            valid,
        )

        self.metrics.log_dict(
            {
                f"{NAMESPACE}/ce": cross_entropy,
                f"{NAMESPACE}/accuracy": accuracy,
            },
            key=module_id,
            window=1,
        )
        return total + weight * cross_entropy
