"""Composable Learner-side loss mixins."""

from losses.next_token import (
    NextTokenAuxLossMixin,
    masked_classification_metrics,
)
from losses.quantile import quantile_huber_loss

__all__ = [
    "NextTokenAuxLossMixin",
    "masked_classification_metrics",
    "quantile_huber_loss",
]
