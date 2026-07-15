"""Composable Learner-side loss mixins."""

from losses.next_token import (
    NextTokenAuxLossMixin,
    masked_classification_metrics,
)

__all__ = ["NextTokenAuxLossMixin", "masked_classification_metrics"]
