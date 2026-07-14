"""Composable Learner-side loss mixins. See ``losses/AGENTS.md`` for the contract."""

from losses.next_token import NextTokenAuxLossMixin, next_token_targets

__all__ = ["NextTokenAuxLossMixin", "next_token_targets"]
