"""PPO next-token prediction on five independent MESS3 HMMs."""

from __future__ import annotations

from experiments.factored_mess3_beliefs_2026_08.shared import (
    build_config as _build_config,
    run_independent,
)
from harness.context import RunContext


def build_config(context: RunContext):
    return _build_config(context, n_factors=5)


def run(context: RunContext):
    return run_independent(context, n_factors=5)


__all__ = ["build_config", "run"]
