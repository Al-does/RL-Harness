"""PPO validation of factored belief geometry on two independent MESS3 HMMs."""

from __future__ import annotations

from experiments.factored_mess3_beliefs_2026_08.shared import (
    build_config,
    run_independent,
)
from harness.context import RunContext


def run(context: RunContext):
    return run_independent(context)


__all__ = ["build_config", "run"]
