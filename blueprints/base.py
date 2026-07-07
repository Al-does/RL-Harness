"""Blueprint pattern for the MESS3-Control program.

A Blueprint is an inert, typed description of ONE experiment arm: environment
entry point + kwargs, model spec, PPO hyperparameters, budget, and gate
requirements.  Any result is reproducible from (blueprint name, seed):

    uv run python scripts/train.py --blueprint a_main --seed 0

Blueprints share config through composition (factory functions building on
common bases), never through inheritance of mutable state.  Training may not
launch unless the blueprint's ``gate`` artifact exists and passes — enforced
by scripts/train.py, per the program rule that no training run launches
before its phase gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Policy core. kind: 'transformer' | 'mlp'."""
    kind: str = "transformer"
    d_model: int = 96
    n_layers: int = 3
    n_heads: int = 4
    context_len: int = 64          # >= 64 per program spec (transformer only)
    mlp_hidden: tuple = (128, 128)  # mlp only


@dataclass(frozen=True)
class PPOSpec:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.003
    train_batch: int = 32_768
    minibatch: int = 4_096
    epochs: int = 6
    num_env_runners: int = 4


@dataclass(frozen=True)
class Blueprint:
    name: str
    phase: int
    env_entry: str                     # "module.path:ClassName"
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    model: ModelSpec = field(default_factory=ModelSpec)
    ppo: PPOSpec = field(default_factory=PPOSpec)
    total_steps: int = 10_000_000
    n_seeds: int = 3
    gate: str = "phase1"               # results/<gate>/GATE_PASSED must exist
    aux_next_token_lambda: float = 0.0  # auxiliary prediction head weight
    rl_loss_enabled: bool = True        # False => prediction-only (A-pred)
    scramble_tokens: bool = False       # N-scramble: i.i.d. uniform token obs
    notes: str = ""

    def with_(self, **kw) -> "Blueprint":
        return replace(self, **kw)


REGISTRY: dict[str, "Blueprint"] = {}


def register(bp: Blueprint) -> Blueprint:
    if bp.name in REGISTRY:
        raise ValueError(f"duplicate blueprint name: {bp.name}")
    REGISTRY[bp.name] = bp
    return bp


def get(name: str) -> Blueprint:
    # Import for side effects so all arm modules register themselves.
    import blueprints.mess3_arms  # noqa: F401

    if name not in REGISTRY:
        raise KeyError(f"unknown blueprint '{name}'; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def all_blueprints() -> dict[str, Blueprint]:
    import blueprints.mess3_arms  # noqa: F401

    return dict(REGISTRY)
