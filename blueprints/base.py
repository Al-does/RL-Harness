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

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ray.rllib.core.learner.learner import Learner
    from ray.rllib.core.rl_module.torch import TorchRLModule


class ModelConfig(Protocol):
    def to_dict(self) -> dict: ...

    def override(self, **values) -> "ModelConfig": ...


@dataclass(frozen=True)
class ModelSpec:
    """A complete RLModule class and its immutable typed configuration."""

    model_class: type["TorchRLModule"]
    config: ModelConfig
    mixin_config: dict[str, Any] = field(default_factory=dict)

    def with_config(self, **values) -> "ModelSpec":
        return replace(self, config=self.config.override(**values))

    def with_mixin_config(
        self, namespace: str, **values: Any
    ) -> "ModelSpec":
        mixin_config = deepcopy(self.mixin_config)
        mixin_config[namespace] = {
            **mixin_config.get(namespace, {}),
            **values,
        }
        return replace(self, mixin_config=mixin_config)

    def to_model_config(self) -> dict:
        config = self.config.to_dict()
        collisions = config.keys() & self.mixin_config.keys()
        if collisions:
            raise ValueError(
                f"base and mixin model config collide: {sorted(collisions)}"
            )
        config.update(deepcopy(self.mixin_config))
        return config

    def to_dict(self) -> dict:
        cls = self.model_class
        return {
            "class": f"{cls.__module__}:{cls.__qualname__}",
            "config": self.to_model_config(),
        }


def default_model_spec() -> ModelSpec:
    from learners.models import TransformerModel, TransformerModelConfig

    return ModelSpec(TransformerModel, TransformerModelConfig())


def default_learner_class() -> type["Learner"]:
    from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import (
        PPOTorchLearner,
    )

    return PPOTorchLearner


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
    model: ModelSpec = field(default_factory=default_model_spec)
    learner_class: type["Learner"] = field(
        default_factory=default_learner_class
    )
    aux_config: dict[str, Any] = field(default_factory=dict)
    ppo: PPOSpec = field(default_factory=PPOSpec)
    total_steps: int = 10_000_000
    n_seeds: int = 3
    gate: str = "phase1"               # results/<gate>/GATE_PASSED must exist
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
