"""Complete memoryless MLP model for RLlib PPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

import torch

from ray.rllib.core.columns import Columns

from learners.components.mlp import MLPEncoder
from learners.models.base import BaseActorCriticModel


@dataclass(frozen=True)
class MLPModelConfig:
    hidden_dims: tuple[int, ...] = (128, 128)

    def __post_init__(self):
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")

    @classmethod
    def from_dict(cls, values: dict) -> "MLPModelConfig":
        own_fields = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in own_fields})

    def to_dict(self) -> dict:
        return asdict(self)

    def override(self, **values) -> "MLPModelConfig":
        return replace(self, **values)


class MLPModel(BaseActorCriticModel):
    def _build_encoder(self) -> int:
        self.config = MLPModelConfig.from_dict(dict(self.model_config))
        self.encoder = MLPEncoder(
            int(self.observation_space.shape[0]), self.config.hidden_dims
        )
        return self.encoder.output_dim

    def _encode_train(self, batch):
        return self.encoder(batch[Columns.OBS]), None

    def _encode_rollout(self, batch):
        return self.encoder(batch[Columns.OBS]), None

    @torch.no_grad()
    def encode_step(self, obs: torch.Tensor, state: dict | None = None):
        return self.encoder(obs), state
