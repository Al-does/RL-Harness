"""Memoryless pre-normalized residual MLP actor-critic."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

import torch
from torch import nn
from ray.rllib.core.columns import Columns

from learners.models.base import BaseActorCriticModel


@dataclass(frozen=True)
class ResidualMLPModelConfig:
    d_model: int = 64
    n_layers: int = 6
    obs_feature_indices: tuple[int, ...] | None = None

    def __post_init__(self):
        if min(self.d_model, self.n_layers) <= 0:
            raise ValueError("MLP dimensions and layer count must be positive")
        if self.obs_feature_indices is not None:
            indices = tuple(self.obs_feature_indices)
            if not indices or any(not isinstance(i, int) or i < 0 for i in indices):
                raise ValueError("obs_feature_indices must contain nonnegative integers")
            object.__setattr__(self, "obs_feature_indices", indices)

    @classmethod
    def from_dict(cls, values: dict) -> "ResidualMLPModelConfig":
        own_fields = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in own_fields})

    def to_dict(self) -> dict:
        return asdict(self)

    def override(self, **values) -> "ResidualMLPModelConfig":
        return replace(self, **values)


class ResidualMLPEncoder(nn.Module):
    def __init__(self, obs_dim: int, config: ResidualMLPModelConfig):
        super().__init__()
        width = config.d_model
        self.input_projection = nn.Linear(obs_dim, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width),
                          nn.GELU(), nn.Linear(4 * width, width))
            for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(width)

    def forward(self, obs, *, apply_final_norm=True):
        residual = self.input_projection(obs)
        for block in self.blocks:
            residual = residual + block(residual)
        return self.final_norm(residual) if apply_final_norm else residual


class ResidualMLPModel(BaseActorCriticModel):
    def _build_encoder(self) -> int:
        self.config = ResidualMLPModelConfig.from_dict(dict(self.model_config))
        obs_dim = int(self.observation_space.shape[0])
        indices = self.config.obs_feature_indices
        if indices is not None:
            if max(indices) >= obs_dim:
                raise ValueError("obs_feature_indices exceeds observation dimension")
            obs_dim = len(indices)
        self.encoder = ResidualMLPEncoder(obs_dim, self.config)
        return self.config.d_model

    def _encode_rollout(self, batch, *, apply_final_norm=True):
        obs = batch[Columns.OBS]
        if self.config.obs_feature_indices is not None:
            obs = obs[..., list(self.config.obs_feature_indices)]
        return self.encoder(obs, apply_final_norm=apply_final_norm), None

    def _encode_train(self, batch):
        return self._encode_rollout(batch)

    @torch.no_grad()
    def encode_step(self, obs: torch.Tensor, state: dict | None = None):
        embedding, _ = self._encode_rollout({Columns.OBS: obs})
        return embedding, state

    @torch.no_grad()
    def encode_step_pre_final_norm(self, obs: torch.Tensor, state: dict | None = None):
        embedding, _ = self._encode_rollout({Columns.OBS: obs}, apply_final_norm=False)
        return embedding, state
