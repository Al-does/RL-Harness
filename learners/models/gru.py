"""Stateful GRU actor-critic with truncated-BPTT RLlib state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

import numpy as np
import torch
from torch import nn
from ray.rllib.core.columns import Columns

from learners.models.base import BaseActorCriticModel


@dataclass(frozen=True)
class GRUModelConfig:
    hidden_dim: int = 128
    n_layers: int = 2
    max_seq_len: int = 32
    obs_feature_indices: tuple[int, ...] | None = None

    def __post_init__(self):
        if min(self.hidden_dim, self.n_layers, self.max_seq_len) <= 0:
            raise ValueError("GRU dimensions and sequence length must be positive")
        if self.obs_feature_indices is not None:
            indices = tuple(self.obs_feature_indices)
            if not indices or any(not isinstance(i, int) or i < 0 for i in indices):
                raise ValueError("obs_feature_indices must contain nonnegative integers")
            object.__setattr__(self, "obs_feature_indices", indices)

    @classmethod
    def from_dict(cls, values: dict) -> "GRUModelConfig":
        own_fields = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in own_fields})

    def to_dict(self) -> dict:
        return asdict(self)

    def override(self, **values) -> "GRUModelConfig":
        return replace(self, **values)


class GRUEncoder(nn.Module):
    def __init__(self, obs_dim: int, config: GRUModelConfig):
        super().__init__()
        self.input_projection = nn.Linear(obs_dim, config.hidden_dim)
        self.gru = nn.GRU(config.hidden_dim, config.hidden_dim, config.n_layers,
                          batch_first=True)
        self.final_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, obs, h, *, apply_final_norm=True):
        residual, h = self.gru(self.input_projection(obs), h.transpose(0, 1).contiguous())
        return (self.final_norm(residual) if apply_final_norm else residual), h.transpose(0, 1)


class GRUModel(BaseActorCriticModel):
    def _build_encoder(self) -> int:
        self.config = GRUModelConfig.from_dict(dict(self.model_config))
        obs_dim = int(self.observation_space.shape[0])
        indices = self.config.obs_feature_indices
        if indices is not None:
            if max(indices) >= obs_dim:
                raise ValueError("obs_feature_indices exceeds observation dimension")
            obs_dim = len(indices)
        self.encoder = GRUEncoder(obs_dim, self.config)
        return self.config.hidden_dim

    def get_initial_state(self):
        return {"h": np.zeros((self.config.n_layers, self.config.hidden_dim), dtype=np.float32)}

    def _encode_rollout(self, batch, *, apply_final_norm=True):
        obs = batch[Columns.OBS]
        if self.config.obs_feature_indices is not None:
            obs = obs[..., list(self.config.obs_feature_indices)]
        embeddings, h = self.encoder(obs, batch[Columns.STATE_IN]["h"],
                                     apply_final_norm=apply_final_norm)
        return embeddings, {"h": h}

    def _encode_train(self, batch):
        return self._encode_rollout(batch)

    @torch.no_grad()
    def encode_step(self, obs: torch.Tensor, state: dict):
        embeddings, state = self._encode_rollout(
            {Columns.OBS: obs.unsqueeze(1), Columns.STATE_IN: state})
        return embeddings[:, 0], state

    @torch.no_grad()
    def encode_step_pre_final_norm(self, obs: torch.Tensor, state: dict):
        embeddings, state = self._encode_rollout(
            {Columns.OBS: obs.unsqueeze(1), Columns.STATE_IN: state}, apply_final_norm=False)
        return embeddings[:, 0], state
