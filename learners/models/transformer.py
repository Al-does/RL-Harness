"""Complete stateful transformer model for RLlib PPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

import numpy as np
import torch

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override

from learners.components.transformer import CausalTransformerEncoder
from learners.models.base import BaseActorCriticModel


@dataclass(frozen=True)
class TransformerModelConfig:
    d_model: int = 96
    n_layers: int = 3
    n_heads: int = 4
    context_len: int = 64
    max_seq_len: int = 32

    def __post_init__(self):
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("transformer dimensions and layer counts must be positive")
        if self.context_len <= 0 or self.max_seq_len <= 0:
            raise ValueError("context_len and max_seq_len must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head dimension must be even for RoPE")

    @classmethod
    def from_dict(cls, values: dict) -> "TransformerModelConfig":
        own_fields = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in own_fields})

    def to_dict(self) -> dict:
        return asdict(self)

    def override(self, **values) -> "TransformerModelConfig":
        return replace(self, **values)


class TransformerModel(BaseActorCriticModel):
    """Banded causal transformer with windowed train and cached rollout paths."""

    def _build_encoder(self) -> int:
        self.config = TransformerModelConfig.from_dict(dict(self.model_config))
        self._obs_dim = int(self.observation_space.shape[0])
        self.encoder = CausalTransformerEncoder(
            obs_dim=self._obs_dim,
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            context_len=self.config.context_len,
        )
        return self.config.d_model

    @property
    def sequence_lookback(self) -> int:
        return self.encoder.lookback

    @override(TorchRLModule)
    def get_initial_state(self) -> Any:
        cache_shape = (
            self.encoder.n_layers,
            self.encoder.n_heads,
            self.encoder.cache_len,
            self.encoder.head_dim,
        )
        return {
            "ctx": np.zeros(
                (self.encoder.lookback, self._obs_dim), dtype=np.float32
            ),
            "len": np.zeros((1,), dtype=np.float32),
            "kv_k": np.zeros(cache_shape, dtype=np.float32),
            "kv_v": np.zeros(cache_shape, dtype=np.float32),
            "kv_len": np.zeros((1,), dtype=np.float32),
        }

    def _advance_context(self, obs, state):
        context, lens = state["ctx"], state["len"].reshape(-1)
        sequence = torch.cat([context, obs], dim=1)
        return (
            sequence[:, -self.encoder.lookback :, :],
            torch.clamp(
                lens + obs.shape[1], max=float(self.encoder.lookback)
            ).reshape(-1, 1),
        )

    def _encode_train(self, batch):
        obs = batch[Columns.OBS]
        state = batch[Columns.STATE_IN]
        embeddings = self.encoder(
            state["ctx"], state["len"].reshape(-1), obs
        )
        context_out, len_out = self._advance_context(obs, state)
        state_out = {"ctx": context_out, "len": len_out}
        for key in ("kv_k", "kv_v", "kv_len"):
            if key in state:
                state_out[key] = state[key]
        return embeddings, state_out

    def _encode_rollout(self, batch):
        obs = batch[Columns.OBS]
        state = batch[Columns.STATE_IN]
        embeddings, kv_k, kv_v, kv_len = self.encoder.forward_cached(
            state["kv_k"],
            state["kv_v"],
            state["kv_len"].reshape(-1),
            obs,
        )
        context_out, len_out = self._advance_context(obs, state)
        return embeddings, {
            "ctx": context_out,
            "len": len_out,
            "kv_k": kv_k,
            "kv_v": kv_v,
            "kv_len": kv_len.reshape(-1, 1),
        }

    @torch.no_grad()
    def encode_step(
        self, obs: torch.Tensor, state: dict
    ) -> tuple[torch.Tensor, dict]:
        embeddings, state_out = self._encode_rollout(
            {Columns.OBS: obs.unsqueeze(1), Columns.STATE_IN: state}
        )
        return embeddings[:, 0, :], state_out

    def encode_chunks(
        self, context: torch.Tensor, lens: torch.Tensor, obs: torch.Tensor
    ):
        return self.encoder(context, lens, obs)
