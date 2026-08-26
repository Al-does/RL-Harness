"""Paper-matched residual-stream transformer adapted to RLlib."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from torch import nn

from learners.models.base import BaseActorCriticModel


@dataclass(frozen=True, slots=True)
class PaperActorCriticConfig:
    """Architecture from Shai et al., arXiv:2405.15943."""

    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 1
    d_head: int = 8
    d_mlp: int = 256
    context_length: int = 10
    max_seq_len: int = 32
    activation: str = "relu"
    normalization: str = "layer_norm"
    positional_embedding: str = "learned_absolute"

    def __post_init__(self) -> None:
        if min(
            self.d_model,
            self.n_layers,
            self.d_head,
            self.d_mlp,
            self.context_length,
            self.max_seq_len,
        ) <= 0:
            raise ValueError("paper transformer dimensions must be positive")
        if self.activation != "relu":
            raise ValueError("the paper architecture requires ReLU")
        if self.normalization != "layer_norm":
            raise ValueError("the paper architecture requires LayerNorm")
        if self.positional_embedding != "learned_absolute":
            raise ValueError(
                "the paper architecture requires learned absolute positions"
            )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PaperActorCriticConfig:
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in names})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SingleHeadCausalAttention(nn.Module):
    """Causal multi-head attention with explicit per-head width."""

    def __init__(self, config: PaperActorCriticConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        inner_width = config.n_heads * config.d_head
        self.query = nn.Linear(config.d_model, inner_width)
        self.key = nn.Linear(config.d_model, inner_width)
        self.value = nn.Linear(config.d_model, inner_width)
        self.output = nn.Linear(inner_width, config.d_model)

    def forward(
        self,
        inputs: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length, _ = inputs.shape

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch_size,
                length,
                self.n_heads,
                self.d_head,
            ).transpose(1, 2)

        query = split_heads(self.query(inputs))
        key = split_heads(self.key(inputs))
        value = split_heads(self.value(inputs))
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.d_head)
        scores = scores.masked_fill(~allowed[:, None], -torch.inf)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            length,
            self.n_heads * self.d_head,
        )
        return self.output(attended)


class PaperTransformerBlock(nn.Module):
    """Pre-LayerNorm attention and ReLU MLP residual block."""

    def __init__(self, config: PaperActorCriticConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = SingleHeadCausalAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_mlp),
            nn.ReLU(),
            nn.Linear(config.d_mlp, config.d_model),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs), allowed)
        return hidden + self.mlp(self.mlp_norm(hidden))


class PaperResidualEncoder(nn.Module):
    """Vectorized strict-window encoder with no host-device transfers."""

    def __init__(self, obs_dim: int, config: PaperActorCriticConfig) -> None:
        super().__init__()
        self.config = config
        self.input_embedding = nn.Linear(obs_dim, config.d_model, bias=False)
        self.position_embedding = nn.Embedding(
            config.context_length,
            config.d_model,
        )
        self.blocks = nn.ModuleList(
            PaperTransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        context: torch.Tensor,
        context_lengths: torch.Tensor,
        observations: torch.Tensor,
        *,
        apply_final_norm: bool = True,
    ) -> torch.Tensor:
        """Encode every observation from its own right-aligned context window."""

        if observations.ndim != 3:
            raise ValueError("observations must have shape (B, T, D)")
        batch_size, steps, _ = observations.shape
        window = self.config.context_length
        combined = torch.cat([context, observations], dim=1)
        windows = combined.unfold(1, window, 1).permute(0, 1, 3, 2)
        if windows.shape[1] != steps:
            raise RuntimeError("context state does not match configured window")
        flat_windows = windows.reshape(batch_size * steps, window, -1)

        offsets = torch.arange(steps, device=observations.device)
        valid_lengths = (
            context_lengths.reshape(-1, 1).to(dtype=torch.long)
            + offsets.reshape(1, -1)
            + 1
        ).clamp(max=window)
        valid_lengths = valid_lengths.reshape(-1)
        slots = torch.arange(window, device=observations.device)
        first_valid = window - valid_lengths
        valid = slots.reshape(1, -1) >= first_valid.reshape(-1, 1)
        positions = (slots.reshape(1, -1) - first_valid.reshape(-1, 1)).clamp_min(
            0
        )

        hidden = self.input_embedding(flat_windows)
        hidden = hidden + self.position_embedding(positions)
        causal = slots.reshape(-1, 1) >= slots.reshape(1, -1)
        allowed = causal.reshape(1, window, window) & valid.reshape(
            -1, 1, window
        )
        diagonal = torch.eye(
            window,
            dtype=torch.bool,
            device=observations.device,
        ).reshape(1, window, window)
        allowed = allowed | diagonal
        for block in self.blocks:
            hidden = block(hidden, allowed)
        if apply_final_norm:
            hidden = self.final_norm(hidden)
        encoded = hidden[:, -1, :]
        return encoded.reshape(batch_size, steps, self.config.d_model)


class PaperActorCriticModel(BaseActorCriticModel):
    """Stateful actor-critic using the exact paper-scale residual stream."""

    def _build_encoder(self) -> int:
        self.paper_config = PaperActorCriticConfig.from_dict(
            dict(self.model_config)
        )
        self._obs_dim = int(np.prod(self.observation_space.shape))
        self.encoder = PaperResidualEncoder(self._obs_dim, self.paper_config)
        return self.paper_config.d_model

    @property
    def sequence_lookback(self) -> int:
        return self.paper_config.context_length - 1

    def get_initial_state(self) -> dict[str, np.ndarray]:
        return {
            "ctx": np.zeros(
                (self.sequence_lookback, self._obs_dim),
                dtype=np.float32,
            ),
            "len": np.zeros((1,), dtype=np.float32),
        }

    def _advance_context(
        self,
        observations: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        sequence = torch.cat([state["ctx"], observations], dim=1)
        lengths = state["len"].reshape(-1) + observations.shape[1]
        return {
            "ctx": sequence[:, -self.sequence_lookback :, :],
            "len": lengths.clamp(max=self.sequence_lookback).reshape(-1, 1),
        }

    def _encode(
        self,
        batch: dict[str, Any],
        *,
        apply_final_norm: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        observations = batch[Columns.OBS]
        state = batch[Columns.STATE_IN]
        embeddings = self.encoder(
            state["ctx"],
            state["len"].reshape(-1),
            observations,
            apply_final_norm=apply_final_norm,
        )
        return embeddings, self._advance_context(observations, state)

    def _encode_train(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self._encode(batch)

    def _encode_rollout(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self._encode(batch)

    @torch.no_grad()
    def encode_step(
        self,
        observation: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embeddings, state_out = self._encode_rollout(
            {
                Columns.OBS: observation.unsqueeze(1),
                Columns.STATE_IN: state,
            }
        )
        return embeddings[:, 0, :], state_out

    @torch.no_grad()
    def encode_step_pre_final_norm(
        self,
        observation: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return final-block residuals before the output LayerNorm."""

        embeddings, state_out = self._encode(
            {
                Columns.OBS: observation.unsqueeze(1),
                Columns.STATE_IN: state,
            },
            apply_final_norm=False,
        )
        return embeddings[:, 0, :], state_out
