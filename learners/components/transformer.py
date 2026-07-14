"""Banded causal transformer components with equivalent train and cached paths."""

from __future__ import annotations

import math

import torch
from torch import nn


def _rope_cos_sin(
    positions: torch.Tensor, dim: int, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    half = dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(half, device=device, dtype=dtype) / half)
    )
    angles = positions.to(dtype)[:, None] * inv_freq[None, :]
    return torch.cos(angles), torch.sin(angles)


def _rope_angles(
    length: int, dim: int, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    return _rope_cos_sin(torch.arange(length, device=device), dim, device, dtype)


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, attn_mask, cos, sin):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)

        def split(tensor):
            return tensor.view(
                batch, length, self.n_heads, self.head_dim
            ).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        attention = torch.nn.functional.scaled_dot_product_attention(
            _apply_rope(q, cos, sin),
            _apply_rope(k, cos, sin),
            v,
            attn_mask=attn_mask,
        )
        attention = attention.transpose(1, 2).reshape(batch, length, width)
        x = x + self.proj(attention)
        return x + self.mlp(self.ln2(x))

    def forward_step(
        self,
        x_t,
        k_cache,
        v_cache,
        valid,
        cos_q,
        sin_q,
        cos_k,
        sin_k,
    ):
        batch, _, width = x_t.shape
        q, k, v = self.qkv(self.ln1(x_t)).chunk(3, dim=-1)

        def split(tensor):
            return tensor.view(
                batch, 1, self.n_heads, self.head_dim
            ).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        k_cache = torch.cat([k_cache[:, :, 1:, :], k], dim=2)
        v_cache = torch.cat([v_cache[:, :, 1:, :], v], dim=2)
        q = _apply_rope(q, cos_q, sin_q)
        k = _apply_rope(k_cache, cos_k, sin_k)
        attention = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = attention.masked_fill(~valid, float("-inf"))
        attention = attention.softmax(dim=-1) @ v_cache
        attention = attention.transpose(1, 2).reshape(batch, 1, width)
        x_t = x_t + self.proj(attention)
        x_t = x_t + self.mlp(self.ln2(x_t))
        return x_t, k_cache, v_cache


class CausalTransformerEncoder(nn.Module):
    """RoPE transformer with a hard causal band at each layer.

    The raw-observation lookback is ``n_layers * context_len``. Recomputing
    over that lookback gives the same embeddings as incremental KV-cached
    rollout, which is required for correct PPO ratios.
    """

    def __init__(
        self,
        obs_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        context_len: int,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.context_len = context_len
        self.cache_len = context_len + 1
        self.lookback = n_layers * context_len
        self.input_projection = nn.Linear(obs_dim, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def _mask(
        self, batch: int, length: int, lens: torch.Tensor, device
    ) -> torch.Tensor:
        indices = torch.arange(length, device=device)
        causal_band = (indices[None, :] <= indices[:, None]) & (
            indices[:, None] - indices[None, :] <= self.context_len
        )
        key_valid = indices[None, :] >= (
            self.lookback - lens.view(batch, 1)
        )
        mask = causal_band[None, :, :] & key_valid[:, None, :]
        mask |= torch.eye(length, dtype=torch.bool, device=device)[None]
        return mask[:, None, :, :]

    def forward(
        self, context: torch.Tensor, lens: torch.Tensor, obs: torch.Tensor
    ) -> torch.Tensor:
        batch, chunk_len, _ = obs.shape
        sequence = torch.cat([context, obs], dim=1)
        length = self.lookback + chunk_len
        x = self.input_projection(sequence)
        cos, sin = _rope_angles(length, self.head_dim, x.device, x.dtype)
        mask = self._mask(batch, length, lens, x.device)
        for block in self.blocks:
            x = block(x, mask, cos, sin)
        return self.final_norm(x)[:, self.lookback :, :]

    def forward_cached(self, kv_k, kv_v, kv_len, obs):
        batch, chunk_len, _ = obs.shape
        device, dtype = obs.device, obs.dtype
        cache_len = self.cache_len
        query_position = self.lookback
        key_positions = torch.arange(
            query_position - self.context_len,
            query_position + 1,
            device=device,
        )
        cos_k, sin_k = _rope_cos_sin(
            key_positions, self.head_dim, device, dtype
        )
        cos_q, sin_q = _rope_cos_sin(
            torch.tensor([query_position], device=device),
            self.head_dim,
            device,
            dtype,
        )
        slots = torch.arange(cache_len, device=device)
        embeddings = []
        for timestep in range(chunk_len):
            x_t = self.input_projection(obs[:, timestep : timestep + 1, :])
            kv_len = torch.clamp(kv_len + 1.0, max=float(cache_len))
            valid = (
                slots[None, :] >= (cache_len - kv_len[:, None])
            ).view(batch, 1, 1, cache_len)
            next_k, next_v = [], []
            for layer, block in enumerate(self.blocks):
                x_t, k_cache, v_cache = block.forward_step(
                    x_t,
                    kv_k[:, layer],
                    kv_v[:, layer],
                    valid,
                    cos_q,
                    sin_q,
                    cos_k,
                    sin_k,
                )
                next_k.append(k_cache)
                next_v.append(v_cache)
            kv_k = torch.stack(next_k, dim=1)
            kv_v = torch.stack(next_v, dim=1)
            embeddings.append(self.final_norm(x_t))
        return torch.cat(embeddings, dim=1), kv_k, kv_v, kv_len
