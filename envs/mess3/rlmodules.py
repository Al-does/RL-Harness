"""Custom RLModules for the MESS3-Control program (RLlib new API stack).

Mess3TransformerRLModule
    The canonical policy core: a small causal transformer over the per-step
    observation stream, with HARD-BANDED causal attention (every position
    attends to at most the previous ``context_len`` steps, per layer) and
    RoPE (so attention depends on relative position only).

    Chunking-invariance (the load-bearing property).  With a band of K and
    n_layers = L, the layer-L output at time t depends on raw observations
    o_{t-LK..t} ONLY, and recomputing over any window that includes those LK
    lookback steps reproduces the infinite-past computation exactly (the band
    cuts every dependency path at K per layer).  The module is therefore
    exposed to RLlib as a STATEFUL module whose recurrent state is a rolling
    buffer of the last L*K raw observations (plus a valid-length counter):
    the train-time forward over zero-padded ``max_seq_len`` chunks (state
    from chunk boundaries) computes EXACTLY the same function as the
    step-by-step rollout forward — unit-tested in tests/test_rlmodules.py.
    Without the lookback (buffer of K only) the two forwards genuinely
    diverge for L >= 2, which would bias PPO's importance ratios.

    Heads: diagonal-Gaussian policy (Box) or categorical policy (Discrete),
    value, and an auxiliary head (3 logits) used for next-token prediction
    (Phase 3 aux arms) or true-state classification (the supervised twin).

Mess3MLPRLModule
    Memoryless baseline core for the reactive / frame-stack / oracle /
    belief-observation arms.  Same head structure (so the probe pipeline can
    treat every arm uniformly through ``encode``).

Both modules expose ``encode(obs, state)`` returning the pre-head embedding at
decision time — the probe's activation target.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override

AUX_LOGITS = "aux_logits"
N_AUX_CLASSES = 3  # tokens or states, both 3-way in this program


# ---------------------------------------------------------------------------
# Transformer internals
# ---------------------------------------------------------------------------


def _rope_cos_sin(
    positions: torch.Tensor, dim: int, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (len(positions), dim/2) at the given positions.

    RoPE is relative: the q.k dot product depends only on position DIFFERENCES,
    so absolute offsets cancel.  This lets the cached rollout path (which uses a
    small fixed position window) match the train-time windowed recompute exactly,
    even though the two assign different absolute indices to the same step.
    """
    half = dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(half, device=device, dtype=dtype) / half))
    ang = positions.to(dtype)[:, None] * inv_freq[None, :]
    return torch.cos(ang), torch.sin(ang)


def _rope_angles(L: int, dim: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (L, dim/2) for positions 0..L-1."""
    return _rope_cos_sin(torch.arange(L, device=device), dim, device, dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, L, hd).  Rotate pairs (even, odd) by the positional angle."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    # cos/sin: (L, hd/2) -> broadcast over (B, H).
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = xr1
    out[..., 1::2] = xr2
    return out


class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, attn_mask, cos, sin):
        B, L, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)

        def split(t):
            return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        att = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask  # (B, 1, L, L) bool: True = attend
        )
        att = att.transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(att)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_step(self, x_t, k_cache, v_cache, valid, cos_q, sin_q, cos_k, sin_k):
        """One-position incremental forward against a rolling K/V cache.

        x_t: (B, 1, D) input for the current position;
        k_cache/v_cache: (B, H, C, hd) RAW (pre-RoPE) keys/values, oldest->newest;
        valid: (B, 1, 1, C) bool, True for filled cache slots;
        cos_q/sin_q: (1, hd/2) for the query position; cos_k/sin_k: (C, hd/2) for
        the C cache-slot positions.  Returns (x_t', k_cache', v_cache') where the
        new position's RAW k/v have been rolled into the caches.

        Computed to match ``forward``'s banded-attention math exactly: same
        1/sqrt(hd) scaling and softmax, same relative RoPE offsets.
        """
        B, _, D = x_t.shape
        h = self.ln1(x_t)
        q, k, v = self.qkv(h).chunk(3, dim=-1)

        def split(t):
            return t.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,1,hd)

        q, k, v = split(q), split(k), split(v)
        # Roll the new RAW k/v into the cache (drop the oldest slot).
        k_cache = torch.cat([k_cache[:, :, 1:, :], k], dim=2)  # (B,H,C,hd)
        v_cache = torch.cat([v_cache[:, :, 1:, :], v], dim=2)
        q_r = _apply_rope(q, cos_q, sin_q)          # (B,H,1,hd)
        k_r = _apply_rope(k_cache, cos_k, sin_k)    # (B,H,C,hd)
        att = (q_r @ k_r.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,1,C)
        att = att.masked_fill(~valid, float("-inf"))
        att = att.softmax(dim=-1) @ v_cache         # (B,H,1,hd)
        att = att.transpose(1, 2).reshape(B, 1, D)
        x_t = x_t + self.proj(att)
        x_t = x_t + self.mlp(self.ln2(x_t))
        return x_t, k_cache, v_cache


class _CausalTransformer(nn.Module):
    """Banded-causal RoPE transformer over (lookback buffer + chunk) windows.

    Band width = ``context_len`` per layer, so the receptive field of the
    stack is ``n_layers * context_len``; the lookback buffer holds exactly
    that many raw past observations, which makes any windowed recompute agree
    exactly with the infinite-past computation at all chunk positions.
    """

    def __init__(self, obs_dim: int, d_model: int, n_layers: int, n_heads: int, context_len: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.K = context_len                 # attention band per layer
        self.cache_len = context_len + 1     # keys in a layer's band: self + K back
        self.lookback = n_layers * context_len  # recurrent buffer length
        self.inp = nn.Linear(obs_dim, d_model)
        self.blocks = nn.ModuleList([_Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def _mask(self, B: int, L: int, lens: torch.Tensor, device) -> torch.Tensor:
        """(B, 1, L, L) bool attention mask.

        Row m may attend col n iff n <= m (causal), m - n <= K (band), and the
        col is a valid key: chunk positions (n >= lookback) always; buffer
        positions only the last ``len`` of them (n >= lookback - len).  The
        diagonal is always allowed so padding rows attend themselves (finite
        garbage that is itself masked out as a key everywhere else).
        """
        idx = torch.arange(L, device=device)
        causal_band = (idx[None, :] <= idx[:, None]) & (idx[:, None] - idx[None, :] <= self.K)
        key_valid = idx[None, :] >= (self.lookback - lens.view(B, 1))  # (B, L)
        mask = causal_band[None, :, :] & key_valid[:, None, :]
        mask |= torch.eye(L, dtype=torch.bool, device=device)[None]
        return mask[:, None, :, :]

    def forward(self, ctx: torch.Tensor, lens: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """ctx: (B, lookback, obs_dim); lens: (B,); obs: (B, T, obs_dim) -> (B, T, d_model)."""
        B, T, _ = obs.shape
        seq = torch.cat([ctx, obs], dim=1)
        L = self.lookback + T
        x = self.inp(seq)
        head_dim = self.d_model // self.blocks[0].n_heads
        cos, sin = _rope_angles(L, head_dim, x.device, x.dtype)
        mask = self._mask(B, L, lens, x.device)
        for blk in self.blocks:
            x = blk(x, mask, cos, sin)
        return self.ln_f(x)[:, self.lookback:, :]

    def forward_cached(self, kv_k, kv_v, kv_len, obs):
        """Incremental rollout forward using per-layer RAW K/V caches.

        Replaces the O(lookback) windowed recompute of ``forward`` with an
        O(n_layers * cache_len) step: each new position attends only to the last
        ``cache_len`` keys per layer (the band), so the cache is bounded by K,
        not by the full lookback.  Produces embeddings identical (to float
        tolerance) to ``forward`` -- pinned by test_rlmodules.py.

        kv_k/kv_v: (B, n_layers, H, cache_len, hd) RAW keys/values, oldest->newest.
        kv_len: (B,) filled-slot count per env (resets to 0 at episode start).
        obs: (B, T, obs_dim).  Returns (emb (B, T, d_model), kv_k', kv_v', kv_len').
        """
        B, T, _ = obs.shape
        device, dtype = obs.device, obs.dtype
        C = self.cache_len
        # Anchor the query at absolute index ``lookback`` and the C cache slots at
        # ``lookback-K .. lookback`` -- exactly the newest C indices the windowed
        # T=1 path assigns, so relative RoPE offsets (hence attention) match.
        p_q = self.lookback
        key_pos = torch.arange(p_q - self.K, p_q + 1, device=device)  # (C,)
        cos_k, sin_k = _rope_cos_sin(key_pos, self.head_dim, device, dtype)
        cos_q, sin_q = _rope_cos_sin(
            torch.tensor([p_q], device=device), self.head_dim, device, dtype
        )
        slot = torch.arange(C, device=device)  # (C,)
        embs = []
        for t in range(T):
            x_t = self.inp(obs[:, t : t + 1, :])  # (B,1,D)
            kv_len = torch.clamp(kv_len + 1.0, max=float(C))
            # Valid = the last ``kv_len`` slots (newest-aligned), mirroring the
            # windowed key_valid mask so the cold-start (short history) matches.
            valid = (slot[None, :] >= (C - kv_len[:, None])).view(B, 1, 1, C)
            new_k, new_v = [], []
            for lyr in range(self.n_layers):
                x_t, k_c, v_c = self.blocks[lyr].forward_step(
                    x_t, kv_k[:, lyr], kv_v[:, lyr], valid, cos_q, sin_q, cos_k, sin_k
                )
                new_k.append(k_c)
                new_v.append(v_c)
            kv_k = torch.stack(new_k, dim=1)  # (B, n_layers, H, C, hd)
            kv_v = torch.stack(new_v, dim=1)
            embs.append(self.ln_f(x_t))  # (B,1,D)
        return torch.cat(embs, dim=1), kv_k, kv_v, kv_len


# ---------------------------------------------------------------------------
# RLModules
# ---------------------------------------------------------------------------


def _build_heads(module: nn.Module, d: int, action_space: gym.Space):
    if isinstance(action_space, gym.spaces.Box):
        module._act_dim = int(np.prod(action_space.shape))
        module._pi_mean = nn.Linear(d, module._act_dim)
        module._log_std = nn.Parameter(torch.zeros(module._act_dim))
        module._discrete = False
    elif isinstance(action_space, gym.spaces.Discrete):
        module._pi_logits = nn.Linear(d, int(action_space.n))
        module._discrete = True
    else:
        raise ValueError(f"unsupported action space {action_space}")
    module._value = nn.Linear(d, 1)
    module._aux = nn.Linear(d, N_AUX_CLASSES)


def _pi_out(module: nn.Module, emb: torch.Tensor) -> torch.Tensor:
    if module._discrete:
        return module._pi_logits(emb)
    mean = module._pi_mean(emb)
    log_std = module._log_std.expand_as(mean)
    return torch.cat([mean, log_std], dim=-1)


class Mess3TransformerRLModule(TorchRLModule, ValueFunctionAPI):
    """Stateful causal-transformer policy core (see module docstring)."""

    @override(TorchRLModule)
    def setup(self):
        cfg = self.model_config
        d_model = int(cfg.get("d_model", 96))
        n_layers = int(cfg.get("n_layers", 3))
        n_heads = int(cfg.get("n_heads", 4))
        context_len = int(cfg.get("context_len", 64))
        self._obs_dim = int(self.observation_space.shape[0])
        self.core = _CausalTransformer(self._obs_dim, d_model, n_layers, n_heads, context_len)
        self._lookback = self.core.lookback
        _build_heads(self, d_model, self.action_space)

    @override(TorchRLModule)
    def get_initial_state(self) -> Any:
        core = self.core
        cache_shape = (core.n_layers, core.n_heads, core.cache_len, core.head_dim)
        return {
            # Raw-obs lookback buffer: the TRAIN-time windowed recompute reads
            # this (unchanged). Kept up to date during rollout too, so the learner
            # gets a correct chunk-start state at every max_seq_len boundary.
            "ctx": np.zeros((self._lookback, self._obs_dim), dtype=np.float32),
            "len": np.zeros((1,), dtype=np.float32),
            # Per-layer RAW K/V caches: the ROLLOUT-time incremental forward reads
            # these (train ignores them). Bounded by the band K, not the lookback.
            "kv_k": np.zeros(cache_shape, dtype=np.float32),
            "kv_v": np.zeros(cache_shape, dtype=np.float32),
            "kv_len": np.zeros((1,), dtype=np.float32),
        }

    def _advance_ctx(self, obs, state):
        """Roll ``obs`` into the raw-obs lookback buffer; return (seq, ctx', len')."""
        ctx, lens = state["ctx"], state["len"].reshape(-1)
        T = obs.shape[1]
        seq = torch.cat([ctx, obs], dim=1)
        return (
            seq,
            seq[:, -self._lookback:, :],
            torch.clamp(lens + T, max=float(self._lookback)).reshape(-1, 1),
        )

    def _core_forward_train(self, batch):
        """Windowed recompute over the raw-obs buffer (correct chunk gradients)."""
        obs = batch[Columns.OBS]
        state = batch[Columns.STATE_IN]
        ctx, lens = state["ctx"], state["len"].reshape(-1)
        emb = self.core(ctx, lens, obs)
        _, ctx_out, len_out = self._advance_ctx(obs, state)
        state_out = {"ctx": ctx_out, "len": len_out}
        # Pass the cache leaves through unchanged so the state schema is stable
        # across the rollout->train boundary (train never reads them).
        for key in ("kv_k", "kv_v", "kv_len"):
            if key in state:
                state_out[key] = state[key]
        return emb, state_out

    def _core_forward_rollout(self, batch):
        """Incremental K/V-cached forward (fast rollout; matches the recompute)."""
        obs = batch[Columns.OBS]
        state = batch[Columns.STATE_IN]
        emb, kv_k, kv_v, kv_len = self.core.forward_cached(
            state["kv_k"], state["kv_v"], state["kv_len"].reshape(-1), obs
        )
        _, ctx_out, len_out = self._advance_ctx(obs, state)
        state_out = {
            "ctx": ctx_out,
            "len": len_out,
            "kv_k": kv_k,
            "kv_v": kv_v,
            "kv_len": kv_len.reshape(-1, 1),
        }
        return emb, state_out

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        emb, state_out = self._core_forward_rollout(batch)
        return {
            Columns.ACTION_DIST_INPUTS: _pi_out(self, emb),
            Columns.STATE_OUT: state_out,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        emb, state_out = self._core_forward_train(batch)
        return {
            Columns.ACTION_DIST_INPUTS: _pi_out(self, emb),
            Columns.STATE_OUT: state_out,
            Columns.EMBEDDINGS: emb,
            AUX_LOGITS: self._aux(emb),
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None):
        if embeddings is None:
            embeddings, _ = self._core_forward_train(batch)
        return self._value(embeddings).squeeze(-1)

    # -- probe / supervised-twin interface ---------------------------------

    @torch.no_grad()
    def encode_step(self, obs: torch.Tensor, state: dict) -> tuple[torch.Tensor, dict]:
        """One decision step: obs (B, obs_dim) + state -> (embedding (B, d), state').

        Uses the incremental K/V-cache path (same as env-runner rollout), so the
        probe sees exactly the embeddings the agent computed while acting.
        """
        batch = {Columns.OBS: obs.unsqueeze(1), Columns.STATE_IN: state}
        emb, state_out = self._core_forward_rollout(batch)
        return emb[:, 0, :], state_out

    def encode_chunks(self, ctx: torch.Tensor, lens: torch.Tensor, obs: torch.Tensor):
        """Differentiable chunk forward for the supervised trainer."""
        return self.core(ctx, lens, obs)


class Mess3MLPRLModule(TorchRLModule, ValueFunctionAPI):
    """Memoryless MLP core with the same head/probe interface."""

    @override(TorchRLModule)
    def setup(self):
        hidden = list(self.model_config.get("mlp_hidden", (128, 128)))
        in_dim = int(self.observation_space.shape[0])
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.core = nn.Sequential(*layers)
        _build_heads(self, in_dim, self.action_space)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        emb = self.core(batch[Columns.OBS])
        return {Columns.ACTION_DIST_INPUTS: _pi_out(self, emb)}

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        emb = self.core(batch[Columns.OBS])
        return {
            Columns.ACTION_DIST_INPUTS: _pi_out(self, emb),
            Columns.EMBEDDINGS: emb,
            AUX_LOGITS: self._aux(emb),
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None):
        if embeddings is None:
            embeddings = self.core(batch[Columns.OBS])
        return self._value(embeddings).squeeze(-1)

    @torch.no_grad()
    def encode_step(self, obs: torch.Tensor, state: dict | None = None):
        return self.core(obs), state
