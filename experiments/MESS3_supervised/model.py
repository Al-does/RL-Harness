"""Small causal transformer matching the architecture reported by Shai et al."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CausalAttention(nn.Module):
    """One causal attention head with an eight-dimensional Q/K/V space."""

    def __init__(self, d_model: int, d_head: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, d_head)
        self.key = nn.Linear(d_model, d_head)
        self.value = nn.Linear(d_model, d_head)
        self.output = nn.Linear(d_head, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = self.query(inputs).unsqueeze(1)
        key = self.key(inputs).unsqueeze(1)
        value = self.value(inputs).unsqueeze(1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
        )
        return self.output(attended.squeeze(1))


class TransformerBlock(nn.Module):
    """TransformerLens-style sequential pre-LayerNorm block."""

    def __init__(self, d_model: int, d_head: int, d_mlp: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalAttention(d_model, d_head)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model),
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        residual = residual + self.attention(self.attention_norm(residual))
        return residual + self.mlp(self.mlp_norm(residual))


class PaperTransformer(nn.Module):
    """Four-layer, approximately 143k-parameter MESS3 transformer."""

    def __init__(
        self,
        *,
        vocab_size: int = 3,
        context_length: int = 10,
        d_model: int = 64,
        d_head: int = 8,
        d_mlp: int = 256,
        n_layers: int = 4,
        initialization_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, d_head, d_mlp)
            for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.unembedding = nn.Linear(d_model, vocab_size)
        self._initialize(initialization_scale)

    def _initialize(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=scale)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_residuals: bool = False,
    ):
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence)")
        if tokens.shape[1] > self.context_length:
            raise ValueError("sequence exceeds the configured context length")

        positions = torch.arange(tokens.shape[1], device=tokens.device)
        residual = self.token_embedding(tokens) + self.position_embedding(
            positions
        )
        residual_stream = []
        for block in self.blocks:
            residual = block(residual)
            if return_residuals:
                residual_stream.append(residual)

        normalized = self.final_norm(residual)
        logits = self.unembedding(normalized)
        if return_residuals:
            return logits, tuple(residual_stream), normalized
        return logits
