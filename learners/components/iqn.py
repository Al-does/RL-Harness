"""Implicit-quantile neural-network components."""

from __future__ import annotations

import torch
from torch import nn


class IQNValueHead(nn.Module):
    """Predict scalar return quantiles conditioned on state embeddings."""

    def __init__(self, embedding_dim: int, *, n_cosines: int) -> None:
        super().__init__()
        if embedding_dim <= 0 or n_cosines <= 0:
            raise ValueError("IQN dimensions must be positive")

        self.embedding_dim = int(embedding_dim)
        self.n_cosines = int(n_cosines)
        self.cosine_projection = nn.Linear(n_cosines, embedding_dim)
        self.output = nn.Linear(embedding_dim, 1)
        self.register_buffer(
            "cosine_frequencies",
            torch.arange(1, n_cosines + 1, dtype=torch.float32) * torch.pi,
            persistent=False,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        taus: torch.Tensor,
    ) -> torch.Tensor:
        """Return quantiles with one output per input ``tau``."""

        if embeddings.shape[:-1] != taus.shape[:-1]:
            raise ValueError("embedding and tau leading dimensions must match")

        frequencies = self.cosine_frequencies.to(dtype=embeddings.dtype)
        cosine_features = torch.cos(taus.unsqueeze(-1) * frequencies)
        tau_embeddings = torch.relu(
            self.cosine_projection(cosine_features)
        )
        joint = embeddings.unsqueeze(-2) * tau_embeddings
        return self.output(joint).squeeze(-1)
