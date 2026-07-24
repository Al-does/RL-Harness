"""Device-native implicit-quantile value components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class IQNValueConfig:
    """Configuration for an implicit-quantile scalar value head."""

    train_quantiles: int = 32
    value_quantiles: int = 64
    n_cosines: int = 64

    def __post_init__(self) -> None:
        if (
            self.train_quantiles <= 0
            or self.value_quantiles <= 0
            or self.n_cosines <= 0
        ):
            raise ValueError("IQN quantile counts and cosine width must be positive")

    @classmethod
    def from_dict(cls, values: dict) -> "IQNValueConfig":
        own_fields = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in own_fields})

    def to_dict(self) -> dict:
        return asdict(self)


class IQNValueHead(nn.Module):
    """Map encoder features and quantile fractions to scalar return quantiles."""

    def __init__(self, embedding_dim: int, *, n_cosines: int = 64) -> None:
        super().__init__()
        if embedding_dim <= 0 or n_cosines <= 0:
            raise ValueError("IQN dimensions must be positive")
        self.embedding_dim = int(embedding_dim)
        self.n_cosines = int(n_cosines)
        self.cosine_projection = nn.Linear(self.n_cosines, self.embedding_dim)
        self.output = nn.Linear(self.embedding_dim, 1)
        # Preserve the basis used by the promoted experiments for reproducibility.
        self.register_buffer(
            "cosine_frequencies",
            torch.arange(1, self.n_cosines + 1, dtype=torch.float32) * torch.pi,
            persistent=False,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        taus: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError("embedding width does not match the IQN head")
        if embeddings.shape[:-1] != taus.shape[:-1]:
            raise ValueError("embedding and tau leading dimensions must match")
        cosine_features = torch.cos(
            taus.unsqueeze(-1)
            * self.cosine_frequencies.to(dtype=embeddings.dtype)
        )
        tau_embeddings = torch.relu(self.cosine_projection(cosine_features))
        joint = embeddings.unsqueeze(-2) * tau_embeddings
        return self.output(joint).squeeze(-1)


def sample_taus(reference: torch.Tensor, count: int) -> torch.Tensor:
    """Sample uniform quantile fractions matching a feature tensor's device."""

    if count <= 0:
        raise ValueError("quantile count must be positive")
    return torch.rand(
        (*reference.shape[:-1], count),
        dtype=reference.dtype,
        device=reference.device,
    )


def midpoint_taus(reference: torch.Tensor, count: int) -> torch.Tensor:
    """Return deterministic equal-mass midpoint fractions on-device."""

    if count <= 0:
        raise ValueError("quantile count must be positive")
    midpoints = (
        torch.arange(count, dtype=reference.dtype, device=reference.device) + 0.5
    ) / count
    return midpoints.expand(*reference.shape[:-1], count)
