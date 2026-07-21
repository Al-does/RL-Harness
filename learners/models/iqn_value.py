"""RLModule composition for an implicit-quantile value critic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from ray.rllib.core.columns import Columns

from learners.components.iqn import IQNValueHead
from learners.models.mlp import MLPModel
from learners.models.transformer import TransformerModel

NAMESPACE = "iqn_value"
FWD_QUANTILES = f"{NAMESPACE}/quantiles"
FWD_TAUS = f"{NAMESPACE}/taus"


@dataclass(frozen=True)
class IQNValueConfig:
    """Validated model-side configuration for an IQN value critic."""

    train_quantiles: int = 32
    value_quantiles: int = 64
    n_cosines: int = 64

    def __post_init__(self) -> None:
        if (
            self.train_quantiles <= 0
            or self.value_quantiles <= 0
            or self.n_cosines <= 0
        ):
            raise ValueError("IQN quantile counts and dimensions must be positive")

    @classmethod
    def from_dict(cls, values: dict) -> "IQNValueConfig":
        known_fields = {
            "train_quantiles",
            "value_quantiles",
            "n_cosines",
        }
        unknown_fields = set(values) - known_fields
        if unknown_fields:
            names = ", ".join(sorted(map(str, unknown_fields)))
            raise ValueError(f"unknown IQN value config fields: {names}")
        return cls(
            train_quantiles=int(values.get("train_quantiles", 32)),
            value_quantiles=int(values.get("value_quantiles", 64)),
            n_cosines=int(values.get("n_cosines", 64)),
        )

    def to_dict(self) -> dict:
        return asdict(self)


class IQNValueMixin:
    """Replace a scalar critic with a sampled return-quantile distribution."""

    def setup(self) -> None:
        super().setup()
        self.iqn_value_config = IQNValueConfig.from_dict(
            dict(self.model_config.get(NAMESPACE, {}))
        )
        self.iqn_value_head = IQNValueHead(
            self._embedding_dim,
            n_cosines=self.iqn_value_config.n_cosines,
        )
        # Keep the actor-critic container API while removing unused parameters.
        self.heads.value = nn.Identity()

    @staticmethod
    def _sample_taus(
        embeddings: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        return torch.rand(
            (*embeddings.shape[:-1], count),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )

    @staticmethod
    def _fixed_taus(
        embeddings: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        midpoints = (
            torch.arange(
                count,
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            + 0.5
        ) / count
        return midpoints.expand(*embeddings.shape[:-1], count)

    def _forward_train(self, batch, **kwargs):
        outputs = super()._forward_train(batch, **kwargs)
        embeddings = outputs[Columns.EMBEDDINGS]
        taus = self._sample_taus(
            embeddings,
            self.iqn_value_config.train_quantiles,
        )
        outputs[FWD_TAUS] = taus
        outputs[FWD_QUANTILES] = self.iqn_value_head(embeddings, taus)
        return outputs

    def compute_values(
        self,
        batch: dict[str, Any],
        embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Estimate the scalar baseline using fixed quantile midpoints."""

        if embeddings is None:
            embeddings, _ = self._encode_train(batch)
        taus = self._fixed_taus(
            embeddings,
            self.iqn_value_config.value_quantiles,
        )
        return self.iqn_value_head(embeddings, taus).mean(dim=-1)


class IQNTransformerModel(IQNValueMixin, TransformerModel):
    """Transformer actor with an IQN distributional value critic."""


class IQNMLPModel(IQNValueMixin, MLPModel):
    """Memoryless MLP actor with an IQN distributional value critic."""
