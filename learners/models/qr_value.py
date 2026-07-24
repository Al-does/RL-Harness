"""Composable fixed-quantile value critic for actor-critic RLModules."""

from __future__ import annotations

from typing import Any

import torch
from ray.rllib.core.columns import Columns
from torch import nn

from learners.components.quantile import QRValueConfig, QRValueHead


NAMESPACE = "qr_value"
FWD_QUANTILES = f"{NAMESPACE}/quantiles"


class QRValueMixin:
    """Replace a compatible actor-critic model's scalar value head with QR.

    This uses the fixed quantile fractions from QR-DQN as a PPO value critic.
    The superclass must follow the :class:`BaseActorCriticModel` contract.
    """

    def setup(self) -> None:
        super().setup()
        self.qr_value_config = QRValueConfig.from_dict(
            dict(self.model_config.get(NAMESPACE, {}))
        )
        self.qr_value_head = QRValueHead(
            self._embedding_dim,
            num_quantiles=self.qr_value_config.num_quantiles,
        )
        self.heads.value = nn.Identity()

    def _forward_train(self, batch, **kwargs):
        outputs = super()._forward_train(batch, **kwargs)
        outputs[FWD_QUANTILES] = self.qr_value_head(outputs[Columns.EMBEDDINGS])
        return outputs

    def compute_values(
        self,
        batch: dict[str, Any],
        embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embeddings is None:
            embeddings, _ = self._encode_train(batch)
        return self.qr_value_head(embeddings).mean(dim=-1)
