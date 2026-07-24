"""Composable implicit-quantile value critic for actor-critic RLModules."""

from __future__ import annotations

from typing import Any

from torch import nn
import torch
from ray.rllib.core.columns import Columns

from learners.components.quantile import (
    IQNValueConfig,
    IQNValueHead,
    midpoint_taus,
    sample_taus,
)


NAMESPACE = "iqn_value"
FWD_QUANTILES = f"{NAMESPACE}/quantiles"
FWD_TAUS = f"{NAMESPACE}/taus"


class IQNValueMixin:
    """Replace a compatible actor-critic model's scalar value head with IQN.

    The superclass must expose ``_embedding_dim``, ``heads.value``,
    ``_forward_train()``, and ``_encode_train()`` as
    :class:`BaseActorCriticModel` does.
    """

    def setup(self) -> None:
        super().setup()
        self.iqn_value_config = IQNValueConfig.from_dict(
            dict(self.model_config.get(NAMESPACE, {}))
        )
        self.iqn_value_head = IQNValueHead(
            self._embedding_dim,
            n_cosines=self.iqn_value_config.n_cosines,
        )
        # PPO's scalar value layer is replaced rather than trained in parallel.
        self.heads.value = nn.Identity()

    def _forward_train(self, batch, **kwargs):
        outputs = super()._forward_train(batch, **kwargs)
        embeddings = outputs[Columns.EMBEDDINGS]
        taus = sample_taus(
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
        if embeddings is None:
            embeddings, _ = self._encode_train(batch)
        taus = midpoint_taus(
            embeddings,
            self.iqn_value_config.value_quantiles,
        )
        return self.iqn_value_head(embeddings, taus).mean(dim=-1)
