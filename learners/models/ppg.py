"""RLModule composition support for single-network Phasic Policy Gradient."""

from __future__ import annotations

import torch
from ray.rllib.core.columns import Columns

from learners.models.mlp import MLPModel
from learners.models.transformer import TransformerModel


NAMESPACE = "ppg"
AUX_VALUE_PREDICTIONS = f"{NAMESPACE}/aux_value_predictions"


class PPGAuxiliaryValueHead:
    """Add PPG's training-only auxiliary value head to an actor-critic module.

    Compose this mixin before a module that exposes ``_embedding_dim`` during
    ``setup()`` and writes ``Columns.EMBEDDINGS`` from ``_forward_train()``.
    The head is absent from rollout forwards, so inference cost is unchanged.
    """

    def setup(self) -> None:
        super().setup()
        self.ppg_auxiliary_value_head = torch.nn.Linear(self._embedding_dim, 1)

    def _forward_train(self, batch, **kwargs):
        outputs = super()._forward_train(batch, **kwargs)
        embeddings = outputs.get(Columns.EMBEDDINGS)
        if embeddings is None:
            raise KeyError(
                "PPGAuxiliaryValueHead requires the base module's training "
                f"forward to emit {Columns.EMBEDDINGS!r}"
            )
        outputs[AUX_VALUE_PREDICTIONS] = self.ppg_auxiliary_value_head(
            embeddings
        ).squeeze(-1)
        return outputs


class PPGMLPModel(PPGAuxiliaryValueHead, MLPModel):
    """Ready-to-use memoryless actor-critic module for PPG."""


class PPGTransformerModel(PPGAuxiliaryValueHead, TransformerModel):
    """Ready-to-use stateful transformer actor-critic module for PPG."""


__all__ = [
    "AUX_VALUE_PREDICTIONS",
    "NAMESPACE",
    "PPGAuxiliaryValueHead",
    "PPGMLPModel",
    "PPGTransformerModel",
]
