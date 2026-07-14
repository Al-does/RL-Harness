"""Thin RLlib template for actor-critic models composed from PyTorch parts."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override

from learners.components.heads import ActorCriticHeads


class BaseActorCriticModel(TorchRLModule, ValueFunctionAPI):
    """Assemble policy/value outputs while subclasses own encoding and state."""

    @override(TorchRLModule)
    def setup(self):
        self._embedding_dim = self._build_encoder()
        self.heads = ActorCriticHeads(self._embedding_dim, self.action_space)

    def _build_encoder(self) -> int:
        raise NotImplementedError

    def _encode_train(self, batch) -> tuple[torch.Tensor, Any | None]:
        raise NotImplementedError

    def _encode_rollout(self, batch) -> tuple[torch.Tensor, Any | None]:
        raise NotImplementedError

    def _outputs(
        self,
        embeddings: torch.Tensor,
        state_out: Any | None,
        *,
        training: bool,
    ) -> dict:
        outputs = {
            Columns.ACTION_DIST_INPUTS: self.heads.action_distribution_inputs(
                embeddings
            )
        }
        if state_out is not None:
            outputs[Columns.STATE_OUT] = state_out
        if training:
            outputs[Columns.EMBEDDINGS] = embeddings
        return outputs

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        embeddings, state_out = self._encode_rollout(batch)
        return self._outputs(embeddings, state_out, training=False)

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        embeddings, state_out = self._encode_train(batch)
        return self._outputs(embeddings, state_out, training=True)

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ):
        if embeddings is None:
            embeddings, _ = self._encode_train(batch)
        return self.heads.values(embeddings)

    def action_distribution_inputs(
        self, embeddings: torch.Tensor
    ) -> torch.Tensor:
        return self.heads.action_distribution_inputs(embeddings)
