"""Complete RLlib model compositions and their typed configurations."""

from learners.components.iqn import IQNValueHead
from learners.models.iqn_value import (
    FWD_QUANTILES,
    FWD_TAUS,
    NAMESPACE as IQN_VALUE_NAMESPACE,
    IQNMLPModel,
    IQNTransformerModel,
    IQNValueConfig,
    IQNValueMixin,
)
from learners.models.mlp import MLPModel, MLPModelConfig
from learners.models.next_token import NextTokenAuxHead
from learners.models.state_aux import StateAuxHead
from learners.models.transformer import TransformerModel, TransformerModelConfig

__all__ = [
    "FWD_QUANTILES",
    "FWD_TAUS",
    "IQNMLPModel",
    "IQNTransformerModel",
    "IQNValueConfig",
    "IQNValueHead",
    "IQNValueMixin",
    "IQN_VALUE_NAMESPACE",
    "MLPModel",
    "MLPModelConfig",
    "NextTokenAuxHead",
    "StateAuxHead",
    "TransformerModel",
    "TransformerModelConfig",
]
