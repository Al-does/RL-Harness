"""Complete RLlib model compositions and their typed configurations."""

from learners.models.idaac import IDAACModel, IDAACModelConfig
from learners.models.gru import GRUModel, GRUModelConfig
from learners.models.iqn_value import IQNValueMixin
from learners.models.mlp import MLPModel, MLPModelConfig
from learners.models.next_token import NextTokenAuxHead
from learners.models.ppg import (
    PPGAuxiliaryValueHead,
    PPGMLPModel,
    PPGTransformerModel,
)
from learners.models.qr_value import QRValueMixin
from learners.models.residual_mlp import ResidualMLPModel, ResidualMLPModelConfig
from learners.models.state_aux import StateAuxHead
from learners.models.transformer import TransformerModel, TransformerModelConfig

__all__ = [
    "GRUModel",
    "GRUModelConfig",
    "ResidualMLPModel",
    "ResidualMLPModelConfig",
    "IDAACModel",
    "IDAACModelConfig",
    "IQNValueMixin",
    "MLPModel",
    "MLPModelConfig",
    "NextTokenAuxHead",
    "PPGAuxiliaryValueHead",
    "PPGMLPModel",
    "PPGTransformerModel",
    "QRValueMixin",
    "StateAuxHead",
    "TransformerModel",
    "TransformerModelConfig",
]
