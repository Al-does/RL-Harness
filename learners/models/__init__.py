"""Complete RLlib model compositions and their typed configurations."""

from learners.models.mlp import MLPModel, MLPModelConfig
from learners.models.next_token import NextTokenAuxHead
from learners.models.state_aux import StateAuxHead
from learners.models.transformer import TransformerModel, TransformerModelConfig
from learners.models.composed import (
    MLPWithNextTokenAux,
    TransformerWithNextTokenAux,
    TransformerWithStateAux,
)

__all__ = [
    "MLPModel",
    "MLPModelConfig",
    "MLPWithNextTokenAux",
    "NextTokenAuxHead",
    "StateAuxHead",
    "TransformerModel",
    "TransformerModelConfig",
    "TransformerWithNextTokenAux",
    "TransformerWithStateAux",
]
