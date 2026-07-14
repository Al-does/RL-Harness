"""Importable RLModule leaf classes assembled from encoder and head mixins."""

from learners.models.mlp import MLPModel
from learners.models.next_token import NextTokenAuxHead
from learners.models.state_aux import StateAuxHead
from learners.models.transformer import TransformerModel


class MLPWithNextTokenAux(NextTokenAuxHead, MLPModel):
    pass


class TransformerWithNextTokenAux(NextTokenAuxHead, TransformerModel):
    pass


class TransformerWithStateAux(StateAuxHead, TransformerModel):
    pass
