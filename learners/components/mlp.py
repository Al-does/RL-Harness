"""Reusable feed-forward encoder components."""

from __future__ import annotations

from torch import nn


class MLPEncoder(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        *,
        activation: type[nn.Module] = nn.Tanh,
    ):
        layers: list[nn.Module] = []
        width = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(width, hidden_dim), activation()))
            width = hidden_dim
        super().__init__(*layers)
        self.output_dim = width
