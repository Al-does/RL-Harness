"""Decoupled policy/advantage and value networks for DAAC and IDAAC."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override

from learners.components.mlp import MLPEncoder


NAMESPACE = "idaac"
ADVANTAGE_PREDICTIONS = f"{NAMESPACE}/advantage_predictions"
PAIRED_OBSERVATIONS = f"{NAMESPACE}/paired_observations"
PAIRED_EMBEDDINGS = f"{NAMESPACE}/paired_embeddings"
PAIR_VALID_MASK = f"{NAMESPACE}/pair_valid_mask"
ORDER_TARGETS = f"{NAMESPACE}/order_targets"
OLD_VALUE_PREDICTIONS = f"{NAMESPACE}/old_value_predictions"


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> nn.Module:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


class ImpalaResidualBlock(nn.Module):
    """Pre-activation residual block used by the paper's IMPALA encoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = F.relu(inputs)
        outputs = self.conv1(outputs)
        outputs = self.conv2(F.relu(outputs))
        return outputs + residual


class ImpalaCNNEncoder(nn.Module):
    """Three-stage IMPALA ResNet used by the original Procgen experiments."""

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        *,
        channels: tuple[int, ...] = (16, 32, 32),
        embedding_dim: int = 256,
        channels_last: bool = True,
        normalize_images: bool = True,
    ):
        super().__init__()
        if len(observation_shape) != 3:
            raise ValueError("ImpalaCNNEncoder requires a three-dimensional image")
        if not channels or any(width <= 0 for width in channels):
            raise ValueError("channels must contain positive widths")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.channels_last = channels_last
        self.normalize_images = normalize_images
        if channels_last:
            height, width, input_channels = observation_shape
        else:
            input_channels, height, width = observation_shape

        stages: list[nn.Module] = []
        width_in = int(input_channels)
        for width_out in channels:
            convolution = nn.Conv2d(width_in, width_out, 3, padding=1)
            nn.init.xavier_uniform_(convolution.weight)
            nn.init.zeros_(convolution.bias)
            stages.append(
                nn.Sequential(
                    convolution,
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    ImpalaResidualBlock(width_out),
                    ImpalaResidualBlock(width_out),
                )
            )
            width_in = width_out
            height = (height + 1) // 2
            width = (width + 1) // 2
        self.stages = nn.Sequential(*stages)
        self.projection = nn.Linear(width_in * height * width, embedding_dim)
        _orthogonal_init(self.projection, nn.init.calculate_gain("relu"))
        self.output_dim = embedding_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        leading_shape = observations.shape[:-3]
        images = observations.reshape(-1, *observations.shape[-3:])
        if self.channels_last:
            images = images.movedim(-1, 1)
        images = images.to(dtype=self.projection.weight.dtype)
        if self.normalize_images:
            images = images / 255.0
        features = self.stages(images)
        features = F.relu(features.flatten(start_dim=1))
        embeddings = F.relu(self.projection(features))
        return embeddings.reshape(*leading_shape, self.output_dim)


class OrderClassifier(nn.Module):
    """Binary temporal-order discriminator with a frozen-parameter forward."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: tuple[int, ...] = (),
    ):
        super().__init__()
        widths = (2 * embedding_dim, *hidden_dims, 1)
        self.layers = nn.ModuleList(
            nn.Linear(input_width, output_width)
            for input_width, output_width in zip(widths, widths[1:])
        )
        for index, layer in enumerate(self.layers):
            gain = nn.init.calculate_gain("relu") if index < len(self.layers) - 1 else 1.0
            _orthogonal_init(layer, gain)

    def _forward(self, inputs: torch.Tensor, *, detach_parameters: bool) -> torch.Tensor:
        outputs = inputs
        for index, layer in enumerate(self.layers):
            weight = layer.weight.detach() if detach_parameters else layer.weight
            bias = (
                layer.bias.detach()
                if detach_parameters and layer.bias is not None
                else layer.bias
            )
            outputs = F.linear(outputs, weight, bias)
            if index < len(self.layers) - 1:
                outputs = F.relu(outputs)
        return outputs.squeeze(-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._forward(inputs, detach_parameters=False)

    def forward_with_frozen_parameters(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._forward(inputs, detach_parameters=True)


@dataclass(frozen=True)
class IDAACModelConfig:
    """Validated architecture choices for the decoupled RLModule."""

    encoder_type: str = "auto"
    hidden_dims: tuple[int, ...] = (128, 128)
    impala_channels: tuple[int, ...] = (16, 32, 32)
    embedding_dim: int = 256
    channels_last: bool = True
    normalize_images: bool = True
    order_hidden_dims: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        object.__setattr__(self, "impala_channels", tuple(self.impala_channels))
        object.__setattr__(self, "order_hidden_dims", tuple(self.order_hidden_dims))
        if self.encoder_type not in {"auto", "mlp", "impala_cnn"}:
            raise ValueError("encoder_type must be 'auto', 'mlp', or 'impala_cnn'")
        for name in ("hidden_dims", "impala_channels", "order_hidden_dims"):
            if any(width <= 0 for width in getattr(self, name)):
                raise ValueError(f"{name} must contain positive widths")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

    @classmethod
    def from_dict(cls, values: dict) -> "IDAACModelConfig":
        own_fields = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in own_fields})

    def to_dict(self) -> dict:
        return asdict(self)


class IDAACModel(TorchRLModule, ValueFunctionAPI):
    """Paper architecture with independent policy and value representations.

    The policy encoder feeds the action and action-conditioned advantage heads.
    A second encoder feeds only the value head. For image observations, the
    default is the paper's IMPALA ResNet; vector observations use an MLP.
    """

    @override(TorchRLModule)
    def setup(self) -> None:
        self.config = IDAACModelConfig.from_dict(dict(self.model_config))
        observation_shape = tuple(int(size) for size in self.observation_space.shape)
        encoder_type = self.config.encoder_type
        if encoder_type == "auto":
            encoder_type = "mlp" if len(observation_shape) == 1 else "impala_cnn"
        self.policy_encoder = self._make_encoder(encoder_type, observation_shape)
        self.value_encoder = self._make_encoder(encoder_type, observation_shape)
        embedding_dim = self.policy_encoder.output_dim
        if self.value_encoder.output_dim != embedding_dim:
            raise ValueError("policy and value encoders must have equal output widths")
        self._embedding_dim = embedding_dim

        self._discrete_actions = isinstance(self.action_space, gym.spaces.Discrete)
        if self._discrete_actions:
            self._action_dim = int(self.action_space.n)
            distribution_width = self._action_dim
            self.policy_log_std = None
        elif isinstance(self.action_space, gym.spaces.Box):
            self._action_dim = int(np.prod(self.action_space.shape))
            distribution_width = self._action_dim
            self.policy_log_std = nn.Parameter(torch.zeros(self._action_dim))
        else:
            raise ValueError(f"unsupported action space {self.action_space}")

        self.policy_head = nn.Linear(embedding_dim, distribution_width)
        self.advantage_head = nn.Linear(embedding_dim + self._action_dim, 1)
        self.value_head = nn.Linear(embedding_dim, 1)
        self.order_classifier = OrderClassifier(
            embedding_dim,
            self.config.order_hidden_dims,
        )
        _orthogonal_init(self.policy_head, 0.01)
        _orthogonal_init(self.advantage_head)
        _orthogonal_init(self.value_head)

    def _make_encoder(
        self,
        encoder_type: str,
        observation_shape: tuple[int, ...],
    ) -> nn.Module:
        if encoder_type == "mlp":
            if len(observation_shape) != 1:
                raise ValueError("MLP IDAAC encoder requires vector observations")
            return MLPEncoder(observation_shape[0], self.config.hidden_dims)
        if encoder_type == "impala_cnn":
            return ImpalaCNNEncoder(
                observation_shape,
                channels=self.config.impala_channels,
                embedding_dim=self.config.embedding_dim,
                channels_last=self.config.channels_last,
                normalize_images=self.config.normalize_images,
            )
        raise AssertionError(f"unhandled encoder type {encoder_type!r}")

    def _distribution_inputs(self, embeddings: torch.Tensor) -> torch.Tensor:
        means_or_logits = self.policy_head(embeddings)
        if self._discrete_actions:
            return means_or_logits
        return torch.cat(
            [means_or_logits, self.policy_log_std.expand_as(means_or_logits)],
            dim=-1,
        )

    def _action_features(
        self,
        actions: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self._discrete_actions:
            return F.one_hot(
                actions.to(dtype=torch.long),
                num_classes=self._action_dim,
            ).to(dtype=dtype)
        return actions.reshape(*actions.shape[:-1], self._action_dim).to(dtype=dtype)

    def predict_advantages(
        self,
        embeddings: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        action_features = self._action_features(actions, dtype=embeddings.dtype)
        return self.advantage_head(
            torch.cat([embeddings, action_features], dim=-1)
        ).squeeze(-1)

    def order_logits(
        self,
        first_embeddings: torch.Tensor,
        second_embeddings: torch.Tensor,
        *,
        detach_embeddings: bool = False,
        detach_classifier: bool = False,
    ) -> torch.Tensor:
        inputs = torch.cat([first_embeddings, second_embeddings], dim=-1)
        if detach_embeddings:
            inputs = inputs.detach()
        if detach_classifier:
            return self.order_classifier.forward_with_frozen_parameters(inputs)
        return self.order_classifier(inputs)

    def _policy_outputs(self, batch: Dict[str, Any], *, training: bool) -> dict:
        embeddings = self.policy_encoder(batch[Columns.OBS])
        outputs = {
            Columns.ACTION_DIST_INPUTS: self._distribution_inputs(embeddings),
        }
        if training:
            outputs[Columns.EMBEDDINGS] = embeddings
            if Columns.ACTIONS in batch:
                outputs[ADVANTAGE_PREDICTIONS] = self.predict_advantages(
                    embeddings,
                    batch[Columns.ACTIONS],
                )
            if PAIRED_OBSERVATIONS in batch:
                outputs[PAIRED_EMBEDDINGS] = self.policy_encoder(
                    batch[PAIRED_OBSERVATIONS]
                )
        return outputs

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        return self._policy_outputs(batch, training=False)

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        return self._policy_outputs(batch, training=True)

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> torch.Tensor:
        del embeddings
        value_embeddings = self.value_encoder(batch[Columns.OBS])
        return self.value_head(value_embeddings).squeeze(-1)

    @torch.no_grad()
    def encode_step(
        self,
        observation: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        return self.policy_encoder(observation), state


__all__ = [
    "ADVANTAGE_PREDICTIONS",
    "IDAACModel",
    "IDAACModelConfig",
    "ImpalaCNNEncoder",
    "OLD_VALUE_PREDICTIONS",
    "ORDER_TARGETS",
    "PAIRED_EMBEDDINGS",
    "PAIRED_OBSERVATIONS",
    "PAIR_VALID_MASK",
]
