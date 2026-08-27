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
from learners.components.transformer import CausalTransformerEncoder


NAMESPACE = "idaac"
ADVANTAGE_PREDICTIONS = f"{NAMESPACE}/advantage_predictions"
PAIRED_OBSERVATIONS = f"{NAMESPACE}/paired_observations"
PAIRED_EMBEDDINGS = f"{NAMESPACE}/paired_embeddings"
PAIR_VALID_MASK = f"{NAMESPACE}/pair_valid_mask"
PAIR_POSITIONS = f"{NAMESPACE}/pair_positions"
ORDER_TARGETS = f"{NAMESPACE}/order_targets"
OLD_VALUE_PREDICTIONS = f"{NAMESPACE}/old_value_predictions"


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> nn.Module:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


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

    encoder_type: str = "transformer"
    hidden_dims: tuple[int, ...] = (128, 128)
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 1
    context_len: int = 10
    max_seq_len: int = 32
    order_hidden_dims: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        object.__setattr__(self, "order_hidden_dims", tuple(self.order_hidden_dims))
        if self.encoder_type not in {"mlp", "transformer"}:
            raise ValueError("encoder_type must be 'mlp' or 'transformer'")
        for name in ("hidden_dims", "order_hidden_dims"):
            if any(width <= 0 for width in getattr(self, name)):
                raise ValueError(f"{name} must contain positive widths")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        if min(
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.context_len,
            self.max_seq_len,
        ) <= 0:
            raise ValueError("transformer dimensions and lengths must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("transformer head dimension must be even for RoPE")

    @classmethod
    def from_dict(cls, values: dict) -> "IDAACModelConfig":
        own_fields = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in own_fields})

    def to_dict(self) -> dict:
        return asdict(self)


class IDAACModel(TorchRLModule, ValueFunctionAPI):
    """IDAAC with independent policy and value representations.

    The policy encoder feeds the action and action-conditioned advantage heads.
    A second encoder feeds only the value head. The default encoder is a small
    causal transformer; a memoryless MLP remains available for simple tasks.
    """

    @override(TorchRLModule)
    def setup(self) -> None:
        self.config = IDAACModelConfig.from_dict(dict(self.model_config))
        observation_shape = tuple(int(size) for size in self.observation_space.shape)
        encoder_type = self.config.encoder_type
        if len(observation_shape) != 1:
            raise ValueError("IDAAC encoders currently require vector observations")
        self._encoder_type = encoder_type
        self._obs_dim = observation_shape[0]
        self.policy_encoder = self._make_encoder(encoder_type, observation_shape)
        self.value_encoder = self._make_encoder(encoder_type, observation_shape)
        embedding_dim = (
            self.config.d_model
            if encoder_type == "transformer"
            else self.policy_encoder.output_dim
        )
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
        if encoder_type == "transformer":
            return CausalTransformerEncoder(
                obs_dim=observation_shape[0],
                d_model=self.config.d_model,
                n_layers=self.config.n_layers,
                n_heads=self.config.n_heads,
                context_len=self.config.context_len,
            )
        raise AssertionError(f"unhandled encoder type {encoder_type!r}")

    @property
    def sequence_lookback(self) -> int:
        if self._encoder_type != "transformer":
            return 0
        return self.policy_encoder.lookback

    @override(TorchRLModule)
    def get_initial_state(self) -> dict[str, np.ndarray]:
        if self._encoder_type != "transformer":
            return {}
        encoder = self.policy_encoder
        cache_shape = (
            encoder.n_layers,
            encoder.n_heads,
            encoder.cache_len,
            encoder.head_dim,
        )
        state: dict[str, np.ndarray] = {}
        for tower in ("policy", "value"):
            state[f"{tower}_ctx"] = np.zeros(
                (encoder.lookback, self._obs_dim),
                dtype=np.float32,
            )
            state[f"{tower}_len"] = np.zeros((1,), dtype=np.float32)
            state[f"{tower}_kv_k"] = np.zeros(cache_shape, dtype=np.float32)
            state[f"{tower}_kv_v"] = np.zeros(cache_shape, dtype=np.float32)
            state[f"{tower}_kv_len"] = np.zeros((1,), dtype=np.float32)
        return state

    def _advance_context(
        self,
        observations: torch.Tensor,
        state: dict[str, torch.Tensor],
        tower: str,
    ) -> dict[str, torch.Tensor]:
        context = state[f"{tower}_ctx"]
        sequence = torch.cat([context, observations], dim=1)
        lengths = state[f"{tower}_len"].reshape(-1) + observations.shape[1]
        return {
            f"{tower}_ctx": sequence[:, -self.sequence_lookback :, :],
            f"{tower}_len": lengths.clamp(
                max=float(self.sequence_lookback)
            ).reshape(-1, 1),
        }

    def _encode_transformer_train(
        self,
        encoder: CausalTransformerEncoder,
        batch: Dict[str, Any],
        tower: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        observations = batch[Columns.OBS]
        state = batch[Columns.STATE_IN]
        embeddings = encoder(
            state[f"{tower}_ctx"],
            state[f"{tower}_len"].reshape(-1),
            observations,
        )
        state_out = self._advance_context(observations, state, tower)
        for suffix in ("kv_k", "kv_v", "kv_len"):
            key = f"{tower}_{suffix}"
            state_out[key] = state[key]
        return embeddings, state_out

    def _encode_transformer_rollout(
        self,
        encoder: CausalTransformerEncoder,
        batch: Dict[str, Any],
        tower: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        observations = batch[Columns.OBS]
        state = batch[Columns.STATE_IN]
        embeddings, kv_k, kv_v, kv_len = encoder.forward_cached(
            state[f"{tower}_kv_k"],
            state[f"{tower}_kv_v"],
            state[f"{tower}_kv_len"].reshape(-1),
            observations,
        )
        state_out = self._advance_context(observations, state, tower)
        state_out.update(
            {
                f"{tower}_kv_k": kv_k,
                f"{tower}_kv_v": kv_v,
                f"{tower}_kv_len": kv_len.reshape(-1, 1),
            }
        )
        return embeddings, state_out

    def _advance_transformer_state_only(
        self,
        batch: Dict[str, Any],
        tower: str,
    ) -> dict[str, torch.Tensor]:
        state = batch[Columns.STATE_IN]
        state_out = self._advance_context(batch[Columns.OBS], state, tower)
        for suffix in ("kv_k", "kv_v", "kv_len"):
            key = f"{tower}_{suffix}"
            state_out[key] = state[key]
        return state_out

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
        state_out = None
        if self._encoder_type == "transformer":
            encode = (
                self._encode_transformer_train
                if training
                else self._encode_transformer_rollout
            )
            embeddings, state_out = encode(
                self.policy_encoder,
                batch,
                "policy",
            )
            if training:
                state_out.update(
                    self._advance_transformer_state_only(batch, "value")
                )
            else:
                _, value_state_out = self._encode_transformer_rollout(
                    self.value_encoder,
                    batch,
                    "value",
                )
                state_out.update(value_state_out)
        else:
            embeddings = self.policy_encoder(batch[Columns.OBS])
        outputs = {
            Columns.ACTION_DIST_INPUTS: self._distribution_inputs(embeddings),
        }
        if state_out is not None:
            outputs[Columns.STATE_OUT] = state_out
        if training:
            outputs[Columns.EMBEDDINGS] = embeddings
            if Columns.ACTIONS in batch:
                outputs[ADVANTAGE_PREDICTIONS] = self.predict_advantages(
                    embeddings,
                    batch[Columns.ACTIONS],
                )
            if (
                self._encoder_type == "transformer"
                and PAIR_POSITIONS in batch
            ):
                positions = batch[PAIR_POSITIONS].to(dtype=torch.long)
                gather_indices = positions.unsqueeze(-1).expand(
                    *positions.shape,
                    embeddings.shape[-1],
                )
                outputs[PAIRED_EMBEDDINGS] = torch.gather(
                    embeddings,
                    dim=1,
                    index=gather_indices,
                )
            elif PAIRED_OBSERVATIONS in batch:
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
        if self._encoder_type == "transformer":
            value_embeddings, _ = self._encode_transformer_train(
                self.value_encoder,
                batch,
                "value",
            )
        else:
            value_embeddings = self.value_encoder(batch[Columns.OBS])
        return self.value_head(value_embeddings).squeeze(-1)

    @torch.no_grad()
    def encode_step(
        self,
        observation: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        if self._encoder_type == "transformer":
            if state is None:
                raise ValueError("transformer IDAAC encoding requires recurrent state")
            batch = {
                Columns.OBS: observation.unsqueeze(1),
                Columns.STATE_IN: state,
            }
            policy_embeddings, state_out = self._encode_transformer_rollout(
                self.policy_encoder,
                batch,
                "policy",
            )
            _, value_state_out = self._encode_transformer_rollout(
                self.value_encoder,
                batch,
                "value",
            )
            state_out.update(value_state_out)
            return policy_embeddings[:, 0], state_out
        return self.policy_encoder(observation), state


__all__ = [
    "ADVANTAGE_PREDICTIONS",
    "IDAACModel",
    "IDAACModelConfig",
    "OLD_VALUE_PREDICTIONS",
    "ORDER_TARGETS",
    "PAIRED_EMBEDDINGS",
    "PAIRED_OBSERVATIONS",
    "PAIR_VALID_MASK",
    "PAIR_POSITIONS",
]
