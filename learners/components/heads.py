"""Algorithm-facing policy and value heads shared by RLlib models."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch import nn


class ActorCriticHeads(nn.Module):
    def __init__(self, embedding_dim: int, action_space: gym.Space):
        super().__init__()
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        if self.is_discrete:
            self.policy = nn.Linear(embedding_dim, int(action_space.n))
            self.log_std = None
        elif isinstance(action_space, gym.spaces.Box):
            action_dim = int(np.prod(action_space.shape))
            self.policy = nn.Linear(embedding_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            raise ValueError(f"unsupported action space {action_space}")
        self.value = nn.Linear(embedding_dim, 1)

    def action_distribution_inputs(self, embeddings: torch.Tensor) -> torch.Tensor:
        policy_output = self.policy(embeddings)
        if self.is_discrete:
            return policy_output
        return torch.cat(
            [policy_output, self.log_std.expand_as(policy_output)], dim=-1
        )

    def values(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.value(embeddings).squeeze(-1)

    def policy_mean_and_std(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_discrete:
            raise TypeError("categorical policies do not have a mean and std")
        mean = self.policy(embeddings)
        return mean, self.log_std.exp().expand_as(mean)
