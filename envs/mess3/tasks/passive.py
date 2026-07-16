"""Fixed-dynamics continuous-action task for passive MESS3 studies."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent


class PassiveTask:
    """Ignore bounded continuous control and reward state occupancy."""

    requires_belief = False

    def __init__(
        self,
        *,
        model: HMMModel,
        action_limit: float = 5.0,
        occupancy_states: Sequence[int] = (2,),
    ) -> None:
        if not np.isfinite(action_limit) or action_limit <= 0.0:
            raise ValueError("action_limit must be finite and positive")
        states = tuple(int(state) for state in occupancy_states)
        if not states:
            raise ValueError("occupancy_states must not be empty")
        if any(not 0 <= state < model.n_states for state in states):
            raise ValueError("occupancy state is outside the model state space")

        self.transition_matrix = model.transition_matrix
        self.occupancy_states = frozenset(states)
        self.action_limit = float(action_limit)
        self.action_space = gym.spaces.Box(
            low=-self.action_limit,
            high=self.action_limit,
            shape=(2,),
            dtype=np.float32,
        )
        self.action_observation_space = self.action_space

    def reset(self) -> None:
        pass

    def resolve_action(
        self,
        action: np.ndarray,
        state: int,
        model: HMMModel,
    ) -> ActionDecision:
        del state, model
        requested = np.asarray(action, dtype=np.float64)
        if requested.shape != (2,):
            raise ValueError("passive-task action must have shape (2,)")
        executed = np.clip(
            requested,
            -self.action_limit,
            self.action_limit,
        )
        return ActionDecision(
            requested_action=requested,
            executed_action=executed,
            transition_matrix=self.transition_matrix,
        )

    def reward(
        self,
        event: TransitionEvent,
        decision: ActionDecision,
    ) -> tuple[float, dict[str, float]]:
        del decision
        occupancy = float(event.state_before in self.occupancy_states)
        return occupancy, {"occupancy_reward": occupancy}

    def encode_action(self, executed_action: np.ndarray) -> np.ndarray:
        return np.asarray(executed_action, dtype=np.float32)
