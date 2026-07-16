"""Task for guessing the current hidden MESS3 state."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent


class StateGuessTask:
    """Score a discrete guess against the pre-transition hidden state."""

    requires_belief = False

    def __init__(self, *, model: HMMModel) -> None:
        self.action_space = gym.spaces.Discrete(model.n_states)
        self.action_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(model.n_states,),
            dtype=np.float32,
        )

    def reset(self) -> None:
        pass

    def resolve_action(
        self,
        action: int,
        state: int,
        model: HMMModel,
    ) -> ActionDecision:
        del state
        guess = int(action)
        if not self.action_space.contains(guess):
            raise ValueError(f"guess {guess} is outside the action space")
        return ActionDecision(
            requested_action=guess,
            executed_action=guess,
            transition_matrix=model.transition_matrix,
        )

    def reward(
        self,
        event: TransitionEvent,
        decision: ActionDecision,
    ) -> tuple[float, dict[str, float]]:
        reward = float(decision.executed_action == event.state_before)
        return reward, {
            "state_guess_reward": reward,
            "state_guess_valid": 1.0,
        }

    def encode_action(self, executed_action: int) -> np.ndarray:
        encoded = np.zeros(self.action_space.n, dtype=np.float32)
        encoded[int(executed_action) - self.action_space.start] = 1.0
        return encoded
