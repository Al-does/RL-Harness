"""Task for predicting a hidden MESS3 state several transitions ahead."""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent


class FutureStateGuessTask:
    """Queue guesses and score each when its future target becomes current.

    Predictions that have not matured when an episode truncates are discarded.
    """

    requires_belief = False

    def __init__(self, *, model: HMMModel, horizon: int = 1) -> None:
        if horizon <= 0:
            raise ValueError("future-state horizon must be positive")
        self.horizon = int(horizon)
        self.action_space = gym.spaces.Discrete(model.n_states)
        self.action_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(model.n_states,),
            dtype=np.float32,
        )
        self._pending: deque[int] = deque()

    @property
    def pending_predictions(self) -> int:
        return len(self._pending)

    def reset(self) -> None:
        self._pending.clear()

    def on_truncation(self) -> None:
        self._pending.clear()

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
        self._pending.append(int(decision.executed_action))
        if len(self._pending) < self.horizon:
            return 0.0, {
                "state_guess_reward": 0.0,
                "state_guess_valid": 0.0,
                "pending_predictions": float(len(self._pending)),
            }

        matured_guess = self._pending.popleft()
        reward = float(matured_guess == event.state_after)
        return reward, {
            "state_guess_reward": reward,
            "state_guess_valid": 1.0,
            "pending_predictions": float(len(self._pending)),
        }

    def encode_action(self, executed_action: int) -> np.ndarray:
        encoded = np.zeros(self.action_space.n, dtype=np.float32)
        encoded[int(executed_action) - self.action_space.start] = 1.0
        return encoded
