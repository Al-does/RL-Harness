"""Environment B: State-Guess (RL-tax measurement).

Same hidden chain as Environment A (P0, FIXED — actions never affect the
dynamics), same emission channel (alpha=0.85), same delay convention.
The agent guesses the current hidden state: Discrete(3) action a_t, reward
r_t = 1[a_t == s_t].  A pure state-estimation problem in RL clothing; its
ceiling-vs-attained gaps calibrate Environment A's.

Observation at t is one-hot(o_{t-delay}); with delay=1 the t=0 observation is
all-zeros (no token revealed yet).  Initial state is drawn from the P0
stationary distribution (0.45, 0.45, 0.10), and the filter is initialized
from it.  ``info`` carries the exact decision-time belief over s_t (the prior
b_t when delay=1, the posterior p_t when delay=0), exactly as in Environment A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from envs.mess3.core import (
    N_STATES,
    N_TOKENS,
    P0,
    emission_matrix,
    stationary_distribution,
)
from envs.mess3.filters import ExactFilter


@dataclass
class StateGuessConfig:
    alpha: float = 0.85
    delay: int = 1
    episode_length: int = 1024
    seed: int | None = None

    def __post_init__(self):
        if self.delay not in (0, 1):
            raise ValueError("delay must be 0 or 1")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "StateGuessConfig":
        return cls(**(d or {}))


class StateGuessEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | StateGuessConfig | None = None):
        if not isinstance(config, StateGuessConfig):
            config = StateGuessConfig.from_dict(config)
        self.cfg = config
        self.E = emission_matrix(config.alpha)
        self.init_belief = stationary_distribution(P0)

        self.action_space = gym.spaces.Discrete(N_STATES)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(N_TOKENS,), dtype=np.float32)

        self._filter = ExactFilter(self.E, config.delay, self.init_belief)
        self._rng = np.random.default_rng(config.seed)
        self._s: int = 0
        self._t: int = 0
        self._pending_token: int = 0
        self._obs_token: int | None = None

    def _sample(self, p: np.ndarray) -> int:
        return int(np.searchsorted(np.cumsum(p), self._rng.random(), side="right"))

    def _emit(self, s: int) -> int:
        return self._sample(self.E[s])

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(N_TOKENS, dtype=np.float32)
        if self._obs_token is not None:
            obs[self._obs_token] = 1.0
        return obs

    def _info(self) -> dict[str, Any]:
        return {
            "belief": self._filter.decision_belief.copy(),
            "posterior_prev": (
                None
                if self._filter.prev_posterior is None
                else self._filter.prev_posterior.copy()
            ),
            "state": self._s,
            "obs_token": self._obs_token,
            "emitted_token": self._pending_token,
        }

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._s = self._sample(self.init_belief)
        self._pending_token = self._emit(self._s)
        if self.cfg.delay == 1:
            self._obs_token = None
            self._filter.reset()
        else:
            self._obs_token = self._pending_token
            self._filter.reset(first_token=self._pending_token)
        return self._build_obs(), self._info()

    def step(self, action):
        reward = float(int(action) == self._s)

        emitted = self._pending_token
        self._s = self._sample(P0[self._s])
        self._t += 1
        self._pending_token = self._emit(self._s)

        if self.cfg.delay == 1:
            self._obs_token = emitted
            self._filter.step_delay1(emitted, P0)
        else:
            self._obs_token = self._pending_token
            self._filter.step_delay0(P0, self._pending_token)

        truncated = self._t >= self.cfg.episode_length
        return self._build_obs(), reward, False, truncated, self._info()
