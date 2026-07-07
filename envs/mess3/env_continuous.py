"""Environment A: MESS3-Continuous.

3-state POMDP with MESS3-style emissions, KL-regularized continuous control
over transition tilts, and (by default) a one-step observation delay.

Step order at time t:
  1. o_t is emitted from s_t (at reset, o_0 is emitted from s_0 pre-action);
  2. the agent acts w_t on an observation built from the DELAYED token
     (o_{t-1} when delay=1, o_t when delay=0) and the previous executed
     action w_{t-1} (zeros at t=0);
  3. s_{t+1} ~ u_{w_t}(. | s_t), where u_w is the exponentially tilted row.

Reward: r_t = 1[s_t == 2] - (1/beta) * KL(u_{w_t}(.|s_t) || P0[s_t]).
Occupancy and control-cost components are logged separately in ``info``.

Observation (delay=1): concat(one-hot(o_{t-1}) [3], w_{t-1} [2]) -> (5,).
At t=0 no token has been revealed to the agent yet (o_{-1} does not exist),
so the token slot is all-zeros and w_{-1} = zeros; o_0 is emitted from s_0
pre-action and is delivered at t=1.  (With delay=0 the token slot is
one-hot(o_t) and is always populated.)

``info`` exposes the exact decision-time belief over s_t (the probe target),
the most recent measurement posterior, the true state, and the token the
agent's observation was built from.

passive_mode: actions are ignored, dynamics are the canonical symmetric
MESS3(x=0.05) chain, emissions alpha=0.85 by default — reproduces
MESS3(0.05, 0.85) for validating the probe pipeline against Shai et al.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from envs.mess3.core import (
    MESS3_PASSIVE_M,
    N_STATES,
    N_TOKENS,
    P0,
    REWARD_STATE,
    emission_matrix,
    kl_cost_per_state,
    tilted_transition,
)
from envs.mess3.filters import ExactFilter

OBS_DIM = N_TOKENS + 2  # one-hot token + previous 2D action


@dataclass
class Mess3ContinuousConfig:
    beta: float = 4.0
    w_max2: float = 5.0
    alpha: float = 0.85  # emission sharpness; do NOT source difficulty from it
    delay: int = 1
    episode_length: int = 1024
    passive_mode: bool = False
    # Observation variants for the Phase-3/4 arms (dynamics/reward unchanged):
    #   "token"  (default): one-hot delayed token + previous action  -> (5,)
    #   "state"  (A-oracle): one-hot TRUE current state              -> (3,)
    #   "belief" (A-beliefobs): exact decision-time filter belief    -> (3,)
    #   "stackK" (A-stack-k): last K visible tokens + last K actions -> (5K,)
    obs_mode: str = "token"
    # N-scramble: the token slot of the agent's observation is replaced by an
    # i.i.d. uniform draw; chain, rewards, info, and the true-token filter run
    # unchanged.
    scramble_tokens: bool = False
    seed: int | None = None

    def __post_init__(self):
        if self.delay not in (0, 1):
            raise ValueError("delay must be 0 or 1")
        if self.obs_mode not in ("token", "state", "belief") and not (
            self.obs_mode.startswith("stack") and self.obs_mode[5:].isdigit()
        ):
            raise ValueError(f"bad obs_mode: {self.obs_mode!r}")

    @property
    def stack_k(self) -> int:
        return int(self.obs_mode[5:]) if self.obs_mode.startswith("stack") else 0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Mess3ContinuousConfig":
        return cls(**(d or {}))


class Mess3ContinuousEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | Mess3ContinuousConfig | None = None):
        if not isinstance(config, Mess3ContinuousConfig):
            config = Mess3ContinuousConfig.from_dict(config)
        self.cfg = config
        self.base = MESS3_PASSIVE_M if config.passive_mode else P0
        self.E = emission_matrix(config.alpha)
        # Initial state uniform (continuing task; episodes truncate only).
        self.init_belief = np.full(N_STATES, 1.0 / N_STATES)

        self.action_space = gym.spaces.Box(
            low=-config.w_max2, high=config.w_max2, shape=(2,), dtype=np.float32
        )
        if config.obs_mode == "token":
            obs_shape = (OBS_DIM,)
        elif config.obs_mode in ("state", "belief"):
            obs_shape = (N_STATES,)
        else:
            obs_shape = (config.stack_k * (N_TOKENS + 2),)
        self.observation_space = gym.spaces.Box(
            low=-config.w_max2, high=config.w_max2, shape=obs_shape, dtype=np.float32
        )

        self._filter = ExactFilter(self.E, config.delay, self.init_belief)
        self._rng = np.random.default_rng(config.seed)
        self._s: int = 0
        self._t: int = 0
        self._pending_token: int = 0  # o_t already emitted from current s_t
        self._obs_token: int | None = None  # true delayed token (info / filter)
        self._vis_token: int | None = None  # what the agent actually sees
        self._prev_action = np.zeros(2, dtype=np.float64)
        self._history: list = []  # newest-first (token, action) frames for stackK

    # -- helpers ------------------------------------------------------------

    def _sample(self, p: np.ndarray) -> int:
        # cumsum + searchsorted is ~10x faster than rng.choice(p=...) per call.
        return int(np.searchsorted(np.cumsum(p), self._rng.random(), side="right"))

    def _emit(self, s: int) -> int:
        return self._sample(self.E[s])

    def _push_frame(self):
        """Record the (agent-visible token, previous action) decision frame.

        Called once per decision point so the scramble draw (if any) is made
        exactly once per step, keeping trajectories deterministic given seed.
        """
        tok = self._obs_token
        if tok is not None and self.cfg.scramble_tokens:
            tok = int(self._rng.integers(N_TOKENS))
        self._vis_token = tok
        if self.cfg.stack_k:
            self._history.insert(0, (tok, self._prev_action.copy()))
            del self._history[self.cfg.stack_k:]

    def _build_obs(self) -> np.ndarray:
        mode = self.cfg.obs_mode
        if mode == "state":
            obs = np.zeros(N_STATES, dtype=np.float32)
            obs[self._s] = 1.0
            return obs
        if mode == "belief":
            return self._filter.decision_belief.astype(np.float32)
        if mode == "token":
            obs = np.zeros(OBS_DIM, dtype=np.float32)
            if self._vis_token is not None:
                obs[self._vis_token] = 1.0
            obs[N_TOKENS:] = self._prev_action
            return obs
        # stackK: newest-first frames of (one-hot token, action); zero-padded.
        k = self.cfg.stack_k
        obs = np.zeros(k * (N_TOKENS + 2), dtype=np.float32)
        for i, (tok, act) in enumerate(self._history):
            base = i * (N_TOKENS + 2)
            if tok is not None:
                obs[base + tok] = 1.0
            obs[base + N_TOKENS: base + N_TOKENS + 2] = act
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

    # -- gym API ------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._s = int(self._rng.integers(N_STATES))  # uniform initial state
        self._prev_action = np.zeros(2, dtype=np.float64)
        self._history = []
        self._pending_token = self._emit(self._s)  # o_0, emitted pre-action
        if self.cfg.delay == 1:
            self._obs_token = None  # o_{-1} does not exist
            self._filter.reset()
        else:
            self._obs_token = self._pending_token
            self._filter.reset(first_token=self._pending_token)
        self._push_frame()
        return self._build_obs(), self._info()

    def step(self, action):
        w = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -self.cfg.w_max2, self.cfg.w_max2)
        if self.cfg.passive_mode:
            U = self.base
            cost = 0.0
        else:
            U = tilted_transition(w, self.base)
            cost = float(kl_cost_per_state(w, self.base)[self._s]) / self.cfg.beta

        occ = float(self._s == REWARD_STATE)
        reward = occ - cost

        emitted = self._pending_token  # o_t, emitted from s_t before the action
        # Transition and emit the next token.
        self._s = self._sample(U[self._s])
        self._t += 1
        self._pending_token = self._emit(self._s)  # o_{t+1}

        if self.cfg.delay == 1:
            self._obs_token = emitted  # agent sees o_t at decision time t+1
            self._filter.step_delay1(emitted, U)
        else:
            self._obs_token = self._pending_token  # agent sees o_{t+1}
            self._filter.step_delay0(U, self._pending_token)
        self._prev_action = w
        self._push_frame()

        truncated = self._t >= self.cfg.episode_length
        info = self._info()
        info["reward_occupancy"] = occ
        info["reward_control_cost"] = cost
        return self._build_obs(), reward, False, truncated, info
