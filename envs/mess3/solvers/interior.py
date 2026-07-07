"""Interior-action share of the belief-VI optimum under its own stationary
closed-loop distribution (Phase-1 saturation analysis).

For each visited step we take the CONTINUOUS optimal action at the nearest
belief-grid point (polished lazily with L-BFGS-B, cached per grid point) and
record whether each coordinate sits off the box boundary, plus the covariates
the spec asks histograms over: time since ejection from state 2, and belief
entropy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from envs.mess3.core import P0, emission_matrix
from envs.mess3.env_continuous import Mess3ContinuousConfig, Mess3ContinuousEnv
from envs.mess3.solvers.belief_vi import BeliefVISolution, _BellmanTensors, _q_continuous
from envs.mess3.solvers.simplex_grid import nearest_index


@dataclass
class InteriorStats:
    beta: float
    w_max: float
    n_steps: int
    share_w0: float          # fraction of steps with |w0| off the boundary
    share_w1: float
    share_joint: float       # both coordinates interior
    share_any: float         # at least one coordinate interior (hedging signal)
    mean_reward: float
    mean_reward_se: float
    # Histogram payloads (counts over all steps and over interior-joint steps).
    tse_edges: np.ndarray = field(default=None)
    tse_all: np.ndarray = field(default=None)
    tse_interior: np.ndarray = field(default=None)
    ent_edges: np.ndarray = field(default=None)
    ent_all: np.ndarray = field(default=None)
    ent_interior: np.ndarray = field(default=None)
    # Subsampled closed-loop trajectory (for attractor / policy-map figures).
    beliefs_sample: np.ndarray = field(default=None)   # (m, 3)
    actions_sample: np.ndarray = field(default=None)   # (m, 2)


class LazyPolishedPolicy:
    """Nearest-grid VI policy with per-grid-point L-BFGS-B polish, cached."""

    def __init__(self, sol: BeliefVISolution, alpha: float = 0.85, base=P0, n_starts: int = 3):
        self.sol = sol
        self.E = emission_matrix(alpha)
        self.base = base
        self.cache: dict[int, np.ndarray] = {}
        T = _BellmanTensors(sol.grid, sol.actions, self.E, sol.beta, sol.delay, base, sol.n_grid)
        self.Q = T.R + T.expected_next_value(sol.V)
        self.top = np.argsort(self.Q, axis=1)[:, -n_starts:]
        self.bounds = [(-sol.w_max, sol.w_max)] * 2

    def action_at_grid(self, g: int) -> np.ndarray:
        if g not in self.cache:
            b = self.sol.grid[g]
            best_w, best_q = None, -np.inf
            for k in self.top[g]:
                res = minimize(
                    lambda w: -_q_continuous(
                        w, b, self.sol.V, self.E, self.sol.beta, self.sol.delay,
                        self.base, self.sol.n_grid,
                    ),
                    self.sol.actions[k], method="L-BFGS-B", bounds=self.bounds,
                )
                if -res.fun > best_q:
                    best_q, best_w = -res.fun, res.x
            self.cache[g] = best_w
        return self.cache[g]

    def __call__(self, belief: np.ndarray) -> np.ndarray:
        return self.action_at_grid(int(nearest_index(belief, self.sol.n_grid)))


def interior_share(
    sol: BeliefVISolution,
    n_steps: int = 300_000,
    seed: int = 0,
    alpha: float = 0.85,
    boundary_tol_frac: float = 1e-3,
) -> InteriorStats:
    policy = LazyPolishedPolicy(sol, alpha=alpha)
    env = Mess3ContinuousEnv(Mess3ContinuousConfig(
        beta=sol.beta, w_max2=sol.w_max, delay=sol.delay, alpha=alpha, seed=seed,
    ))
    _, info = env.reset(seed=seed)

    tol = boundary_tol_frac * sol.w_max
    W = np.empty((n_steps, 2))
    B = np.empty((n_steps, 3))
    ent = np.empty(n_steps)
    tse = np.full(n_steps, -1, dtype=np.int64)  # -1 = state 2 not yet visited
    rewards = np.empty(n_steps)
    last2 = None
    for t in range(n_steps):
        b = info["belief"]
        w = policy(b)
        W[t] = w
        B[t] = b
        ent[t] = -np.sum(b * np.log(np.maximum(b, 1e-300)))
        if info["state"] == 2:
            last2 = t
        if last2 is not None:
            tse[t] = t - last2
        _, r, _, truncated, info = env.step(w)
        rewards[t] = r
        if truncated:
            _, info = env.reset()

    interior = np.abs(np.abs(W) - sol.w_max) > tol   # (n, 2) True = off boundary
    joint = interior.all(axis=1)
    anyint = interior.any(axis=1)
    valid = tse >= 0

    n_b = 100
    batch = rewards[: n_steps - n_steps % n_b].reshape(n_b, -1).mean(axis=1)
    tse_edges = np.arange(0, 31)
    ent_edges = np.linspace(0.0, np.log(3), 25)
    return InteriorStats(
        beta=sol.beta,
        w_max=sol.w_max,
        n_steps=n_steps,
        share_w0=float(interior[:, 0].mean()),
        share_w1=float(interior[:, 1].mean()),
        share_joint=float(joint.mean()),
        share_any=float(anyint.mean()),
        mean_reward=float(rewards.mean()),
        mean_reward_se=float(batch.std(ddof=1) / np.sqrt(n_b)),
        tse_edges=tse_edges,
        tse_all=np.histogram(np.minimum(tse[valid], 30), bins=tse_edges)[0],
        tse_interior=np.histogram(np.minimum(tse[valid & anyint], 30), bins=tse_edges)[0],
        ent_edges=ent_edges,
        ent_all=np.histogram(ent, bins=ent_edges)[0],
        ent_interior=np.histogram(ent[anyint], bins=ent_edges)[0],
        beliefs_sample=B[:: max(1, n_steps // 20_000)],
        actions_sample=W[:: max(1, n_steps // 20_000)],
    )
