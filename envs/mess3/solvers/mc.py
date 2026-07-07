"""Monte-Carlo validation of the analytic solvers, run through the ACTUAL
Environment A / B implementations so solver and environment cross-check each
other.  All estimates report a batch-means standard error (100 batches).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from envs.mess3.env_continuous import Mess3ContinuousConfig, Mess3ContinuousEnv
from envs.mess3.env_stateguess import StateGuessConfig, StateGuessEnv


@dataclass
class MCResult:
    mean: float
    se: float
    n_steps: int
    occupancy: float = float("nan")
    control_cost: float = float("nan")


def _batch_se(vals: np.ndarray, n_batches: int = 100) -> float:
    b = vals[: len(vals) - len(vals) % n_batches].reshape(n_batches, -1).mean(axis=1)
    return float(b.std(ddof=1) / np.sqrt(n_batches))


def mc_env_a(
    policy: Callable[[np.ndarray, dict], np.ndarray],
    beta: float,
    w_max: float,
    delay: int,
    n_steps: int = 1_000_000,
    seed: int = 0,
    alpha: float = 0.85,
) -> MCResult:
    """Average reward of ``policy(obs, info) -> w`` on Mess3ContinuousEnv.

    Continuing task: episodes truncate at the env's episode_length and chain
    across resets (each reset re-draws the initial state; with >= 1e6 steps
    the truncation bias is < 1/episode_length of the mixing scale).
    """
    env = Mess3ContinuousEnv(
        Mess3ContinuousConfig(beta=beta, w_max2=w_max, delay=delay, alpha=alpha, seed=seed)
    )
    obs, info = env.reset(seed=seed)
    rewards = np.empty(n_steps)
    occ = np.empty(n_steps)
    cost = np.empty(n_steps)
    for t in range(n_steps):
        w = policy(obs, info)
        obs, r, _, truncated, info = env.step(w)
        rewards[t] = r
        occ[t] = info["reward_occupancy"]
        cost[t] = info["reward_control_cost"]
        if truncated:
            obs, info = env.reset()
    return MCResult(
        mean=float(rewards.mean()),
        se=_batch_se(rewards),
        n_steps=n_steps,
        occupancy=float(occ.mean()),
        control_cost=float(cost.mean()),
    )


def mc_env_b(
    policy: Callable[[np.ndarray, dict], int],
    delay: int,
    n_steps: int = 1_000_000,
    seed: int = 0,
    alpha: float = 0.85,
) -> MCResult:
    """Average accuracy of ``policy(obs, info) -> guess`` on StateGuessEnv."""
    env = StateGuessEnv(StateGuessConfig(delay=delay, alpha=alpha, seed=seed))
    obs, info = env.reset(seed=seed)
    rewards = np.empty(n_steps)
    for t in range(n_steps):
        a = policy(obs, info)
        obs, r, _, truncated, info = env.step(a)
        rewards[t] = r
        if truncated:
            obs, info = env.reset()
    return MCResult(mean=float(rewards.mean()), se=_batch_se(rewards), n_steps=n_steps)


# -- ready-made policies -----------------------------------------------------

def oracle_policy(W_by_state: np.ndarray):
    """Cheats via info['state'] — MC twin of the oracle solver."""
    def pi(obs, info):
        return W_by_state[info["state"]]
    return pi


def belief_vi_policy(sol):
    """Nearest-grid-point action from a BeliefVISolution, driven by the exact
    filter belief the env exposes."""
    def pi(obs, info):
        return sol.policy(info["belief"])
    return pi


def table_policy(table: np.ndarray, fallback: np.ndarray | None = None):
    """Token-context policy: reactive (3 rows) or constant (1 row).

    Uses the visible token in obs (all-zeros at t=0 with delay=1 -> fallback,
    default = the table mean)."""
    fb = table.mean(axis=0) if fallback is None else fallback

    def pi(obs, info):
        tok = info["obs_token"]
        if table.shape[0] == 1:
            return table[0]
        return table[tok] if tok is not None else fb
    return pi


def stack2_policy(table: np.ndarray, fallback: np.ndarray | None = None):
    """Stack-2 policy over (newest, previous) visible tokens; context index
    newest * 3 + previous, matching itertools.product enumeration order."""
    fb = table.mean(axis=0) if fallback is None else fallback
    prev_tok = [None]

    def pi(obs, info):
        tok = info["obs_token"]
        if tok is None or prev_tok[0] is None:
            a = fb if tok is None else table[tok * 3 + tok]
        else:
            a = table[tok * 3 + prev_tok[0]]
        prev_tok[0] = tok
        return a
    return pi


def argmax_belief_policy():
    def pi(obs, info):
        return int(np.argmax(info["belief"]))
    return pi


def memoryless_guess_policy(mapping):
    def pi(obs, info):
        tok = info["obs_token"]
        return int(mapping[tok]) if tok is not None else 0
    return pi
