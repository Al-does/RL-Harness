"""Monte Carlo cross-checks of MESS3 analytic solvers against task envs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from envs.hmm import HMMEnv


SOLVER_DIAGNOSTICS = {
    "state": True,
    "belief": True,
    "tokens": True,
    "rewards": True,
}


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
    """Average KL-control reward of ``policy(obs, info) -> w``."""
    env = HMMEnv(
        {
            "model": {
                "factory": "envs.mess3.model:control_model",
                "kwargs": {"alpha": alpha},
            },
            "task": {
                "class": (
                    "envs.mess3.tasks.occupancy_control:"
                    "OccupancyControlTask"
                ),
                "kwargs": {
                    "transition_kl_beta": beta,
                    "action_limit": w_max,
                },
            },
            "delay": delay,
            "diagnostics": SOLVER_DIAGNOSTICS,
            "seed": seed,
        }
    )
    obs, info = env.reset(seed=seed)
    rewards = np.empty(n_steps)
    occ = np.empty(n_steps)
    cost = np.empty(n_steps)
    for t in range(n_steps):
        w = policy(obs, info)
        obs, r, _, truncated, info = env.step(w)
        rewards[t] = r
        components = info["reward_components"]
        occ[t] = components["occupancy_reward"]
        cost[t] = -components["transition_kl_penalty"]
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
    """Average state-guess accuracy of ``policy(obs, info) -> guess``."""
    env = HMMEnv(
        {
            "model": {
                "factory": "envs.mess3.model:state_guess_model",
                "kwargs": {"alpha": alpha},
            },
            "task": {
                "class": "envs.mess3.tasks.state_guess:StateGuessTask",
            },
            "observation": {"action": None},
            "delay": delay,
            "diagnostics": SOLVER_DIAGNOSTICS,
            "seed": seed,
        }
    )
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
    """Use privileged current-state diagnostics for an oracle cross-check."""
    def pi(obs, info):
        return W_by_state[info["state_current"]]
    return pi


def belief_vi_policy(sol):
    """Nearest-grid-point action from a BeliefVISolution, driven by the exact
    filter belief the env exposes."""
    def pi(obs, info):
        return sol.policy(info["belief_current"])
    return pi


def table_policy(table: np.ndarray, fallback: np.ndarray | None = None):
    """Token-context policy: reactive (3 rows) or constant (1 row).

    Uses the visible token in obs (all-zeros at t=0 with delay=1 -> fallback,
    default = the table mean)."""
    fb = table.mean(axis=0) if fallback is None else fallback

    def pi(obs, info):
        tok = info["visible_token_current"]
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
        tok = info["visible_token_current"]
        if tok is None or prev_tok[0] is None:
            a = fb if tok is None else table[tok * 3 + tok]
        else:
            a = table[tok * 3 + prev_tok[0]]
        prev_tok[0] = tok
        return a
    return pi


def argmax_belief_policy():
    def pi(obs, info):
        return int(np.argmax(info["belief_current"]))
    return pi


def memoryless_guess_policy(mapping):
    def pi(obs, info):
        tok = info["visible_token_current"]
        return int(mapping[tok]) if tok is not None else 0
    return pi
