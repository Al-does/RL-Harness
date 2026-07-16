"""Analytic ceilings for Environment B (State-Guess).

- random: 1/3.
- memoryless: brute force over all 27 maps token -> guess against the exact
  stationary joint P(s_t, o_available).  With delay=1 the available token is
  o_{t-1} (emitted from s_{t-1}), so the joint routes through one transition
  step; with delay=0 it is o_t (emitted from s_t) and the ceiling equals
  alpha = 0.85 exactly for the sharp symmetric channel.
- filter ceiling: expected accuracy of argmax of the decision-time belief,
  via a long exact-filter simulation (>= 1e6 steps, SE reported).  We use the
  Rao-Blackwellized estimator E_t[max_s b_t(s)]: given the information the
  belief conditions on, argmax-of-belief is correct with probability
  max_s b_t(s) exactly, so averaging max b over the simulated belief
  trajectory estimates the same quantity as raw guess-vs-state accuracy but
  with far lower variance.  (mc.py cross-checks with the raw estimator.)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from envs.hmm import measure, predict, stationary_distribution
from envs.mess3.model import (
    CONTROL_TRANSITION_MATRIX,
    N_STATES,
    N_TOKENS,
    emission_matrix,
)


@dataclass
class StateGuessTable:
    delay: int
    random: float
    memoryless: float
    memoryless_map: tuple
    filter_ceiling: float
    filter_ceiling_se: float  # 0.0 for the exact enumeration path


def joint_state_token(delay: int, alpha: float = 0.85) -> np.ndarray:
    """J[s, o] = stationary P(s_t = s, available token = o)."""
    E = emission_matrix(alpha)
    pi = stationary_distribution(CONTROL_TRANSITION_MATRIX)
    if delay == 0:
        return pi[:, None] * E                     # o_t emitted from s_t
    # delay=1: token is o_{t-1} from s_{t-1}; s_t follows one
    # CONTROL_TRANSITION_MATRIX step.
    #   J[s, o] = sum_{s'} pi[s'] E[s', o] CONTROL_TRANSITION_MATRIX[s', s]
    return np.einsum(
        "a,ao,as->so",
        pi,
        E,
        CONTROL_TRANSITION_MATRIX,
    )


def best_memoryless(delay: int, alpha: float = 0.85):
    J = joint_state_token(delay, alpha)
    best_v, best_map = -1.0, None
    for m in product(range(N_STATES), repeat=N_TOKENS):
        v = sum(J[m[o], o] for o in range(N_TOKENS))
        if v > best_v:
            best_v, best_map = v, m
    return float(best_v), best_map


def filter_ceiling_sim(
    delay: int,
    alpha: float = 0.85,
    n_steps: int = 2_000_000,
    burn_in: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """(E[max_s b_t(s)], SE) along a long exact-filter belief trajectory.

    Simulates the token stream (belief evolution only needs tokens, which we
    sample by first sampling the hidden path), runs the exact filter, and
    averages max_s b_t(s).  SE via batch means (100 batches) to respect the
    trajectory's autocorrelation.
    """
    rng = np.random.default_rng(seed)
    E = emission_matrix(alpha)
    pi = stationary_distribution(CONTROL_TRANSITION_MATRIX)

    # Sample the hidden path and tokens up front (vectorized inverse-CDF).
    total = burn_in + n_steps
    s = int(rng.choice(N_STATES, p=pi))
    states = np.empty(total + 1, dtype=np.int64)
    cdf_P = CONTROL_TRANSITION_MATRIX.cumsum(axis=1)
    cdf_E = E.cumsum(axis=1)
    u_s = rng.random(total + 1)
    u_o = rng.random(total + 1)
    tokens = np.empty(total + 1, dtype=np.int64)
    for t in range(total + 1):
        states[t] = s
        tokens[t] = np.searchsorted(cdf_E[s], u_o[t], side="right")
        s = int(np.searchsorted(cdf_P[s], u_s[t], side="right"))

    vals = np.empty(n_steps)
    if delay == 1:
        b = pi.copy()  # decision belief over s_t; token o_{t} measured after acting
        for t in range(total):
            if t >= burn_in:
                vals[t - burn_in] = b.max()
            b = predict(
                measure(b, E, tokens[t]),
                CONTROL_TRANSITION_MATRIX,
            )
    else:
        b = measure(pi, E, tokens[0])  # posterior over s_0
        for t in range(total):
            if t >= burn_in:
                vals[t - burn_in] = b.max()
            b = measure(
                predict(b, CONTROL_TRANSITION_MATRIX),
                E,
                tokens[t + 1],
            )

    n_batches = 100
    batches = vals[: n_steps - n_steps % n_batches].reshape(n_batches, -1).mean(axis=1)
    se = float(batches.std(ddof=1) / np.sqrt(n_batches))
    return float(vals.mean()), se


def stateguess_table(delay: int, alpha: float = 0.85, n_steps: int = 2_000_000,
                     seed: int = 0) -> StateGuessTable:
    mem_v, mem_map = best_memoryless(delay, alpha)
    ceiling, se = filter_ceiling_sim(delay, alpha, n_steps=n_steps, seed=seed)
    return StateGuessTable(
        delay=delay,
        random=1.0 / 3.0,
        memoryless=mem_v,
        memoryless_map=mem_map,
        filter_ceiling=ceiling,
        filter_ceiling_se=se,
    )
