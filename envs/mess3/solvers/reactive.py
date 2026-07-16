"""Memoryless (reactive), stack-2, and constant-action ceilings for
MESS3-Continuous, by direct optimization of the EXACT induced stationary
average reward.

A deterministic map f: token-context -> w induces a finite Markov chain on
(hidden state, token context); its stationary distribution — hence the
average reward — is an exact, smooth function of the action table.  We
optimize the table with L-BFGS-B (finite-difference gradients) from multiple
starts, so no Monte Carlo enters the ceiling itself (MC is used only to
cross-validate, see mc.py).

Context conventions (deterministic policies; the agent's own past actions add
no information beyond past tokens, so contexts are token histories):
  - reactive: context = the token visible at decision time
    (o_{t-1} when delay=1, o_t when delay=0); 3 contexts.
  - stack-2: the last TWO visible tokens; 9 contexts.
  - constant: no context; the no-information optimum.
The measure-zero t=0 "no token yet" observation is ignored (stationary limit).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import minimize

from envs.hmm import stationary_distribution
from envs.mess3.model import (
    CONTROL_TRANSITION_MATRIX,
    N_STATES,
    N_TOKENS,
    emission_matrix,
)
from envs.mess3.tasks.occupancy_control import (
    REWARD_VEC,
    kl_costs_batch,
    tilted_transitions_batch,
)


@dataclass
class ReactiveSolution:
    value: float               # exact stationary average reward
    table: np.ndarray          # (n_contexts, 2) optimal action per context
    kind: str                  # "constant" | "reactive" | "stack2"
    delay: int


def chain_value(table: np.ndarray, delay: int, depth: int, alpha: float, beta: float,
                base: np.ndarray) -> float:
    """Exact average reward of the deterministic token-context policy ``table``.

    depth = 0: constant policy; chain on s_t alone.
    depth = 1: chain on (s_t, visible token); depth = 2 adds one more token.
    """
    E = emission_matrix(alpha)
    W = table.reshape(-1, 2)
    U = tilted_transitions_batch(W, base)      # (C, 3, 3)
    kl = kl_costs_batch(W, base)               # (C, 3)

    if depth == 0:
        pi = stationary_distribution(U[0])
        return float(pi @ (REWARD_VEC - kl[0] / beta))

    # Enumerate joint states (s, c) where c indexes the token context.
    contexts = list(product(range(N_TOKENS), repeat=depth))
    C = len(contexts)
    n = N_STATES * C
    T = np.zeros((n, n))
    r = np.zeros(n)
    for ci, ctx in enumerate(contexts):
        # ctx = (newest visible token, ..., oldest); action for this context.
        u, k = U[ci], kl[ci]
        for s in range(N_STATES):
            i = s * C + ci
            r[i] = REWARD_VEC[s] - k[s] / beta
            for s2 in range(N_STATES):
                if delay == 1:
                    # Newest visible token at t+1 is o_t, emitted from s_t = s.
                    for o in range(N_TOKENS):
                        ctx2 = (o,) + ctx[: depth - 1]
                        j = s2 * C + contexts.index(ctx2)
                        T[i, j] += u[s, s2] * E[s, o]
                else:
                    # Newest visible token at t+1 is o_{t+1}, emitted from s_{t+1} = s2.
                    for o in range(N_TOKENS):
                        ctx2 = (o,) + ctx[: depth - 1]
                        j = s2 * C + contexts.index(ctx2)
                        T[i, j] += u[s, s2] * E[s2, o]
    pi = stationary_distribution(T)
    return float(pi @ r)


def _polish(starts, C, delay, depth, alpha, beta, w_max, base):
    def neg_value(x):
        return -chain_value(x.reshape(C, 2), delay, depth, alpha, beta, base)

    best_x, best_v = None, -np.inf
    for x0 in starts:
        res = minimize(neg_value, np.asarray(x0, dtype=np.float64),
                       method="L-BFGS-B", bounds=[(-w_max, w_max)] * (2 * C))
        if -res.fun > best_v:
            best_v, best_x = -res.fun, res.x
    return best_x, best_v


def _lattice_starts(C, delay, depth, alpha, beta, w_max, base, n_pts, n_keep):
    """Coarse global lattice scan over the full table; return the top tables.

    The stationary-value landscape is multimodal (L-BFGS from few random
    starts reliably finds only local optima), so global coverage matters more
    than polish density."""
    g = np.linspace(-w_max, w_max, n_pts)
    combos = np.array(list(product(g, repeat=2 * C)))
    vals = np.array([
        chain_value(x.reshape(C, 2), delay, depth, alpha, beta, base) for x in combos
    ])
    return combos[np.argsort(vals)[-n_keep:]]


def solve_constant(
    beta,
    w_max,
    alpha=0.85,
    base=CONTROL_TRANSITION_MATRIX,
    n_restarts=8,
    seed=0,
) -> ReactiveSolution:
    starts = _lattice_starts(1, 1, 0, alpha, beta, w_max, base, n_pts=9, n_keep=6)
    x, v = _polish(starts, 1, 1, 0, alpha, beta, w_max, base)
    return ReactiveSolution(value=v, table=x.reshape(1, 2), kind="constant", delay=1)


def solve_reactive(
    beta,
    w_max,
    delay,
    alpha=0.85,
    base=CONTROL_TRANSITION_MATRIX,
    n_restarts=8,
    seed=0,
) -> ReactiveSolution:
    # 5^6 = 15625 lattice evaluations (~2 s) give global coverage of the 6D table.
    starts = _lattice_starts(3, delay, 1, alpha, beta, w_max, base, n_pts=5, n_keep=10)
    x, v = _polish(starts, 3, delay, 1, alpha, beta, w_max, base)
    return ReactiveSolution(value=v, table=x.reshape(3, 2), kind="reactive", delay=delay)


def solve_stack2(
    beta,
    w_max,
    delay,
    alpha=0.85,
    base=CONTROL_TRANSITION_MATRIX,
    n_restarts=24,
    seed=0,
) -> ReactiveSolution:
    """18D table: lattice scan is infeasible, so seed from the reactive optimum
    (replicated over the older token) plus corner and random restarts."""
    reactive = solve_reactive(beta, w_max, delay, alpha, base)
    rng = np.random.default_rng(seed)
    # Context index = newest * 3 + oldest (itertools.product order).
    seeded = np.repeat(reactive.table, N_TOKENS, axis=0).reshape(-1)
    starts = [seeded, np.zeros(18), np.full(18, w_max), np.full(18, -w_max)]
    starts += [rng.choice([-w_max, 0.0, w_max], size=18) for _ in range(n_restarts // 2)]
    starts += [rng.uniform(-w_max, w_max, size=18) for _ in range(n_restarts - n_restarts // 2)]
    x, v = _polish(starts, 9, delay, 2, alpha, beta, w_max, base)
    return ReactiveSolution(value=v, table=x.reshape(9, 2), kind="stack2", delay=delay)
