"""Oracle (full state observation) solver for MESS3-Continuous.

Unconstrained case: KL-control / linearly-solvable MDP.  With desirability
z(s) = exp(beta * V(s)), the optimality equation becomes the eigenproblem

    diag(exp(beta * r)) @ P0 @ z = exp(beta * rho) * z,

with rho the optimal average reward and optimal controlled row
u*(s'|s) proportional to P0[s, s'] * z(s').  In the gauge-fixed tilt
parameterization this corresponds to w*(s) = (log z(s+1) - log z(s),
log z(s+2) - log z(s)) (indices mod 3).

Box-constrained case: relative value iteration over the 3 physical states,
with the per-state action maximization done numerically inside the box
(dense lattice + L-BFGS-B polish from the best lattice point).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from envs.mess3.core import (
    N_STATES,
    P0,
    REWARD_VEC,
    kl_cost_per_state,
    kl_costs_batch,
    stationary_distribution,
    tilted_transition,
    tilted_transitions_batch,
)


@dataclass
class OracleSolution:
    rho: float                # optimal average reward
    V: np.ndarray             # relative values (bias), V[ref] = 0
    W: np.ndarray             # optimal action per state, shape (3, 2)
    U: np.ndarray             # optimal transition matrix, shape (3, 3)
    boundary: np.ndarray      # per state, per coord: |w| at the box edge?
    stationary: np.ndarray    # stationary distribution under U


def solve_oracle_unconstrained(beta: float, base: np.ndarray = P0) -> OracleSolution:
    A = np.diag(np.exp(beta * REWARD_VEC)) @ base
    vals, vecs = np.linalg.eig(A)
    i = int(np.argmax(np.real(vals)))
    lam = float(np.real(vals[i]))
    z = np.abs(np.real(vecs[:, i]))
    rho = np.log(lam) / beta
    V = np.log(z) / beta
    V = V - V[0]

    s = np.arange(N_STATES)
    logz = np.log(z)
    W = np.stack([logz[(s + 1) % 3] - logz[s], logz[(s + 2) % 3] - logz[s]], axis=1)
    U = base * z[None, :]
    U /= U.sum(axis=1, keepdims=True)
    return OracleSolution(
        rho=rho,
        V=V,
        W=W,
        U=U,
        boundary=np.zeros((N_STATES, 2), dtype=bool),
        stationary=stationary_distribution(U),
    )


def _state_q(w: np.ndarray, s: int, h: np.ndarray, beta: float, base: np.ndarray) -> float:
    U = tilted_transition(w, base)
    kl = kl_cost_per_state(w, base)[s]
    return REWARD_VEC[s] - kl / beta + float(U[s] @ h)


def _best_action_for_state(
    s: int, h: np.ndarray, beta: float, w_max: float, base: np.ndarray, n_lattice: int = 41
) -> tuple[np.ndarray, float]:
    grid = np.linspace(-w_max, w_max, n_lattice)
    W = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
    U = tilted_transitions_batch(W, base)
    kl = kl_costs_batch(W, base)
    q = REWARD_VEC[s] - kl[:, s] / beta + U[:, s, :] @ h
    w0 = W[int(np.argmax(q))]
    res = minimize(
        lambda w: -_state_q(w, s, h, beta, base),
        w0,
        method="L-BFGS-B",
        bounds=[(-w_max, w_max)] * 2,
    )
    return res.x, -res.fun


def solve_oracle_box(
    beta: float,
    w_max: float,
    base: np.ndarray = P0,
    tol: float = 1e-10,
    max_iter: int = 5000,
) -> OracleSolution:
    """Box-constrained oracle via relative value iteration over the 3 states."""
    h = np.zeros(N_STATES)
    rho = 0.0
    for _ in range(max_iter):
        Th = np.empty(N_STATES)
        for s in range(N_STATES):
            _, Th[s] = _best_action_for_state(s, h, beta, w_max, base)
        diff = Th - h
        rho = 0.5 * (diff.max() + diff.min())
        h_new = Th - Th[0]
        if diff.max() - diff.min() < tol:
            h = h_new
            break
        h = h_new

    W = np.empty((N_STATES, 2))
    for s in range(N_STATES):
        W[s], _ = _best_action_for_state(s, h, beta, w_max, base)
    U = np.stack([tilted_transition(W[s], base)[s] for s in range(N_STATES)])
    boundary = np.abs(np.abs(W) - w_max) < 1e-6
    return OracleSolution(
        rho=float(rho),
        V=h,
        W=W,
        U=U,
        boundary=boundary,
        stationary=stationary_distribution(U),
    )


def oracle_value_exact(W: np.ndarray, beta: float, base: np.ndarray = P0) -> float:
    """Exact average reward of a fixed per-state policy W (3, 2) — no MC needed."""
    U = np.stack([tilted_transition(W[s], base)[s] for s in range(N_STATES)])
    pi = stationary_distribution(U)
    kl = np.array([kl_cost_per_state(W[s], base)[s] for s in range(N_STATES)])
    return float(pi @ (REWARD_VEC - kl / beta))
