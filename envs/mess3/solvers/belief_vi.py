"""Belief-MDP ceiling for MESS3-Continuous: average-reward value iteration on a
regular simplex grid (n >= 120 per edge), maximizing w over the 2D box.

Method
------
- Grid the 2-simplex; interpolate values barycentrically (piecewise-linear).
- Sweep a dense action lattice for the Bellman improvement step (vectorized:
  stencils for every (grid point, token, action) triple are precomputed), then
  accelerate with modified policy iteration (cheap evaluation sweeps under the
  greedy action).
- After convergence, polish per grid point with L-BFGS-B multi-started from
  the best lattice actions -> the continuous optimal map w*(belief).

Bellman operators (average reward, relative form):

delay = 1 (decision belief b over s_t conditions on tokens through o_{t-1}):
    Q(b, w) = b @ r - (b @ kl_w) / beta
              + sum_o P(o | b) * V( measure(b, o) @ U_w )        - rho
    with P(o | b) = b @ E[:, o]  (o_t is emitted from s_t, independent of w).

delay = 0 (decision belief b over s_t conditions on o_t):
    Q(b, w) = b @ r - (b @ kl_w) / beta
              + sum_o' P(o' | b, w) * V( measure(b @ U_w, o') )  - rho
    with P(o' | b, w) = (b @ U_w) @ E[:, o'].

The same machinery doubles as the Phase-5 discrete-action solver: pass an
explicit ``actions`` array and skip the polish — the lattice-only solution IS
the discrete-N belief MDP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from envs.mess3.core import (
    P0,
    REWARD_VEC,
    emission_matrix,
    kl_cost_per_state,
    kl_costs_batch,
    tilted_transition,
    tilted_transitions_batch,
)
from envs.mess3.filters import measure
from envs.mess3.solvers.simplex_grid import interp_weights, nearest_index, simplex_grid


def action_lattice(w_max: float, n_per_axis: int) -> np.ndarray:
    g = np.linspace(-w_max, w_max, n_per_axis)
    return np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)


@dataclass
class BeliefVISolution:
    beta: float
    w_max: float
    delay: int
    n_grid: int
    grid: np.ndarray          # (G, 3) belief grid
    actions: np.ndarray       # (K, 2) action lattice used in the sweep
    rho: float                # optimal average reward (the belief ceiling)
    V: np.ndarray             # relative values on the grid, V[0] = 0
    greedy_k: np.ndarray      # (G,) lattice-greedy action index
    W: np.ndarray | None = None  # (G, 2) polished continuous actions
    iterations: int = 0
    span: float = float("nan")

    def policy(self, belief: np.ndarray) -> np.ndarray:
        """Optimal action at an arbitrary belief (nearest polished grid point)."""
        W = self.W if self.W is not None else self.actions[self.greedy_k]
        return W[nearest_index(belief, self.n_grid)]


class _BellmanTensors:
    """Precomputed reward table and interpolation stencils for one config."""

    def __init__(self, grid, actions, E, beta, delay, base, n_grid):
        G, K = grid.shape[0], actions.shape[0]
        self.G, self.K, self.delay = G, K, delay
        U = tilted_transitions_batch(actions, base)          # (K, 3, 3)
        kl = kl_costs_batch(actions, base)                   # (K, 3)
        self.R = (grid @ REWARD_VEC)[:, None] - (grid @ kl.T) / beta  # (G, K)

        if delay == 1:
            self.tokprob = grid @ E                          # (G, O)
            post = grid[:, None, :] * E.T[None, :, :]        # (G, O, 3)
            post /= post.sum(axis=2, keepdims=True)
            nxt = np.einsum("gos,kst->gokt", post, U)        # (G, O, K, 3)
        else:
            prior = np.einsum("gs,kst->gkt", grid, U)        # (G, K, 3)
            self.tokprob = np.einsum("gkt,to->gko", prior, E)  # (G, K, O)
            nxt = prior[:, :, None, :] * E.T[None, None, :, :]  # (G, K, O, 3)
            nxt /= nxt.sum(axis=3, keepdims=True)
            nxt = nxt.transpose(0, 2, 1, 3)                  # (G, O, K, 3)
        idx, wts = interp_weights(nxt, n_grid)               # (G, O, K, 3) each
        self.idx = idx.astype(np.int32)
        self.wts = wts

    def expected_next_value(self, V: np.ndarray) -> np.ndarray:
        """EV[g, k] = sum_o P(o) * V_interp(next belief)."""
        Vn = (self.wts * V[self.idx]).sum(axis=3)            # (G, O, K)
        if self.delay == 1:
            return np.einsum("go,gok->gk", self.tokprob, Vn)
        return np.einsum("gko,gok->gk", self.tokprob, Vn)

    def next_value_for_policy(self, V: np.ndarray, k_of_g: np.ndarray) -> np.ndarray:
        g = np.arange(self.G)
        # Advanced indices at axes 0 and 2 (separated by a slice): result (G, O, 3).
        idx = self.idx[g, :, k_of_g, :]
        wts = self.wts[g, :, k_of_g, :]
        Vn = (wts * V[idx]).sum(axis=2)                      # (G, O)
        if self.delay == 1:
            return (self.tokprob * Vn).sum(axis=1)
        return (self.tokprob[g, k_of_g, :] * Vn).sum(axis=1)


def solve_belief_vi(
    beta: float,
    w_max: float,
    delay: int,
    n_grid: int = 120,
    n_act_per_axis: int = 21,
    actions: np.ndarray | None = None,
    alpha: float = 0.85,
    base: np.ndarray = P0,
    tol: float = 1e-9,
    max_outer: int = 400,
    eval_sweeps: int = 60,
    polish: bool = True,
) -> BeliefVISolution:
    grid = simplex_grid(n_grid)
    if actions is None:
        actions = action_lattice(w_max, n_act_per_axis)
    E = emission_matrix(alpha)
    T = _BellmanTensors(grid, actions, E, beta, delay, base, n_grid)

    V = np.zeros(T.G)
    rho, span, it = 0.0, float("inf"), 0
    greedy = np.zeros(T.G, dtype=np.int64)
    for it in range(1, max_outer + 1):
        Q = T.R + T.expected_next_value(V)
        greedy = Q.argmax(axis=1)
        TV = Q[np.arange(T.G), greedy]
        diff = TV - V
        span = float(diff.max() - diff.min())
        rho = 0.5 * float(diff.max() + diff.min())
        V = TV - TV[0]
        if span < tol:
            break
        # Modified policy iteration: cheap evaluation sweeps under the greedy policy.
        Rpi = T.R[np.arange(T.G), greedy]
        for _ in range(eval_sweeps):
            TV = Rpi + T.next_value_for_policy(V, greedy)
            V = TV - TV[0]

    sol = BeliefVISolution(
        beta=beta, w_max=w_max, delay=delay, n_grid=n_grid, grid=grid,
        actions=actions, rho=rho, V=V, greedy_k=greedy, iterations=it, span=span,
    )
    if polish:
        sol.W = _polish_actions(sol, E, beta, base)
    return sol


def _q_continuous(w, b, V, E, beta, delay, base, n_grid) -> float:
    """Q(b, w) with the converged interpolated V — the polish objective."""
    U = tilted_transition(w, base)
    kl = kl_cost_per_state(w, base)
    r = float(b @ REWARD_VEC - (b @ kl) / beta)
    if delay == 1:
        tok = b @ E                                          # (O,)
        nxt = np.stack([measure(b, E, o) @ U for o in range(E.shape[1])])
    else:
        prior = b @ U
        tok = prior @ E
        nxt = np.stack([measure(prior, E, o) for o in range(E.shape[1])])
    idx, wts = interp_weights(nxt, n_grid)
    Vn = (wts * V[idx]).sum(axis=1)
    return r + float(tok @ Vn)


def _polish_actions(
    sol: BeliefVISolution, E: np.ndarray, beta: float, base: np.ndarray, n_starts: int = 3
) -> np.ndarray:
    """L-BFGS-B refinement of the greedy lattice action, multi-started from the
    top-``n_starts`` lattice actions per grid point."""
    T = _BellmanTensors(sol.grid, sol.actions, E, beta, sol.delay, base, sol.n_grid)
    Q = T.R + T.expected_next_value(sol.V)
    top = np.argsort(Q, axis=1)[:, -n_starts:]
    bounds = [(-sol.w_max, sol.w_max)] * 2
    W = np.empty((T.G, 2))
    for g in range(T.G):
        b = sol.grid[g]
        best_w, best_q = None, -np.inf
        for k in top[g]:
            res = minimize(
                lambda w: -_q_continuous(w, b, sol.V, E, beta, sol.delay, base, sol.n_grid),
                sol.actions[k], method="L-BFGS-B", bounds=bounds,
            )
            if -res.fun > best_q:
                best_q, best_w = -res.fun, res.x
        W[g] = best_w
    return W
