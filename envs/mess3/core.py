"""Core MESS3-Control math: hidden chain, tilted transitions, emissions, KL cost.

Conventions used throughout the program
---------------------------------------
- Beliefs and distributions are ROW vectors; transition matrices are
  row-stochastic (``P[s, s']``), so prediction is ``b @ P``.
- Actions are gauge-fixed log-tilts ``w = (tilt_+1, tilt_+2)`` anchored to the
  occupied state: displacement d = (s' - s) mod 3 gets tilt
  ``[0, w[0], w[1]][d]``.  The self-loop tilt is structurally 0 (gauge choice),
  so the 2D box constraint is physical and w* (belief) can be injective.
- Reward: ``r_t = 1[s_t == 2] - (1/beta) * KL(u_w(.|s_t) || P0[s_t])``.
"""

from __future__ import annotations

import numpy as np

N_STATES = 3
N_TOKENS = 3
REWARD_STATE = 2

# Default (zero-action) transition matrix.  Stationary: (0.45, 0.45, 0.10).
P0 = np.array(
    [
        [0.75, 0.15, 0.10],
        [0.15, 0.75, 0.10],
        [0.45, 0.45, 0.10],
    ]
)

# Canonical MESS3(x=0.05) dynamics for passive mode (Shai et al. validation).
MESS3_PASSIVE_M = np.array(
    [
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
    ]
)

REWARD_VEC = np.array([0.0, 0.0, 1.0])


def emission_matrix(alpha: float = 0.85) -> np.ndarray:
    """E[s, o] = P(o | s): alpha on the diagonal, (1-alpha)/2 off-diagonal."""
    off = (1.0 - alpha) / 2.0
    E = np.full((N_STATES, N_TOKENS), off)
    np.fill_diagonal(E, alpha)
    return E


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Stationary row vector of a row-stochastic matrix (unit left eigenvector)."""
    vals, vecs = np.linalg.eig(P.T)
    i = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, i])
    pi = np.abs(pi)
    return pi / pi.sum()


def tilt_matrix(w: np.ndarray) -> np.ndarray:
    """T[s, s'] = tilt applied to the (s -> s') entry: [0, w0, w1][(s'-s) % 3]."""
    tilts = np.array([0.0, w[0], w[1]])
    s = np.arange(N_STATES)
    d = (s[None, :] - s[:, None]) % N_STATES
    return tilts[d]


def tilted_transition(w: np.ndarray, base: np.ndarray = P0) -> np.ndarray:
    """Row-stochastic U_w with U_w[s, s'] = base[s, s'] * exp(tilt(d)) / Z(s, w)."""
    T = tilt_matrix(np.asarray(w, dtype=np.float64))
    U = base * np.exp(T)
    return U / U.sum(axis=1, keepdims=True)


def kl_cost_per_state(w: np.ndarray, base: np.ndarray = P0) -> np.ndarray:
    """KL(u_w(.|s) || base[s]) for each s.

    Closed form: log(U/base) = T - log Z(s), so KL[s] = sum_s' U[s,s'] T[s,s'] - log Z(s).
    """
    T = tilt_matrix(np.asarray(w, dtype=np.float64))
    G = base * np.exp(T)
    Z = G.sum(axis=1)
    U = G / Z[:, None]
    return (U * T).sum(axis=1) - np.log(Z)


def reward_components(s: int, w: np.ndarray, beta: float, base: np.ndarray = P0):
    """(occupancy reward, control cost) for state s and action w; r = occ - cost."""
    occ = float(s == REWARD_STATE)
    cost = float(kl_cost_per_state(w, base)[s]) / beta
    return occ, cost


# ---------------------------------------------------------------------------
# Vectorized variants (used by the belief-VI solver's action-lattice sweeps).
# ---------------------------------------------------------------------------

def _batch_tilt_pieces(W: np.ndarray, base: np.ndarray):
    """Shared internals for the batch variants: (U, T, Z) for W of shape (K, 2)."""
    W = np.asarray(W, dtype=np.float64)
    K = W.shape[0]
    tilts = np.concatenate([np.zeros((K, 1)), W], axis=1)  # (K, 3): tilt per displacement
    s = np.arange(N_STATES)
    d = (s[None, :] - s[:, None]) % N_STATES  # (3, 3)
    T = tilts[:, d]  # (K, 3, 3)
    G = base[None, :, :] * np.exp(T)
    Z = G.sum(axis=2)
    return G / Z[:, :, None], T, Z


def tilted_transitions_batch(W: np.ndarray, base: np.ndarray = P0) -> np.ndarray:
    """U_w for a batch of actions.  W: (K, 2) -> (K, 3, 3) row-stochastic."""
    U, _, _ = _batch_tilt_pieces(W, base)
    return U


def kl_costs_batch(W: np.ndarray, base: np.ndarray = P0) -> np.ndarray:
    """KL(u_w(.|s) || base[s]) for a batch of actions.  (K, 2) -> (K, 3)."""
    U, T, Z = _batch_tilt_pieces(W, base)
    return (U * T).sum(axis=2) - np.log(Z)
