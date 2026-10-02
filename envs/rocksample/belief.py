"""Exact action/observation-only posteriors over current rock qualities."""

from __future__ import annotations

import numpy as np

from envs.rocksample.model import Action, Observation, sensor_efficiency


def configuration_bits(k: int) -> np.ndarray:
    """Bit i is one iff rock i is Good; configuration zero is all Bad."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    return ((np.arange(2**k)[:, None] >> np.arange(k)) & 1).astype(bool)


def joint_from_marginals(marginals: np.ndarray) -> np.ndarray:
    """Product posterior; accepts one vector or a batch of vectors."""
    m = np.asarray(marginals, dtype=np.float64)
    if m.ndim < 1 or m.shape[-1] == 0 or not np.isfinite(m).all():
        raise ValueError("marginals must be finite with nonzero width")
    if ((m < 0) | (m > 1)).any():
        raise ValueError("marginals must lie in [0, 1]")
    bits = configuration_bits(m.shape[-1])
    return np.prod(np.where(bits, m[..., None, :], 1 - m[..., None, :]), axis=-1)


def _event(
    k: int,
    rocks: np.ndarray,
    position: tuple[int, int] | np.ndarray,
    action: int,
    symbol: int,
    half_efficiency_distance: float,
) -> tuple[int | None, float | None]:
    rocks = np.asarray(rocks, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    if rocks.shape != (k, 2) or position.shape != (2,):
        raise ValueError("rocks and position must have shapes (k, 2) and (2,)")
    if not np.isfinite(rocks).all() or not np.isfinite(position).all():
        raise ValueError("rocks and position must be finite")
    if not np.isfinite(half_efficiency_distance) or half_efficiency_distance <= 0:
        raise ValueError("half_efficiency_distance must be positive and finite")
    if not isinstance(action, (int, np.integer)) or not 0 <= action < len(Action) + k:
        raise ValueError("invalid action")
    if action >= len(Action):
        if symbol not in (Observation.GOOD, Observation.BAD):
            raise ValueError("a Check must return Good or Bad")
        rock = action - len(Action)
        efficiency = sensor_efficiency(
            float(np.linalg.norm(position - rocks[rock])),
            half_efficiency_distance,
        )
        return rock, efficiency
    if symbol != Observation.NONE:
        raise ValueError("moves and Sample must return None")
    if action == Action.SAMPLE:
        matches = np.flatnonzero(np.all(rocks == position, axis=1))
        if len(matches):
            return int(matches[0]), None
    return None, None


def update_marginals(
    marginals: np.ndarray,
    *,
    rocks: np.ndarray,
    position: tuple[int, int] | np.ndarray,
    action: int,
    symbol: int,
    half_efficiency_distance: float,
) -> np.ndarray:
    """Update M_t to M_{t+1}, using the position *before* the action.

    Reset to a vector of 0.5 for each episode. Call only on non-exit
    transitions. Rewards and true qualities are deliberately absent.
    """
    m = np.asarray(marginals, dtype=np.float64).copy()
    if (
        m.ndim != 1
        or len(m) == 0
        or not np.isfinite(m).all()
        or ((m < 0) | (m > 1)).any()
    ):
        raise ValueError("marginals must be a probability vector")
    rock, efficiency = _event(
        len(m), rocks, position, action, symbol, half_efficiency_distance
    )
    if rock is None:
        return m
    if efficiency is None:
        m[rock] = 0.0
        return m
    good_likelihood = (1 + efficiency * (1 if symbol == Observation.GOOD else -1)) / 2
    good_mass = m[rock] * good_likelihood
    evidence = good_mass + (1 - m[rock]) * (1 - good_likelihood)
    if evidence <= 0:
        raise ValueError("observation has zero probability under the belief")
    m[rock] = good_mass / evidence
    return m


def update_joint(
    joint: np.ndarray,
    *,
    rocks: np.ndarray,
    position: tuple[int, int] | np.ndarray,
    action: int,
    symbol: int,
    half_efficiency_distance: float,
) -> np.ndarray:
    """Apply the transition/likelihood operator directly to J_t.

    Works for correlated priors too; does not reconstruct from marginals.
    """
    j = np.asarray(joint, dtype=np.float64).copy()
    if j.ndim != 1 or len(j) < 2 or len(j) & (len(j) - 1):
        raise ValueError("joint width must be 2**k with k > 0")
    if not np.isfinite(j).all() or (j < 0).any() or not np.isclose(j.sum(), 1):
        raise ValueError("joint must be a normalized probability vector")
    k = len(j).bit_length() - 1
    rock, efficiency = _event(
        k, rocks, position, action, symbol, half_efficiency_distance
    )
    if rock is None:
        return j
    bits = configuration_bits(k)[:, rock]
    if efficiency is None:
        destinations = np.arange(len(j)) & ~(1 << rock)
        return np.bincount(destinations, weights=j, minlength=len(j))
    matches = bits == (symbol == Observation.GOOD)
    j *= np.where(matches, (1 + efficiency) / 2, (1 - efficiency) / 2)
    evidence = j.sum()
    if evidence <= 0:
        raise ValueError("observation has zero probability under the belief")
    return j / evidence
