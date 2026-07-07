"""Regular grid on the 2-simplex with barycentric (piecewise-linear) interpolation.

Grid points are b = (i, j, n - i - j) / n for i + j <= n, enumerated i-major.
Interpolation uses the standard Freudenthal split of each grid cell into a
lower and an upper micro-triangle.
"""

from __future__ import annotations

import numpy as np


def simplex_grid(n: int) -> np.ndarray:
    """All (i/n, j/n, (n-i-j)/n) points, shape ((n+1)(n+2)/2, 3), i-major order."""
    pts = [
        (i / n, j / n, (n - i - j) / n)
        for i in range(n + 1)
        for j in range(n + 1 - i)
    ]
    return np.array(pts)


def _row_offsets(n: int) -> np.ndarray:
    """offset[i] = flat index of grid point (i, 0)."""
    i = np.arange(n + 2)
    return (i * (n + 1) - i * (i - 1) // 2).astype(np.int64)


def flat_index(i: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    return _row_offsets(n)[i] + j


def interp_weights(points: np.ndarray, n: int):
    """Barycentric interpolation stencil for arbitrary simplex points.

    points: (..., 3) beliefs.  Returns (idx, wts), each (..., 3): flat grid
    indices and convex weights such that f(points) ~= sum wts * f_grid[idx].
    """
    p = np.asarray(points, dtype=np.float64)
    shape = p.shape[:-1]
    p = p.reshape(-1, 3)
    x = p[:, 0] * n
    y = p[:, 1] * n

    i0 = np.clip(np.floor(x).astype(np.int64), 0, n - 1)
    j0 = np.clip(np.floor(y).astype(np.int64), 0, n - 1)
    j0 = np.minimum(j0, n - 1 - i0)  # keep the cell inside the simplex
    fx = x - i0
    fy = y - j0

    # Lower micro-triangle: vertices (i0,j0), (i0+1,j0), (i0,j0+1) — always valid.
    idx = np.stack(
        [
            flat_index(i0, j0, n),
            flat_index(i0 + 1, j0, n),
            flat_index(i0, j0 + 1, n),
        ],
        axis=1,
    )
    wts = np.stack([1.0 - fx - fy, fx, fy], axis=1)

    # Upper micro-triangle where fx+fy > 1 and the (i0+1, j0+1) vertex exists.
    upper = (fx + fy > 1.0) & (i0 + j0 <= n - 2)
    if np.any(upper):
        iu, ju = i0[upper], j0[upper]
        idx[upper] = np.stack(
            [
                flat_index(iu + 1, ju + 1, n),
                flat_index(iu, ju + 1, n),
                flat_index(iu + 1, ju, n),
            ],
            axis=1,
        )
        wts[upper] = np.stack(
            [fx[upper] + fy[upper] - 1.0, 1.0 - fx[upper], 1.0 - fy[upper]], axis=1
        )

    # Numerical spill (points on the simplex boundary of a diagonal cell edge):
    # clip to the convex hull of the lower triangle and renormalize.
    wts = np.clip(wts, 0.0, None)
    wts /= wts.sum(axis=1, keepdims=True)
    return idx.reshape(*shape, 3), wts.reshape(*shape, 3)


def nearest_index(points: np.ndarray, n: int) -> np.ndarray:
    """Flat index of the nearest grid point (by rounded barycentric coords)."""
    p = np.asarray(points, dtype=np.float64)
    i = np.clip(np.round(p[..., 0] * n).astype(np.int64), 0, n)
    j = np.clip(np.round(p[..., 1] * n).astype(np.int64), 0, n)
    j = np.minimum(j, n - i)
    return flat_index(i, j, n)
