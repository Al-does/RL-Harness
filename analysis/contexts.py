"""Uniform finite-alphabet contexts for exhaustive probe evaluation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def discrete_context_count(n_symbols: int, context_length: int) -> int:
    """Return the number of fixed-length contexts in a finite alphabet."""
    if n_symbols <= 0:
        raise ValueError("n_symbols must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    return n_symbols**context_length


def iter_discrete_context_batches(
    n_symbols: int,
    context_length: int,
    *,
    batch_size: int,
    dtype: np.dtype | type = np.int64,
) -> Iterator[np.ndarray]:
    """Yield every fixed-length context exactly once in lexicographic order.

    The iterator avoids allocating the full exponentially sized context set.
    Experiment adapters remain responsible for computing model activations and
    exact targets for each batch.
    """
    total = discrete_context_count(n_symbols, context_length)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    powers = n_symbols ** np.arange(
        context_length - 1,
        -1,
        -1,
        dtype=np.int64,
    )
    for start in range(0, total, batch_size):
        indices = np.arange(
            start,
            min(start + batch_size, total),
            dtype=np.int64,
        )
        contexts = (indices[:, None] // powers[None, :]) % n_symbols
        yield contexts.astype(dtype, copy=False)
