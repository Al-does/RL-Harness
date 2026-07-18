"""Stable derivation of purpose-specific NumPy random streams."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

import numpy as np


SeedSource: TypeAlias = int | np.random.SeedSequence
SeedKey: TypeAlias = int | tuple[int, ...]


def as_seed_sequence(seed: SeedSource) -> np.random.SeedSequence:
    """Normalize an external seed without consuming or spawning children."""
    if isinstance(seed, np.random.SeedSequence):
        return seed
    return np.random.SeedSequence(seed)


def child_seed_sequence(
    seed: SeedSource,
    key: SeedKey,
) -> np.random.SeedSequence:
    """Derive one stable child by explicit spawn key.

    Explicit keys avoid the order sensitivity of ``SeedSequence.spawn()``.
    Existing keys must never be repurposed, but new unique keys are safe to add.
    """
    parent = as_seed_sequence(seed)
    suffix = (key,) if isinstance(key, int) else tuple(key)
    if not suffix or any(part < 0 for part in suffix):
        raise ValueError("seed child keys must contain non-negative integers")
    return np.random.SeedSequence(
        parent.entropy,
        spawn_key=(*parent.spawn_key, *suffix),
        pool_size=parent.pool_size,
    )


def named_seed_sequences(
    seed: SeedSource,
    stream_keys: Mapping[str, SeedKey],
) -> dict[str, np.random.SeedSequence]:
    """Map purpose names to stable, explicitly keyed child sequences."""
    normalized_keys = [
        (key,) if isinstance(key, int) else tuple(key)
        for key in stream_keys.values()
    ]
    if len(set(normalized_keys)) != len(normalized_keys):
        raise ValueError("named random streams must use distinct child keys")
    return {
        name: child_seed_sequence(seed, key)
        for name, key in stream_keys.items()
    }


def seed_sequence_to_int(
    seed: SeedSource,
    *,
    bits: int = 32,
) -> int:
    """Materialize an integer only for an API that cannot accept SeedSequence."""
    if bits == 32:
        dtype = np.uint32
    elif bits == 64:
        dtype = np.uint64
    else:
        raise ValueError("seed integer width must be 32 or 64 bits")
    return int(as_seed_sequence(seed).generate_state(1, dtype=dtype)[0])
