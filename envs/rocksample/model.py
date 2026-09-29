"""Canonical RockSample instance data and domain constants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


DISCOUNT = 0.95
EXIT_REWARD = 10.0
GOOD_ROCK_REWARD = 10.0
BAD_ROCK_REWARD = -10.0


class Action(IntEnum):
    """Actions shared by every RockSample instance."""

    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    SAMPLE = 4


class Observation(IntEnum):
    """Observation symbols emitted by the environment."""

    GOOD = 0
    BAD = 1
    NONE = 2


@dataclass(frozen=True, slots=True)
class InstanceDefinition:
    """Fixed evaluation data for one RockSample instance."""

    n: int
    k: int
    start: tuple[int, int]
    rocks: tuple[tuple[int, int], ...]
    sensor_half_efficiency_distance: float
    layout_seed: int | None = None


CANONICAL_INSTANCES = {
    (4, 4): InstanceDefinition(
        n=4,
        k=4,
        start=(0, 2),
        rocks=((3, 1), (2, 1), (1, 3), (1, 0)),
        sensor_half_efficiency_distance=float(np.log(2.0)),
    ),
    (5, 5): InstanceDefinition(
        n=5,
        k=5,
        start=(0, 2),
        rocks=((2, 4), (0, 4), (3, 3), (2, 2), (4, 1)),
        sensor_half_efficiency_distance=4.0,
    ),
    (5, 7): InstanceDefinition(
        n=5,
        k=7,
        start=(0, 2),
        rocks=((1, 0), (2, 1), (1, 2), (2, 2), (4, 2), (0, 3), (3, 4)),
        sensor_half_efficiency_distance=20.0,
    ),
    (7, 8): InstanceDefinition(
        n=7,
        k=8,
        start=(0, 3),
        rocks=(
            (2, 0),
            (0, 1),
            (3, 1),
            (6, 3),
            (2, 4),
            (3, 4),
            (5, 5),
            (1, 6),
        ),
        sensor_half_efficiency_distance=20.0,
    ),
    (10, 10): InstanceDefinition(
        n=10,
        k=10,
        start=(0, 5),
        rocks=(
            (0, 3),
            (0, 7),
            (1, 8),
            (3, 3),
            (3, 8),
            (4, 3),
            (5, 8),
            (6, 1),
            (9, 3),
            (9, 9),
        ),
        sensor_half_efficiency_distance=20.0,
    ),
    (11, 11): InstanceDefinition(
        n=11,
        k=11,
        start=(0, 5),
        rocks=(
            (0, 3),
            (0, 7),
            (1, 8),
            (2, 4),
            (3, 3),
            (3, 8),
            (4, 3),
            (5, 8),
            (6, 1),
            (9, 3),
            (9, 9),
        ),
        sensor_half_efficiency_distance=20.0,
    ),
}


def check_action(rock_index: int) -> int:
    """Return the action index for checking one rock."""

    if rock_index < 0:
        raise ValueError("rock_index must be non-negative")
    return len(Action) + rock_index


def action_names(k: int) -> tuple[str, ...]:
    """Return action names in environment index order."""

    if k <= 0:
        raise ValueError("k must be positive")
    return (
        "north",
        "south",
        "east",
        "west",
        "sample",
        *(f"check_{index}" for index in range(k)),
    )


def sensor_efficiency(
    distance: float,
    half_efficiency_distance: float,
) -> float:
    """Return the canonical distance-dependent sensor efficiency."""

    return float(2.0 ** (-distance / half_efficiency_distance))


def instance_definition(
    n: int,
    k: int,
    *,
    layout_seed: int,
) -> InstanceDefinition:
    """Resolve canonical data or generate a deterministic non-standard layout."""

    canonical = CANONICAL_INSTANCES.get((n, k))
    if canonical is not None:
        return canonical

    start = (0, n // 2)
    candidates = np.array(
        [
            (x, y)
            for x in range(n)
            for y in range(n)
            if (x, y) != start
        ],
        dtype=np.int64,
    )
    rng = np.random.default_rng(layout_seed)
    indices = rng.choice(len(candidates), size=k, replace=False)
    rocks = tuple(
        (int(candidates[index, 0]), int(candidates[index, 1]))
        for index in indices
    )
    return InstanceDefinition(
        n=n,
        k=k,
        start=start,
        rocks=rocks,
        sensor_half_efficiency_distance=20.0,
        layout_seed=layout_seed,
    )
