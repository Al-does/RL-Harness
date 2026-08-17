"""Canonical factored model from Cassandra's machine-maintenance POMDP."""

from __future__ import annotations

from enum import IntEnum

import numpy as np


N_COMPONENTS = 4
N_CONDITIONS = 4
N_STATES = N_CONDITIONS**N_COMPONENTS
N_OBSERVATIONS = 2**N_COMPONENTS
DISCOUNT = 0.999


class Condition(IntEnum):
    """Ordered condition of one machine component."""

    BROKEN = 0
    BAD = 1
    FAIR = 2
    GOOD = 3


class Action(IntEnum):
    """Semantic names for the four numeric actions in ``machine.POMDP``."""

    OPERATE = 0
    INSPECT = 1
    REPAIR = 2
    REPLACE = 3


CONDITION_NAMES = ("broken", "bad", "fair", "good")
ACTION_NAMES = ("operate", "inspect", "repair", "replace")

# A component degrades by one condition with probability 0.03 while operating.
OPERATE_COMPONENT_TRANSITION = np.array(
    [
        [1.00, 0.00, 0.00, 0.00],
        [0.03, 0.97, 0.00, 0.00],
        [0.00, 0.03, 0.97, 0.00],
        [0.00, 0.00, 0.03, 0.97],
    ],
    dtype=np.float64,
)

# Repair improves each non-broken, non-good component by one level with
# probability 0.8. Broken components cannot be repaired by this action.
REPAIR_COMPONENT_TRANSITION = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.8, 0.0],
        [0.0, 0.0, 0.2, 0.8],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

INSPECT_COMPONENT_TRANSITION = np.eye(N_CONDITIONS, dtype=np.float64)
REPLACE_COMPONENT_TRANSITION = np.tile(
    np.array([0.0, 0.0, 0.0, 1.0]),
    (N_CONDITIONS, 1),
)

COMPONENT_TRANSITIONS = np.stack(
    [
        OPERATE_COMPONENT_TRANSITION,
        INSPECT_COMPONENT_TRANSITION,
        REPAIR_COMPONENT_TRANSITION,
        REPLACE_COMPONENT_TRANSITION,
    ]
)

# P(positive inspection bit | component condition). The canonical inspection
# emits one noisy binary reading per component.
INSPECTION_POSITIVE_PROBABILITY = np.array(
    [0.02, 0.05, 0.80, 0.97],
    dtype=np.float64,
)
INSPECTION_COMPONENT_OBSERVATION = np.column_stack(
    (
        1.0 - INSPECTION_POSITIVE_PROBABILITY,
        INSPECTION_POSITIVE_PROBABILITY,
    )
)

# P(a component passes production | post-transition condition). The machine's
# product passes only when every component passes, so operating emits either
# symbol 0 (failed product) or 15 (passed product).
OPERATE_COMPONENT_PASS_PROBABILITY = np.array(
    [0.0, 0.75, 0.95, 1.0],
    dtype=np.float64,
)

# The operating reward is the product of these four component quality terms.
# Each term is the expected post-degradation pass probability for one current
# component condition.
OPERATE_COMPONENT_REWARD = np.array(
    [0.0, 0.7275, 0.9440, 0.9985],
    dtype=np.float64,
)
ACTION_COST = np.array([0.0, -1.0, -3.0, -15.0], dtype=np.float64)

for _array in (
    OPERATE_COMPONENT_TRANSITION,
    REPAIR_COMPONENT_TRANSITION,
    INSPECT_COMPONENT_TRANSITION,
    REPLACE_COMPONENT_TRANSITION,
    COMPONENT_TRANSITIONS,
    INSPECTION_POSITIVE_PROBABILITY,
    INSPECTION_COMPONENT_OBSERVATION,
    OPERATE_COMPONENT_PASS_PROBABILITY,
    OPERATE_COMPONENT_REWARD,
    ACTION_COST,
):
    _array.setflags(write=False)


def encode_state(conditions: np.ndarray) -> int:
    """Encode component 0 as the least-significant base-four digit."""

    values = np.asarray(conditions, dtype=np.int64)
    if values.shape != (N_COMPONENTS,):
        raise ValueError(f"conditions must have shape ({N_COMPONENTS},)")
    if ((values < 0) | (values >= N_CONDITIONS)).any():
        raise ValueError("conditions contain an invalid component state")
    return int(values @ (N_CONDITIONS ** np.arange(N_COMPONENTS)))


def decode_state(state: int) -> np.ndarray:
    """Decode a canonical state index into four component conditions."""

    if not 0 <= int(state) < N_STATES:
        raise ValueError(f"state must lie in [0, {N_STATES})")
    value = int(state)
    return np.array(
        [
            (value // (N_CONDITIONS**component)) % N_CONDITIONS
            for component in range(N_COMPONENTS)
        ],
        dtype=np.int8,
    )


def encode_observation(bits: np.ndarray) -> int:
    """Encode component 0 as the least-significant observation bit."""

    values = np.asarray(bits, dtype=np.int64)
    if values.shape != (N_COMPONENTS,):
        raise ValueError(f"observation bits must have shape ({N_COMPONENTS},)")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("observation bits must be binary")
    return int(values @ (2 ** np.arange(N_COMPONENTS)))


def decode_observation(observation: int) -> np.ndarray:
    """Decode a canonical observation index into four binary readings."""

    if not 0 <= int(observation) < N_OBSERVATIONS:
        raise ValueError(f"observation must lie in [0, {N_OBSERVATIONS})")
    value = int(observation)
    return np.array(
        [(value // (2**component)) % 2 for component in range(N_COMPONENTS)],
        dtype=np.int8,
    )


def _factored_matrix(component_matrix: np.ndarray) -> np.ndarray:
    matrix = np.array([[1.0]], dtype=np.float64)
    for _ in range(N_COMPONENTS):
        matrix = np.kron(component_matrix, matrix)
    return matrix


def transition_matrix(action: int | Action) -> np.ndarray:
    """Return the canonical dense ``P(s' | s, action)`` matrix."""

    action_index = int(action)
    if not 0 <= action_index < len(Action):
        raise ValueError("invalid machine-maintenance action")
    return _factored_matrix(COMPONENT_TRANSITIONS[action_index])


def observation_matrix(action: int | Action) -> np.ndarray:
    """Return the canonical dense ``P(observation | s', action)`` matrix."""

    action_index = int(action)
    if not 0 <= action_index < len(Action):
        raise ValueError("invalid machine-maintenance action")
    if action_index == Action.INSPECT:
        return _factored_matrix(INSPECTION_COMPONENT_OBSERVATION)
    matrix = np.zeros((N_STATES, N_OBSERVATIONS), dtype=np.float64)
    if action_index == Action.OPERATE:
        pass_probability = np.array(
            [
                np.prod(
                    OPERATE_COMPONENT_PASS_PROBABILITY[decode_state(state)]
                )
                for state in range(N_STATES)
            ]
        )
        matrix[:, 0] = 1.0 - pass_probability
        matrix[:, N_OBSERVATIONS - 1] = pass_probability
    else:
        matrix[:, 0] = 1.0
    return matrix


def reward_vector(action: int | Action) -> np.ndarray:
    """Return the canonical immediate reward ``R(action, state)``."""

    action_index = int(action)
    if not 0 <= action_index < len(Action):
        raise ValueError("invalid machine-maintenance action")
    if action_index != Action.OPERATE:
        return np.full(N_STATES, ACTION_COST[action_index], dtype=np.float64)
    return np.array(
        [
            float(np.prod(OPERATE_COMPONENT_REWARD[decode_state(state)]))
            for state in range(N_STATES)
        ],
        dtype=np.float64,
    )
