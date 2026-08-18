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


class TargetedAction(IntEnum):
    """Actions for the component-addressable maintenance variant."""

    OPERATE = 0
    INSPECT = 1
    REPAIR_COMPONENT_0 = 2
    REPAIR_COMPONENT_1 = 3
    REPAIR_COMPONENT_2 = 4
    REPAIR_COMPONENT_3 = 5
    REPLACE_COMPONENT_0 = 6
    REPLACE_COMPONENT_1 = 7
    REPLACE_COMPONENT_2 = 8
    REPLACE_COMPONENT_3 = 9


CONDITION_NAMES = ("broken", "bad", "fair", "good")
ACTION_NAMES = ("operate", "inspect", "repair", "replace")
TARGETED_ACTION_NAMES = (
    "operate",
    "inspect",
    "repair_component_0",
    "repair_component_1",
    "repair_component_2",
    "repair_component_3",
    "replace_component_0",
    "replace_component_1",
    "replace_component_2",
    "replace_component_3",
)

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
TARGETED_COMPONENT_TRANSITIONS = np.broadcast_to(
    np.eye(N_CONDITIONS, dtype=np.float64),
    (len(TargetedAction), N_COMPONENTS, N_CONDITIONS, N_CONDITIONS),
).copy()
TARGETED_COMPONENT_TRANSITIONS[TargetedAction.OPERATE, :] = (
    OPERATE_COMPONENT_TRANSITION
)
for _component in range(N_COMPONENTS):
    TARGETED_COMPONENT_TRANSITIONS[
        TargetedAction.REPAIR_COMPONENT_0 + _component,
        _component,
    ] = REPAIR_COMPONENT_TRANSITION
    TARGETED_COMPONENT_TRANSITIONS[
        TargetedAction.REPLACE_COMPONENT_0 + _component,
        _component,
    ] = REPLACE_COMPONENT_TRANSITION

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
TARGETED_ACTION_COST = np.array(
    [0.0, -1.0, *([-0.75] * N_COMPONENTS), *([-3.75] * N_COMPONENTS)],
    dtype=np.float64,
)

for _array in (
    OPERATE_COMPONENT_TRANSITION,
    REPAIR_COMPONENT_TRANSITION,
    INSPECT_COMPONENT_TRANSITION,
    REPLACE_COMPONENT_TRANSITION,
    COMPONENT_TRANSITIONS,
    TARGETED_COMPONENT_TRANSITIONS,
    INSPECTION_POSITIVE_PROBABILITY,
    INSPECTION_COMPONENT_OBSERVATION,
    OPERATE_COMPONENT_PASS_PROBABILITY,
    OPERATE_COMPONENT_REWARD,
    ACTION_COST,
    TARGETED_ACTION_COST,
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


def _componentwise_matrix(component_matrices: np.ndarray) -> np.ndarray:
    matrix = np.array([[1.0]], dtype=np.float64)
    for component_matrix in component_matrices:
        matrix = np.kron(component_matrix, matrix)
    return matrix


def _action_count(action_scope: str) -> int:
    if action_scope == "global":
        return len(Action)
    if action_scope == "targeted":
        return len(TargetedAction)
    raise ValueError("action_scope must be 'global' or 'targeted'")


def action_names(action_scope: str = "global") -> tuple[str, ...]:
    """Return ordered action labels for one maintenance action scope."""

    _action_count(action_scope)
    return ACTION_NAMES if action_scope == "global" else TARGETED_ACTION_NAMES


def component_transition_matrices(
    action: int | Action | TargetedAction,
    *,
    action_scope: str = "global",
) -> np.ndarray:
    """Return one condition-transition matrix per component."""

    action_index = int(action)
    if not 0 <= action_index < _action_count(action_scope):
        raise ValueError("invalid machine-maintenance action")
    if action_scope == "global":
        return np.broadcast_to(
            COMPONENT_TRANSITIONS[action_index],
            (N_COMPONENTS, N_CONDITIONS, N_CONDITIONS),
        )
    return TARGETED_COMPONENT_TRANSITIONS[action_index]


def transition_matrix(
    action: int | Action | TargetedAction,
    *,
    action_scope: str = "global",
) -> np.ndarray:
    """Return dense ``P(s' | s, action)`` for the selected action scope."""

    return _componentwise_matrix(
        component_transition_matrices(action, action_scope=action_scope)
    )


def observation_matrix(
    action: int | Action | TargetedAction,
    *,
    action_scope: str = "global",
) -> np.ndarray:
    """Return dense ``P(observation | s', action)`` for one action scope."""

    action_index = int(action)
    if not 0 <= action_index < _action_count(action_scope):
        raise ValueError("invalid machine-maintenance action")
    if action_index == int(Action.INSPECT):
        return _factored_matrix(INSPECTION_COMPONENT_OBSERVATION)
    matrix = np.zeros((N_STATES, N_OBSERVATIONS), dtype=np.float64)
    if action_index == int(Action.OPERATE):
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


def reward_vector(
    action: int | Action | TargetedAction,
    *,
    action_scope: str = "global",
) -> np.ndarray:
    """Return immediate reward ``R(action, state)`` for one action scope."""

    action_index = int(action)
    if not 0 <= action_index < _action_count(action_scope):
        raise ValueError("invalid machine-maintenance action")
    if action_index != int(Action.OPERATE):
        costs = (
            ACTION_COST
            if action_scope == "global"
            else TARGETED_ACTION_COST
        )
        return np.full(N_STATES, costs[action_index], dtype=np.float64)
    return np.array(
        [
            float(np.prod(OPERATE_COMPONENT_REWARD[decode_state(state)]))
            for state in range(N_STATES)
        ],
        dtype=np.float64,
    )
