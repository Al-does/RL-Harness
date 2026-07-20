"""Continuous MESS3 transition control with an occupancy objective."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent
from envs.mess3.model import CONTROL_TRANSITION_MATRIX, N_STATES

REWARD_STATE = 2
REWARD_VEC = np.array([0.0, 0.0, 1.0])
REWARD_VEC.setflags(write=False)

_STATES = np.arange(N_STATES)
_DISPLACEMENTS = (
    _STATES[None, :] - _STATES[:, None]
) % N_STATES


def tilt_matrix(action: np.ndarray) -> np.ndarray:
    """Return the anchored log tilt applied to each source/destination pair."""

    action = np.asarray(action, dtype=np.float64)
    if action.shape != (2,):
        raise ValueError("a MESS3 tilt action must have shape (2,)")
    tilts = np.array([0.0, action[0], action[1]])
    return tilts[_DISPLACEMENTS]


def controlled_transition_and_kl(
    action: np.ndarray,
    base: np.ndarray = CONTROL_TRANSITION_MATRIX,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the tilted transition and per-source-state KL in one pass."""

    tilts = tilt_matrix(action)
    unnormalized = np.asarray(base, dtype=np.float64) * np.exp(tilts)
    normalizer = unnormalized.sum(axis=1)
    transition = unnormalized / normalizer[:, None]
    kl = (transition * tilts).sum(axis=1) - np.log(normalizer)
    return transition, kl


def tilted_transition(
    action: np.ndarray,
    base: np.ndarray = CONTROL_TRANSITION_MATRIX,
) -> np.ndarray:
    """Return the row-stochastic transition selected by one tilt action."""

    transition, _ = controlled_transition_and_kl(action, base)
    return transition


def kl_cost_per_state(
    action: np.ndarray,
    base: np.ndarray = CONTROL_TRANSITION_MATRIX,
) -> np.ndarray:
    """Return ``KL(U_action(.|s) || base[s])`` for each source state."""

    _, kl = controlled_transition_and_kl(action, base)
    return kl


def _batch_tilt_pieces(
    actions: np.ndarray,
    base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("batched MESS3 actions must have shape (n, 2)")
    count = actions.shape[0]
    tilts = np.concatenate([np.zeros((count, 1)), actions], axis=1)
    tilt_matrices = tilts[:, _DISPLACEMENTS]
    unnormalized = np.asarray(base)[None, :, :] * np.exp(tilt_matrices)
    normalizer = unnormalized.sum(axis=2)
    return (
        unnormalized / normalizer[:, :, None],
        tilt_matrices,
        normalizer,
    )


def tilted_transitions_batch(
    actions: np.ndarray,
    base: np.ndarray = CONTROL_TRANSITION_MATRIX,
) -> np.ndarray:
    """Vectorized :func:`tilted_transition` for shape ``(n, 2)`` actions."""

    transitions, _, _ = _batch_tilt_pieces(actions, base)
    return transitions


def kl_costs_batch(
    actions: np.ndarray,
    base: np.ndarray = CONTROL_TRANSITION_MATRIX,
) -> np.ndarray:
    """Vectorized :func:`kl_cost_per_state` for shape ``(n, 2)`` actions."""

    transitions, tilts, normalizers = _batch_tilt_pieces(actions, base)
    return (transitions * tilts).sum(axis=2) - np.log(normalizers)


class OccupancyControlTask:
    """Tilt MESS3 transitions and reward selected pre-transition states."""

    requires_belief = False

    def __init__(
        self,
        *,
        model: HMMModel,
        action_limit: float = 5.0,
        occupancy_states: Sequence[int] = (REWARD_STATE,),
        transition_kl_beta: float | None = None,
        report_transition_kl: bool = False,
        action_norm_coefficient: float | None = None,
        report_action_norm: bool = False,
        reference_transition_matrix: Sequence[Sequence[float]] | None = None,
    ) -> None:
        if not np.isfinite(action_limit) or action_limit <= 0.0:
            raise ValueError("action_limit must be finite and positive")
        states = tuple(int(state) for state in occupancy_states)
        if not states:
            raise ValueError("occupancy_states must not be empty")
        if any(not 0 <= state < model.n_states for state in states):
            raise ValueError("occupancy state is outside the model state space")
        if transition_kl_beta is not None and transition_kl_beta <= 0.0:
            raise ValueError("transition_kl_beta must be positive")
        if (
            action_norm_coefficient is not None
            and (
                not np.isfinite(action_norm_coefficient)
                or action_norm_coefficient <= 0.0
            )
        ):
            raise ValueError("action_norm_coefficient must be finite and positive")

        reference = np.array(
            (
                model.transition_matrix
                if reference_transition_matrix is None
                else reference_transition_matrix
            ),
            dtype=np.float64,
            copy=True,
        )
        expected_shape = (model.n_states, model.n_states)
        if reference.shape != expected_shape:
            raise ValueError(
                f"reference_transition_matrix must have shape {expected_shape}"
            )
        if (
            (reference < 0.0).any()
            or not np.isfinite(reference).all()
            or not np.allclose(reference.sum(axis=1), 1.0, atol=1e-12)
        ):
            raise ValueError(
                "reference_transition_matrix must be row-stochastic"
            )

        self.action_limit = float(action_limit)
        self.occupancy_states = frozenset(states)
        self.transition_kl_beta = transition_kl_beta
        self.report_transition_kl = bool(report_transition_kl)
        self.action_norm_coefficient = action_norm_coefficient
        self.report_action_norm = bool(report_action_norm)
        reference.setflags(write=False)
        self.reference_transition_matrix = reference
        self.action_space = gym.spaces.Box(
            low=-self.action_limit,
            high=self.action_limit,
            shape=(2,),
            dtype=np.float32,
        )
        self.action_observation_space = self.action_space

    def reset(self) -> None:
        pass

    def resolve_action(
        self,
        action: np.ndarray,
        state: int,
        model: HMMModel,
    ) -> ActionDecision:
        del state, model
        requested = np.asarray(action, dtype=np.float64)
        if requested.shape != (2,):
            raise ValueError("occupancy-control action must have shape (2,)")
        executed = np.clip(
            requested,
            -self.action_limit,
            self.action_limit,
        )
        transition, transition_kl = controlled_transition_and_kl(
            executed,
            self.reference_transition_matrix,
        )
        return ActionDecision(
            requested_action=requested,
            executed_action=executed,
            transition_matrix=transition,
            metadata={
                "transition_kl_by_state": transition_kl,
                "reference_transition_matrix": (
                    self.reference_transition_matrix
                ),
            },
        )

    def reward(
        self,
        event: TransitionEvent,
        decision: ActionDecision,
    ) -> tuple[float, dict[str, float]]:
        occupancy = float(event.state_before in self.occupancy_states)
        reward = occupancy
        components = {"occupancy_reward": occupancy}
        if self.transition_kl_beta is not None or self.report_transition_kl:
            transition_kl = float(
                decision.metadata["transition_kl_by_state"][
                    event.state_before
                ]
            )
            components["transition_kl"] = transition_kl
            if self.transition_kl_beta is not None:
                penalty = -transition_kl / self.transition_kl_beta
                components["transition_kl_penalty"] = penalty
                reward += penalty
        if self.action_norm_coefficient is not None or self.report_action_norm:
            action_norm = float(np.linalg.norm(decision.executed_action))
            components["action_norm"] = action_norm
            if self.action_norm_coefficient is not None:
                penalty = -self.action_norm_coefficient * action_norm
                components["action_norm_penalty"] = penalty
                reward += penalty
        return reward, components

    def encode_action(self, executed_action: np.ndarray) -> np.ndarray:
        return np.asarray(executed_action, dtype=np.float32)
