"""Focused tests for the MESS3 occupancy-control task."""

from __future__ import annotations

import numpy as np
import pytest

from envs.hmm import ActionDecision, TransitionEvent
from envs.mess3.model import control_model
from envs.mess3.tasks.occupancy_control import OccupancyControlTask


def test_action_norm_reward_uses_executed_two_dimensional_action():
    model = control_model()
    task = OccupancyControlTask(
        model=model,
        action_norm_coefficient=0.05,
    )
    decision = ActionDecision(
        requested_action=np.array([30.0, 40.0]),
        executed_action=np.array([3.0, 4.0]),
        transition_matrix=model.transition_matrix,
    )
    event = TransitionEvent(
        step=0,
        state_before=2,
        state_after=1,
        raw_token_before=2,
        raw_token_after=1,
    )

    reward, components = task.reward(event, decision)

    assert components["action_norm"] == pytest.approx(5.0)
    assert components["action_norm_penalty"] == pytest.approx(-0.25)
    assert reward == pytest.approx(0.75)
