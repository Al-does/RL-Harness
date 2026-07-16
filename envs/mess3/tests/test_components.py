"""Direct contract tests for concrete MESS3 tasks."""

import numpy as np
import pytest

from envs.hmm import ActionDecision, TransitionEvent
from envs.mess3.model import (
    CONTROL_TRANSITION_MATRIX,
    control_model,
    passive_model,
    state_guess_model,
)
from envs.mess3.tasks import (
    FutureStateGuessTask,
    OccupancyControlTask,
    PassiveTask,
    StateGuessTask,
)


def event(
    *,
    step: int = 0,
    state_before: int = 2,
    state_after: int = 1,
) -> TransitionEvent:
    return TransitionEvent(
        step=step,
        state_before=state_before,
        state_after=state_after,
        raw_token_before=state_before,
        raw_token_after=state_after,
    )


def test_occupancy_task_clips_action_and_reports_pre_transition_kl():
    model = control_model()
    task = OccupancyControlTask(
        model=model,
        action_limit=2.0,
        transition_kl_beta=4.0,
    )
    decision = task.resolve_action(
        np.array([10.0, -10.0]),
        state=2,
        model=model,
    )

    assert isinstance(decision, ActionDecision)
    np.testing.assert_allclose(decision.requested_action, [10.0, -10.0])
    np.testing.assert_allclose(decision.executed_action, [2.0, -2.0])
    np.testing.assert_allclose(decision.transition_matrix.sum(axis=1), 1.0)
    direct_kl = (
        decision.transition_matrix
        * np.log(decision.transition_matrix / CONTROL_TRANSITION_MATRIX)
    ).sum(axis=1)
    np.testing.assert_allclose(
        decision.metadata["transition_kl_by_state"],
        direct_kl,
    )

    reward, components = task.reward(event(), decision)
    expected_penalty = -direct_kl[2] / 4.0
    assert reward == pytest.approx(1.0 + expected_penalty)
    assert components["occupancy_reward"] == 1.0
    assert components["transition_kl"] == pytest.approx(direct_kl[2])
    assert components["transition_kl_penalty"] == pytest.approx(
        expected_penalty
    )


def test_occupancy_task_owns_an_immutable_reference_transition_copy():
    model = control_model()
    supplied = np.array(CONTROL_TRANSITION_MATRIX, copy=True)
    task = OccupancyControlTask(
        model=model,
        reference_transition_matrix=supplied,
    )
    supplied[0] = [1.0, 0.0, 0.0]

    np.testing.assert_allclose(
        task.reference_transition_matrix,
        CONTROL_TRANSITION_MATRIX,
    )
    with pytest.raises(ValueError):
        task.reference_transition_matrix[0, 0] = 0.0


def test_passive_task_ignores_action_for_dynamics_and_scores_occupancy():
    model = passive_model()
    task = PassiveTask(
        model=model,
        action_limit=0.25,
        occupancy_states=(1,),
    )
    decision = task.resolve_action(
        np.array([4.0, -4.0]),
        state=1,
        model=model,
    )

    np.testing.assert_allclose(decision.executed_action, [0.25, -0.25])
    np.testing.assert_allclose(
        decision.transition_matrix,
        model.transition_matrix,
    )
    reward, components = task.reward(
        event(state_before=1, state_after=0),
        decision,
    )
    assert reward == 1.0
    assert components == {"occupancy_reward": 1.0}


def test_state_guess_task_scores_current_state_without_changing_dynamics():
    model = state_guess_model()
    task = StateGuessTask(model=model)
    correct = task.resolve_action(2, state=2, model=model)
    incorrect = task.resolve_action(1, state=2, model=model)

    np.testing.assert_allclose(correct.transition_matrix, model.transition_matrix)
    np.testing.assert_allclose(
        incorrect.transition_matrix,
        model.transition_matrix,
    )
    correct_reward, correct_components = task.reward(event(), correct)
    incorrect_reward, incorrect_components = task.reward(event(), incorrect)
    assert correct_reward == 1.0
    assert incorrect_reward == 0.0
    assert correct_components["state_guess_valid"] == 1.0
    assert incorrect_components["state_guess_valid"] == 1.0
    np.testing.assert_array_equal(task.encode_action(2), [0.0, 0.0, 1.0])


def test_future_state_guess_queue_matures_at_horizon_and_discards_on_truncation():
    model = state_guess_model()
    task = FutureStateGuessTask(model=model, horizon=2)

    first_decision = task.resolve_action(2, state=0, model=model)
    first_reward, first_components = task.reward(
        event(step=0, state_before=0, state_after=1),
        first_decision,
    )
    assert first_reward == 0.0
    assert first_components["state_guess_valid"] == 0.0
    assert task.pending_predictions == 1

    second_decision = task.resolve_action(0, state=1, model=model)
    second_reward, second_components = task.reward(
        event(step=1, state_before=1, state_after=2),
        second_decision,
    )
    assert second_reward == 1.0
    assert second_components["state_guess_valid"] == 1.0
    assert task.pending_predictions == 1

    task.on_truncation()
    assert task.pending_predictions == 0
    after_truncation, components = task.reward(
        event(step=2, state_before=2, state_after=0),
        task.resolve_action(0, state=2, model=model),
    )
    assert after_truncation == 0.0
    assert components["state_guess_valid"] == 0.0


def test_future_state_guess_rejects_nonpositive_horizon():
    with pytest.raises(ValueError, match="horizon"):
        FutureStateGuessTask(model=state_guess_model(), horizon=0)
