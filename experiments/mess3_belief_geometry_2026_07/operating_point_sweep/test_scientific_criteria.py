"""Study-specific acceptance criteria for the selected operating point."""

import pytest

from envs.mess3.solvers.belief_vi import solve_belief_vi
from envs.mess3.solvers.reactive import solve_reactive, solve_stack2


@pytest.mark.slow
def test_memory_premiums_at_selected_operating_point():
    beta = 4.0
    action_limit = 5.0
    belief = solve_belief_vi(
        beta,
        action_limit,
        delay=1,
        polish=False,
    )
    reactive = solve_reactive(beta, action_limit, delay=1)
    stack_2 = solve_stack2(beta, action_limit, delay=1)

    assert (belief.rho - reactive.value) / belief.rho >= 0.15
    assert (belief.rho - stack_2.value) / belief.rho >= 0.08
