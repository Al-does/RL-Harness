"""Solver regression tests against the known-good values from prior rounds
(these are the Phase-1 gate's ground truth) plus internal cross-checks.

Known-good at (beta=4, w_max2=5, delay=1):
    reactive 0.192, stack-2 0.295, belief ceiling 0.381, oracle 0.4625.
Note: our stack-2 solver finds 0.3055 (> 0.295) — a strictly better table,
MC-confirmed; documented in FINDINGS_phase1.md.  The test asserts >= the
prior value rather than equality, since a ceiling can only legitimately move
UP with a better optimizer.
"""

import numpy as np
import pytest

from envs.mess3.solvers.belief_vi import solve_belief_vi
from envs.mess3.solvers.oracle import (
    oracle_value_exact,
    solve_oracle_box,
    solve_oracle_unconstrained,
)
from envs.mess3.solvers.reactive import solve_constant, solve_reactive, solve_stack2
from envs.mess3.solvers.stateguess_analytic import best_memoryless, joint_state_token


BETA, WMAX = 4.0, 5.0


class TestOracle:
    def test_known_good_value(self):
        u = solve_oracle_unconstrained(BETA)
        assert u.rho == pytest.approx(0.4625, abs=5e-4)

    def test_box_matches_unconstrained_when_interior(self):
        # At (4, 5) the unconstrained optimum |w| = 4 sits inside the box.
        u = solve_oracle_unconstrained(BETA)
        b = solve_oracle_box(BETA, WMAX)
        assert b.rho == pytest.approx(u.rho, abs=1e-6)
        assert not b.boundary.any()
        np.testing.assert_allclose(np.sort(b.W.ravel()), np.sort(u.W.ravel()), atol=1e-3)

    def test_box_binds_for_small_box(self):
        u = solve_oracle_unconstrained(BETA)
        b = solve_oracle_box(BETA, 2.0)
        assert b.rho < u.rho - 1e-3
        assert b.boundary.any()

    def test_exact_policy_value_matches_rho(self):
        b = solve_oracle_box(BETA, WMAX)
        assert oracle_value_exact(b.W, BETA) == pytest.approx(b.rho, abs=1e-8)


class TestReactiveFamily:
    def test_reactive_known_good(self):
        r = solve_reactive(BETA, WMAX, delay=1)
        assert r.value == pytest.approx(0.192, abs=2e-3)

    def test_stack2_at_least_prior_round(self):
        s2 = solve_stack2(BETA, WMAX, delay=1)
        assert s2.value >= 0.295 - 1e-3
        assert s2.value == pytest.approx(0.3055, abs=2e-3)  # this round's optimum

    def test_ordering_constant_reactive_stack2(self):
        c = solve_constant(BETA, WMAX)
        r = solve_reactive(BETA, WMAX, delay=1)
        s2 = solve_stack2(BETA, WMAX, delay=1)
        assert c.value <= r.value + 1e-9 <= s2.value + 1e-9

    def test_delay0_reactive_beats_delay1(self):
        r1 = solve_reactive(BETA, WMAX, delay=1)
        r0 = solve_reactive(BETA, WMAX, delay=0)
        assert r0.value > r1.value + 0.05


class TestBeliefVI:
    def test_known_good_ceiling(self):
        sol = solve_belief_vi(BETA, WMAX, delay=1, polish=False)
        assert sol.rho == pytest.approx(0.381, abs=1.5e-3)

    def test_ceiling_between_stack2_and_oracle(self):
        sol = solve_belief_vi(BETA, WMAX, delay=1, polish=False)
        s2 = solve_stack2(BETA, WMAX, delay=1)
        o = solve_oracle_box(BETA, WMAX)
        assert s2.value < sol.rho < o.rho

    def test_delay0_ceiling_higher(self):
        d1 = solve_belief_vi(BETA, WMAX, delay=1, polish=False)
        d0 = solve_belief_vi(BETA, WMAX, delay=0, polish=False)
        assert d0.rho > d1.rho + 0.01

    def test_grid_refinement_stability(self):
        lo = solve_belief_vi(BETA, WMAX, delay=1, n_grid=80, polish=False)
        hi = solve_belief_vi(BETA, WMAX, delay=1, n_grid=140, polish=False)
        assert lo.rho == pytest.approx(hi.rho, abs=2e-3)

    def test_gate_thresholds_at_operating_point(self):
        sol = solve_belief_vi(BETA, WMAX, delay=1, polish=False)
        r = solve_reactive(BETA, WMAX, delay=1)
        s2 = solve_stack2(BETA, WMAX, delay=1)
        assert (sol.rho - r.value) / sol.rho >= 0.15
        assert (sol.rho - s2.value) / sol.rho >= 0.08


class TestStateGuessAnalytic:
    def test_memoryless_delay0_is_alpha_exactly(self):
        v, m = best_memoryless(delay=0)
        assert v == pytest.approx(0.85, abs=1e-12)
        assert m == (0, 1, 2)

    def test_joint_is_a_distribution(self):
        for d in (0, 1):
            J = joint_state_token(d)
            assert J.sum() == pytest.approx(1.0, abs=1e-12)
            assert (J >= 0).all()

    def test_memoryless_delay1_below_delay0(self):
        v1, _ = best_memoryless(delay=1)
        assert 1 / 3 < v1 < 0.85


@pytest.mark.slow
class TestMonteCarloCrossChecks:
    """Solver values vs the ACTUAL environments (>= 1e6 steps for the ceiling)."""

    def test_oracle_mc(self):
        from envs.mess3.solvers.mc import mc_env_a, oracle_policy

        b = solve_oracle_box(BETA, WMAX)
        res = mc_env_a(oracle_policy(b.W), BETA, WMAX, delay=1, n_steps=1_000_000)
        assert abs(res.mean - b.rho) < 4 * res.se + 1e-3

    def test_belief_vi_mc(self):
        from envs.mess3.solvers.mc import belief_vi_policy, mc_env_a

        sol = solve_belief_vi(BETA, WMAX, delay=1, polish=False)
        res = mc_env_a(belief_vi_policy(sol), BETA, WMAX, delay=1, n_steps=1_000_000)
        # Nearest-grid lattice policy is slightly suboptimal: below rho, close.
        assert res.mean < sol.rho + 3 * res.se
        assert res.mean > sol.rho - 0.004

    def test_reactive_mc(self):
        from envs.mess3.solvers.mc import mc_env_a, table_policy

        r = solve_reactive(BETA, WMAX, delay=1)
        res = mc_env_a(table_policy(r.table), BETA, WMAX, delay=1, n_steps=1_000_000)
        assert abs(res.mean - r.value) < 4 * res.se + 1e-3

    def test_stateguess_filter_ceiling_mc(self):
        from envs.mess3.solvers.mc import argmax_belief_policy, mc_env_b
        from envs.mess3.solvers.stateguess_analytic import filter_ceiling_sim

        ceiling, se = filter_ceiling_sim(delay=1, n_steps=1_000_000)
        res = mc_env_b(argmax_belief_policy(), delay=1, n_steps=1_000_000)
        assert abs(res.mean - ceiling) < 4 * (res.se + se) + 1e-3

    def test_stateguess_memoryless_mc(self):
        from envs.mess3.solvers.mc import mc_env_b, memoryless_guess_policy

        v, m = best_memoryless(delay=1)
        res = mc_env_b(memoryless_guess_policy(m), delay=1, n_steps=1_000_000)
        assert abs(res.mean - v) < 4 * res.se + 1e-3
