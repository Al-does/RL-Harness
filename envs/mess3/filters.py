"""Exact Bayesian filters for the MESS3-Control program.

The filter is the probe target and the analytic yardstick, so its conventions
are pinned down here once and reused by both environments and all solvers.

Timing conventions (see env docstrings for the step order):

delay = 1
    Decision-time belief over s_t conditions on tokens through o_{t-1} and
    executed actions through w_{t-1}: measurement updates run over s_{t-1}
    (tokens through o_{t-1}, actions through w_{t-2}), then ONE prediction
    step through u_{w_{t-1}}.  At t=0 the decision belief is the initial
    state distribution (no tokens seen yet).

delay = 0
    Decision-time belief over s_t is the posterior after measuring o_t
    (which is emitted from s_t before the agent acts).

Passive mode (canonical MESS3 validation)
    Emit-from-source-before-transition convention; the belief update is the
    observable-operator form b' propto b @ T(o) with T(o) = diag(E[:, o]) @ M.
    This is exactly measure-then-predict with U = M, which is what
    ``measure`` + ``predict`` compute; a unit test pins the equivalence.
"""

from __future__ import annotations

import numpy as np


def measure(belief: np.ndarray, E: np.ndarray, token: int) -> np.ndarray:
    """Posterior over s given prior ``belief`` and an observed token."""
    p = belief * E[:, token]
    total = p.sum()
    if total <= 0.0:
        raise ValueError("measurement update produced zero mass")
    return p / total


def predict(belief: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Push a belief through one (possibly action-tilted) transition step."""
    return belief @ U


def observable_operator(E: np.ndarray, M: np.ndarray, token: int) -> np.ndarray:
    """T(o) = diag(E[:, o]) @ M  (emit-from-source-before-transition)."""
    return np.diag(E[:, token]) @ M


class ExactFilter:
    """Action-conditioned exact filter, orchestrated by the environments.

    State after ``t`` completed environment steps:
      - ``decision_belief``: belief over the current hidden state at decision
        time, per the delay convention above.
      - ``prev_posterior``: the most recent measurement posterior (over
        s_{t-1} when delay=1; over s_t pre-transition when delay=0).
    """

    def __init__(self, E: np.ndarray, delay: int, init_belief: np.ndarray):
        if delay not in (0, 1):
            raise ValueError(f"delay must be 0 or 1, got {delay}")
        self.E = np.asarray(E, dtype=np.float64)
        self.delay = delay
        self.init_belief = np.asarray(init_belief, dtype=np.float64)
        self.decision_belief: np.ndarray | None = None
        self.prev_posterior: np.ndarray | None = None

    def reset(self, first_token: int | None = None) -> np.ndarray:
        """Initialize at t=0.  ``first_token`` (o_0) is required iff delay=0."""
        self.prev_posterior = None
        if self.delay == 1:
            self.decision_belief = self.init_belief.copy()
        else:
            if first_token is None:
                raise ValueError("delay=0 filter needs o_0 at reset")
            self.decision_belief = measure(self.init_belief, self.E, first_token)
        return self.decision_belief

    def step_delay1(self, emitted_token: int, U: np.ndarray) -> np.ndarray:
        """Advance after acting: measure o_t (emitted from s_t), predict U_{w_t}."""
        post = measure(self.decision_belief, self.E, emitted_token)
        self.prev_posterior = post
        self.decision_belief = predict(post, U)
        return self.decision_belief

    def step_delay0(self, U: np.ndarray, next_token: int) -> np.ndarray:
        """Advance after acting: predict U_{w_t}, then measure o_{t+1}."""
        self.prev_posterior = self.decision_belief.copy()
        prior = predict(self.decision_belief, U)
        self.decision_belief = measure(prior, self.E, next_token)
        return self.decision_belief
