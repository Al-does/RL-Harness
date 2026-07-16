"""Study-local interior-action analysis for the operating-point sweep."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from envs.hmm import HMMEnv
from envs.mess3.solvers.belief_vi import (
    BeliefVISolution,
    PolishedBeliefPolicy,
)


@dataclass
class InteriorStats:
    beta: float
    w_max: float
    n_steps: int
    share_w0: float
    share_w1: float
    share_joint: float
    share_any: float
    mean_reward: float
    mean_reward_se: float
    tse_edges: np.ndarray = field(default=None)
    tse_all: np.ndarray = field(default=None)
    tse_interior: np.ndarray = field(default=None)
    ent_edges: np.ndarray = field(default=None)
    ent_all: np.ndarray = field(default=None)
    ent_interior: np.ndarray = field(default=None)
    beliefs_sample: np.ndarray = field(default=None)
    actions_sample: np.ndarray = field(default=None)


def interior_share(
    solution: BeliefVISolution,
    n_steps: int = 300_000,
    seed: int = 0,
    alpha: float = 0.85,
    boundary_tol_frac: float = 1e-3,
) -> InteriorStats:
    policy = PolishedBeliefPolicy(solution, alpha=alpha)
    environment = HMMEnv(
        {
            "model": {
                "factory": "envs.mess3.model:control_model",
                "kwargs": {"alpha": alpha},
            },
            "task": {
                "class": (
                    "envs.mess3.tasks.occupancy_control:"
                    "OccupancyControlTask"
                ),
                "kwargs": {
                    "transition_kl_beta": solution.beta,
                    "action_limit": solution.w_max,
                },
            },
            "delay": solution.delay,
            "episode_length": 1024,
            "diagnostics": {
                "state": True,
                "belief": True,
            },
            "seed": seed,
        }
    )
    _, info = environment.reset(seed=seed)

    tolerance = boundary_tol_frac * solution.w_max
    actions = np.empty((n_steps, 2))
    beliefs = np.empty((n_steps, 3))
    entropy = np.empty(n_steps)
    time_since_state_2 = np.full(n_steps, -1, dtype=np.int64)
    rewards = np.empty(n_steps)
    last_state_2 = None

    for step in range(n_steps):
        belief = info["belief_current"]
        action = policy(belief)
        actions[step] = action
        beliefs[step] = belief
        entropy[step] = -np.sum(
            belief * np.log(np.maximum(belief, 1e-300))
        )
        if info["state_current"] == 2:
            last_state_2 = step
        if last_state_2 is not None:
            time_since_state_2[step] = step - last_state_2
        _, reward, _, truncated, info = environment.step(action)
        rewards[step] = reward
        if truncated:
            _, info = environment.reset()

    interior = (
        np.abs(np.abs(actions) - solution.w_max) > tolerance
    )
    joint = interior.all(axis=1)
    any_interior = interior.any(axis=1)
    valid = time_since_state_2 >= 0

    n_batches = 100
    batches = rewards[
        : n_steps - n_steps % n_batches
    ].reshape(n_batches, -1).mean(axis=1)
    tse_edges = np.arange(0, 31)
    entropy_edges = np.linspace(0.0, np.log(3), 25)
    return InteriorStats(
        beta=solution.beta,
        w_max=solution.w_max,
        n_steps=n_steps,
        share_w0=float(interior[:, 0].mean()),
        share_w1=float(interior[:, 1].mean()),
        share_joint=float(joint.mean()),
        share_any=float(any_interior.mean()),
        mean_reward=float(rewards.mean()),
        mean_reward_se=float(
            batches.std(ddof=1) / np.sqrt(n_batches)
        ),
        tse_edges=tse_edges,
        tse_all=np.histogram(
            np.minimum(time_since_state_2[valid], 30),
            bins=tse_edges,
        )[0],
        tse_interior=np.histogram(
            np.minimum(
                time_since_state_2[valid & any_interior],
                30,
            ),
            bins=tse_edges,
        )[0],
        ent_edges=entropy_edges,
        ent_all=np.histogram(entropy, bins=entropy_edges)[0],
        ent_interior=np.histogram(
            entropy[any_interior],
            bins=entropy_edges,
        )[0],
        beliefs_sample=beliefs[:: max(1, n_steps // 20_000)],
        actions_sample=actions[:: max(1, n_steps // 20_000)],
    )
