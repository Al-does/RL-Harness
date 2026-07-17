"""MESS3 representation and target adapters for generic affine probes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from analysis.probes import (
    conditional_residual_r2,
    fit_affine_probe,
    predictive_belief_update,
    probe_predict,
    r2_score,
)
from analysis.rollouts import collect_batched_rollout_data


ActionOutcomeOperator = Callable[[Mapping[str, Any]], np.ndarray]


@dataclass(frozen=True, slots=True)
class ProbeData:
    activations: np.ndarray
    beliefs: np.ndarray
    diagnostic_beliefs: np.ndarray
    tokens: np.ndarray
    previous_tokens: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray


def _initial_state(module: Any, batch_size: int, device: torch.device):
    state = module.get_initial_state()
    return {
        key: torch.from_numpy(value)
        .unsqueeze(0)
        .repeat(batch_size, *([1] * value.ndim))
        .to(device)
        for key, value in state.items()
    }


def make_io_moore_operator(
    outcome_likelihood: np.ndarray,
) -> ActionOutcomeOperator:
    """Build ``K(y|a) = diag(P(y|s)) U(a)`` from public step diagnostics."""

    likelihood = np.array(outcome_likelihood, dtype=np.float64, copy=True)
    if likelihood.ndim != 2:
        raise ValueError("outcome_likelihood must be two-dimensional")
    likelihood.setflags(write=False)

    def action_outcome_operator(info: Mapping[str, Any]) -> np.ndarray:
        outcome = info.get("visible_token_current")
        if outcome is None:
            raise ValueError("a transducer update requires a visible outcome")
        try:
            transition = np.asarray(
                info["executed_transition_matrix"],
                dtype=np.float64,
            )
        except KeyError as error:
            raise KeyError(
                "transducer probing requires transition diagnostics"
            ) from error
        return np.diag(likelihood[:, int(outcome)]) @ transition

    return action_outcome_operator


@torch.no_grad()
def collect_probe_data(
    module: Any,
    env_factory,
    *,
    n_steps: int,
    seed: int,
    policy_mode: str = "policy",
    n_envs: int = 16,
    device: str | torch.device = "cpu",
    warmup: int = 64,
    initial_belief: np.ndarray | None = None,
    action_outcome_operator: ActionOutcomeOperator | None = None,
) -> ProbeData:
    """Collect aligned activations and public MESS3 diagnostics."""
    if policy_mode not in {"policy", "random", "greedy"}:
        raise ValueError(f"unsupported policy mode {policy_mode!r}")
    if (initial_belief is None) != (action_outcome_operator is None):
        raise ValueError(
            "initial_belief and action_outcome_operator must be provided together"
        )
    device = torch.device(device)
    module = module.to(device).eval()
    stateful = module.is_stateful()
    discrete = module.heads.is_discrete
    previous_tokens = np.full(n_envs, -1, dtype=np.int64)
    transducer_beliefs = (
        None
        if initial_belief is None
        else np.repeat(
            np.asarray(initial_belief, dtype=np.float64)[None, :],
            n_envs,
            axis=0,
        )
    )

    def initial_state(batch_size: int):
        return _initial_state(module, batch_size, device)

    def reset_state(state, indices: np.ndarray):
        fresh = _initial_state(module, len(indices), device)
        index_tensor = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=device,
        )
        for key, value in state.items():
            value.index_copy_(0, index_tensor, fresh[key])
        return state

    def step_adapter(observations, state, rng, action_spaces):
        del rng
        observation_tensor = torch.from_numpy(observations).float().to(device)
        if stateful:
            embedding, state = module.encode_step(
                observation_tensor,
                state,
            )
        else:
            embedding, _ = module.encode_step(observation_tensor)

        if policy_mode == "random":
            env_actions = [
                action_space.sample() for action_space in action_spaces
            ]
        elif discrete:
            logits = module.action_distribution_inputs(embedding)
            if policy_mode == "greedy":
                env_actions = logits.argmax(dim=-1).cpu().numpy()
            else:
                env_actions = (
                    torch.distributions.Categorical(logits=logits)
                    .sample()
                    .cpu()
                    .numpy()
                )
        else:
            mean, standard_deviation = module.heads.policy_mean_and_std(
                embedding
            )
            normalized = (
                mean
                if policy_mode == "greedy"
                else torch.normal(mean, standard_deviation)
            ).cpu().numpy()
            action_low = action_spaces[0].low
            action_high = action_spaces[0].high
            env_actions = np.clip(
                action_low
                + (normalized + 1.0)
                * (action_high - action_low)
                / 2.0,
                action_low,
                action_high,
            )
        return env_actions, state, embedding.cpu().numpy()

    def target_adapter(observations, infos, episode_steps):
        del observations
        diagnostic_beliefs = np.stack(
            [info["belief_current"] for info in infos]
        )
        if transducer_beliefs is None:
            beliefs = diagnostic_beliefs
        else:
            assert initial_belief is not None
            assert action_outcome_operator is not None
            for index, (info, episode_step) in enumerate(
                zip(infos, episode_steps)
            ):
                if episode_step == 0:
                    transducer_beliefs[index] = initial_belief
                else:
                    transducer_beliefs[index] = predictive_belief_update(
                        transducer_beliefs[index],
                        action_outcome_operator(info),
                    )
            beliefs = transducer_beliefs.copy()
        tokens = np.asarray(
            [
                -1
                if info.get("visible_token_current") is None
                else int(info["visible_token_current"])
                for info in infos
            ],
            dtype=np.int64,
        )
        previous = np.where(episode_steps == 0, -1, previous_tokens)
        targets = {
            "beliefs": beliefs,
            "diagnostic_beliefs": diagnostic_beliefs,
            "tokens": tokens,
            "previous_tokens": previous,
            "states": np.asarray(
                [info["state_current"] for info in infos],
                dtype=np.int64,
            ),
        }
        previous_tokens[:] = tokens
        return targets

    collected = collect_batched_rollout_data(
        env_factory,
        step_adapter,
        target_adapter,
        n_steps=n_steps,
        seed=seed,
        n_envs=n_envs,
        initial_state=initial_state if stateful else None,
        reset_state=reset_state if stateful else None,
        warmup=warmup,
    )

    return ProbeData(
        activations=np.asarray(
            collected.representations,
            dtype=np.float64,
        ),
        beliefs=np.asarray(
            collected.targets["beliefs"],
            dtype=np.float64,
        ),
        diagnostic_beliefs=np.asarray(
            collected.targets["diagnostic_beliefs"],
            dtype=np.float64,
        ),
        tokens=np.asarray(
            collected.targets["tokens"],
            dtype=np.int64,
        ),
        previous_tokens=np.asarray(
            collected.targets["previous_tokens"],
            dtype=np.int64,
        ),
        states=np.asarray(
            collected.targets["states"],
            dtype=np.int64,
        ),
        actions=np.asarray(collected.actions, dtype=np.float64),
        rewards=np.asarray(collected.rewards, dtype=np.float64),
    )


def branch_keys(data: ProbeData, depth: int = 2) -> np.ndarray:
    """Encode one or two most recent visible MESS3 tokens."""
    current = np.where(data.tokens < 0, 3, data.tokens)
    if depth == 1:
        return current
    if depth != 2:
        raise ValueError("MESS3 branch depth must be one or two")
    previous = np.where(
        data.previous_tokens < 0,
        3,
        data.previous_tokens,
    )
    return current * 4 + previous


def evaluate_probe(
    train: ProbeData,
    test: ProbeData,
    *,
    branch_depth: int = 2,
) -> dict[str, Any]:
    """Fit on one seed range and evaluate on a disjoint range."""
    weight, bias = fit_affine_probe(
        train.activations,
        train.beliefs,
    )
    predicted = probe_predict(weight, bias, test.activations)
    result = {
        "r2_global": r2_score(predicted, test.beliefs),
        "r2_fine_depth1": conditional_residual_r2(
            predicted,
            test.beliefs,
            branch_keys(test, 1),
            min_group_size=50,
        ),
        "r2_fine_depth2": conditional_residual_r2(
            predicted,
            test.beliefs,
            branch_keys(test, 2),
            min_group_size=50,
        ),
        "n_train": len(train.beliefs),
        "n_test": len(test.beliefs),
        "probe": (weight, bias),
    }
    result["r2_fine"] = result[f"r2_fine_depth{branch_depth}"]
    return result


def within_branch_action_variance_fraction(
    data: ProbeData,
    *,
    depth: int = 2,
) -> float:
    branches = branch_keys(data, depth)
    actions = data.actions
    total = np.square(actions - actions.mean(axis=0)).sum()
    within = 0.0
    for branch in np.unique(branches):
        members = branches == branch
        within += np.square(
            actions[members] - actions[members].mean(axis=0)
        ).sum()
    return float(within / total) if total > 0 else float("nan")
