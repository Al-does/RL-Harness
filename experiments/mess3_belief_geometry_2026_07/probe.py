"""MESS3 representation and target adapters for generic affine probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from analysis.probes import (
    conditional_residual_r2,
    fit_affine_probe,
    probe_predict,
    r2_score,
)


@dataclass(frozen=True, slots=True)
class ProbeData:
    activations: np.ndarray
    beliefs: np.ndarray
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
) -> ProbeData:
    """Collect aligned activations and public MESS3 diagnostics."""
    if policy_mode not in {"policy", "random", "greedy"}:
        raise ValueError(f"unsupported policy mode {policy_mode!r}")
    device = torch.device(device)
    module = module.to(device).eval()
    rng = np.random.default_rng(seed)
    envs = [env_factory() for _ in range(n_envs)]
    observations, infos = [], []
    for env in envs:
        episode_seed = int(rng.integers(2**31 - 1))
        env.action_space.seed(episode_seed)
        observation, info = env.reset(seed=episode_seed)
        observations.append(observation)
        infos.append(info)

    stateful = module.is_stateful()
    state = (
        _initial_state(module, n_envs, device) if stateful else None
    )
    activations, beliefs, tokens, previous_tokens = [], [], [], []
    states, action_records, rewards = [], [], []
    previous_token = [-1] * n_envs
    episode_step = np.zeros(n_envs, dtype=int)
    discrete = module.heads.is_discrete
    if not discrete:
        action_low = envs[0].action_space.low
        action_high = envs[0].action_space.high

    try:
        while len(activations) < n_steps:
            observation_tensor = torch.from_numpy(
                np.stack(observations)
            ).float().to(device)
            if stateful:
                embedding, state = module.encode_step(
                    observation_tensor,
                    state,
                )
            else:
                embedding, _ = module.encode_step(observation_tensor)

            if policy_mode == "random":
                env_actions = [
                    env.action_space.sample() for env in envs
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
                env_actions = np.clip(
                    action_low
                    + (normalized + 1.0)
                    * (action_high - action_low)
                    / 2.0,
                    action_low,
                    action_high,
                )

            embedding_array = embedding.cpu().numpy()
            for index, env in enumerate(envs):
                decision_info = infos[index]
                token = decision_info.get("obs_token")
                token = -1 if token is None else int(token)
                next_observation, reward, terminated, truncated, next_info = (
                    env.step(env_actions[index])
                )
                if episode_step[index] >= warmup:
                    activations.append(embedding_array[index])
                    beliefs.append(decision_info["belief"])
                    tokens.append(token)
                    previous_tokens.append(previous_token[index])
                    states.append(decision_info["state"])
                    action_records.append(
                        np.atleast_1d(
                            np.asarray(
                                env_actions[index],
                                dtype=np.float64,
                            )
                        )
                    )
                    rewards.append(reward)
                previous_token[index] = token
                episode_step[index] += 1

                if terminated or truncated:
                    episode_seed = int(rng.integers(2**31 - 1))
                    env.action_space.seed(episode_seed)
                    next_observation, next_info = env.reset(
                        seed=episode_seed
                    )
                    previous_token[index] = -1
                    episode_step[index] = 0
                    if stateful:
                        fresh = _initial_state(module, 1, device)
                        for key in state:
                            state[key][index] = fresh[key][0]
                observations[index] = next_observation
                infos[index] = next_info
    finally:
        for env in envs:
            env.close()

    return ProbeData(
        activations=np.asarray(activations[:n_steps], dtype=np.float64),
        beliefs=np.asarray(beliefs[:n_steps], dtype=np.float64),
        tokens=np.asarray(tokens[:n_steps], dtype=np.int64),
        previous_tokens=np.asarray(
            previous_tokens[:n_steps],
            dtype=np.int64,
        ),
        states=np.asarray(states[:n_steps], dtype=np.int64),
        actions=np.asarray(action_records[:n_steps], dtype=np.float64),
        rewards=np.asarray(rewards[:n_steps], dtype=np.float64),
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
