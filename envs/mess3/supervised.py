"""Supervised trainer for the non-RL arms (B-SL, A-pred, passive validation).

These arms train the IDENTICAL transformer core with a cross-entropy loss on
rollouts collected under RANDOM actions (for B-SL the guesses do not affect
the dynamics, so the data distribution matches the RL arms; for A-pred random
tilts are part of the arm definition).

Targets:
  - "state":      true hidden state s_t at decision time t   (B-SL)
  - "next_token": the next VISIBLE token, i.e. the token slot of the
                  observation at t+1                          (A-pred, passive)

Training runs on full episodes (the banded transformer handles length-1024
sequences directly; band semantics identical to the RL path), on MPS when
available.  Log-spaced checkpoints (powers of 2 in optimizer steps, plus
step 0) are saved as light ``module_state`` .pt files — the same format the
RL launcher writes — so the probe/N-init pipeline treats every arm uniformly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from losses.next_token import next_token_targets

N_TOKENS = 3


def rollout_episodes(env_factory, n_episodes: int, seed: int, policy: str = "random"):
    """Random-action episodes: (obs (N, L, D), states (N, L), rng-consistent)."""
    rng = np.random.default_rng(seed)
    obs_list, state_list = [], []
    env = env_factory()
    for _ in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(2**31 - 1)))
        O, S = [obs], [info["state"]]
        done = False
        while not done:
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            if not done:
                O.append(obs)
                S.append(info["state"])
        obs_list.append(np.stack(O))
        state_list.append(np.array(S, dtype=np.int64))
    return np.stack(obs_list), np.stack(state_list)


def make_targets(obs: torch.Tensor, states: torch.Tensor, target: str):
    """(targets (B, T'), valid (B, T'), logits_slice) per target type."""
    if target == "state":
        return states, torch.ones_like(states, dtype=torch.bool), slice(None)
    # next_token: token slot of obs at t+1; valid where populated.
    mask = torch.ones(obs.shape[:2], dtype=torch.bool, device=obs.device)
    targets, valid = next_token_targets(
        obs, mask, num_token_classes=N_TOKENS
    )
    return targets, valid, slice(None, -1)


def train_supervised(
    *,
    env_factory,
    model_class,
    model_config: dict,
    obs_dim: int,
    action_space,
    target: str,
    total_steps: int,
    outdir: Path,
    seed: int,
    batch_episodes: int = 8,
    lr: float = 3e-4,
    fresh_data_episodes: int = 512,
    log_every: int = 25,
):
    """Streaming supervised training; returns the trained module (on CPU)."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed)
    import gymnasium as gym

    module = model_class(
        observation_space=gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
        action_space=action_space,
        model_config=model_config,
    ).to(device)
    if not hasattr(module, "encode_chunks"):
        raise TypeError(
            f"{model_class.__name__} does not support supervised sequence encoding"
        )
    head_name = {
        "state": "state_aux_head",
        "next_token": "next_token_aux_head",
    }.get(target)
    if head_name is None:
        raise ValueError(f"unsupported supervised target {target!r}")
    if not hasattr(module, head_name):
        raise TypeError(
            f"{model_class.__name__} does not provide {head_name}"
        )
    auxiliary_head = getattr(module, head_name)
    opt = torch.optim.Adam(module.parameters(), lr=lr)

    outdir.mkdir(parents=True, exist_ok=True)
    log = open(outdir / "progress.jsonl", "a")

    def save(tag: str, env_steps: int):
        torch.save(
            {"state_dict": {k: v.cpu() for k, v in module.state_dict().items()},
             "env_steps": env_steps},
            outdir / f"module_state_{tag}.pt",
        )

    save("00000000", 0)  # N-init checkpoint

    env_steps = 0
    opt_step = 0
    next_ckpt = 1
    data_seed = seed * 1000
    t0 = time.time()
    while env_steps < total_steps:
        data_seed += 1
        obs_np, st_np = rollout_episodes(env_factory, fresh_data_episodes, data_seed)
        obs_all = torch.from_numpy(obs_np).float()
        st_all = torch.from_numpy(st_np)
        n_ep, ep_len = obs_all.shape[0], obs_all.shape[1]
        perm = torch.randperm(n_ep)
        for i in range(0, n_ep, batch_episodes):
            idx = perm[i : i + batch_episodes]
            obs = obs_all[idx].to(device)
            states = st_all[idx].to(device)
            B, L, _ = obs.shape
            ctx = torch.zeros(
                B, module.sequence_lookback, obs.shape[-1], device=device
            )
            lens = torch.zeros(B, device=device)
            emb = module.encode_chunks(ctx, lens, obs)
            logits = auxiliary_head(emb)
            targets, valid, sl = make_targets(obs, states, target)
            logits = logits[:, sl, :]
            loss = torch.nn.functional.cross_entropy(logits[valid], targets[valid])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            opt_step += 1
            env_steps += B * L
            if opt_step % log_every == 0:
                with torch.no_grad():
                    acc = (logits[valid].argmax(-1) == targets[valid]).float().mean()
                rec = {
                    "opt_step": opt_step,
                    "env_steps": env_steps,
                    "ce": loss.detach().item(),
                    "accuracy": acc.item(),
                    "wall_s": round(time.time() - t0, 1),
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)
            if opt_step >= next_ckpt:
                save(f"{env_steps:08d}", env_steps)
                next_ckpt *= 2
            if env_steps >= total_steps:
                break
    save("final", env_steps)
    log.close()
    return module.cpu()
