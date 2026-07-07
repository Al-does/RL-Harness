"""Probe methodology (program section 4), shared by every arm and phase.

- Collect (decision-time activation, exact decision-time belief) pairs from
  rollouts of the policy under test.  Train trajectories and held-out
  trajectories come from DISJOINT seed ranges; all reported R^2 is held-out.
- GLOBAL R^2: standard held-out coefficient of determination over all belief
  components (SS_tot from the held-out mean).
- FINE-STRUCTURE R^2: branch = last ``branch_depth`` visible tokens; subtract
  the branch centroid of TRUE beliefs from predictions and targets; R^2 on
  the residuals.  Centroid-perfect-but-nothing-finer scores 0; systematic
  displacement scores negative.
- Visualization: barycentric scatter of decoded beliefs next to ground truth.

Policy modes:
  "policy"  — actions sampled from the module's own action distribution,
              reproducing RLlib's rollout convention (Gaussian sample in
              normalized space, linearly unsquashed to the Box; categorical
              sample for Discrete).
  "random"  — uniform action-space sampling (A-pred, passive validation,
              untrained-floor probes at initialization).
  "greedy"  — deterministic actions (distribution mean / argmax), matching
              the convention of the analytic ceilings; used for reward
              evaluation, not for probe fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------


@dataclass
class ProbeData:
    activations: np.ndarray  # (N, d)
    beliefs: np.ndarray      # (N, 3) exact decision-time beliefs
    tokens: np.ndarray       # (N,) visible token at decision time (-1 = none)
    prev_tokens: np.ndarray  # (N,) visible token one step earlier (-1 = none)
    states: np.ndarray       # (N,) true hidden state
    actions: np.ndarray      # (N, act_dim) executed env actions
    rewards: np.ndarray      # (N,)


def _init_state(module, batch: int, device):
    init = module.get_initial_state()
    return {
        k: torch.from_numpy(v).unsqueeze(0).repeat(batch, *([1] * v.ndim)).to(device)
        for k, v in init.items()
    }


@torch.no_grad()
def collect_probe_data(
    module,
    env_factory,
    *,
    n_steps: int,
    seed: int,
    policy_mode: str = "policy",
    n_envs: int = 16,
    device: str | torch.device = "cpu",
    warmup: int = 64,
) -> ProbeData:
    """Roll out ``n_envs`` parallel episodes until >= n_steps pairs collected.

    ``warmup`` initial steps of each episode are dropped (transient before the
    closed-loop attractor; also avoids the empty-token cold start dominating
    branch statistics).
    """
    device = torch.device(device)
    module = module.to(device).eval()
    rng = np.random.default_rng(seed)
    envs = [env_factory() for _ in range(n_envs)]
    obs = []
    for e in envs:
        o, info = e.reset(seed=int(rng.integers(2**31 - 1)))
        obs.append(o)
    is_stateful = module.is_stateful()
    state = _init_state(module, n_envs, device) if is_stateful else None

    acts, bels, toks, ptoks, sts, actions_log, rews = [], [], [], [], [], [], []
    prev_tok = [-1] * n_envs
    t_in_ep = np.zeros(n_envs, dtype=int)

    discrete = not hasattr(module, "_pi_mean")
    if not discrete:
        low = envs[0].action_space.low
        high = envs[0].action_space.high

    total = 0
    while total < n_steps:
        obs_t = torch.from_numpy(np.stack(obs)).float().to(device)
        if is_stateful:
            emb, state = module.encode_step(obs_t, state)
        else:
            emb, _ = module.encode_step(obs_t)

        if policy_mode == "random":
            env_actions = [e.action_space.sample() for e in envs]
        else:
            if discrete:
                logits = module._pi_logits(emb)
                if policy_mode == "greedy":
                    env_actions = logits.argmax(-1).cpu().numpy()
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    env_actions = dist.sample().cpu().numpy()
            else:
                mean = module._pi_mean(emb)
                if policy_mode == "greedy":
                    a = mean.cpu().numpy()
                else:
                    std = module._log_std.exp().expand_as(mean)
                    a = torch.normal(mean, std).cpu().numpy()
                # RLlib normalize_actions convention: linear unsquash then clip.
                env_actions = np.clip(low + (a + 1.0) * (high - low) / 2.0, low, high)

        emb_np = emb.cpu().numpy()
        for i, e in enumerate(envs):
            # Record the DECISION-TIME tuple before stepping: belief over s_t,
            # visible token, and the true s_t itself (post-step info["state"]
            # would be s_{t+1} — misaligned with the belief).
            info_belief = e._filter.decision_belief.copy()
            tok = e._obs_token if e._obs_token is not None else -1
            state_t = e._s
            o2, r, term, trunc, info = e.step(env_actions[i])
            if t_in_ep[i] >= warmup:
                acts.append(emb_np[i])
                bels.append(info_belief)
                toks.append(tok)
                ptoks.append(prev_tok[i])
                sts.append(state_t)
                actions_log.append(np.atleast_1d(np.asarray(env_actions[i], dtype=np.float64)))
                rews.append(r)
                total += 1
            prev_tok[i] = tok
            t_in_ep[i] += 1
            if term or trunc:
                o2, _ = e.reset(seed=int(rng.integers(2**31 - 1)))
                prev_tok[i] = -1
                t_in_ep[i] = 0
                if is_stateful:
                    fresh = _init_state(module, 1, device)
                    for k in state:
                        state[k][i] = fresh[k][0]
            obs[i] = o2

    return ProbeData(
        activations=np.asarray(acts, dtype=np.float64),
        beliefs=np.asarray(bels, dtype=np.float64),
        tokens=np.asarray(toks, dtype=np.int64),
        prev_tokens=np.asarray(ptoks, dtype=np.int64),
        states=np.asarray(sts, dtype=np.int64),
        actions=np.asarray(actions_log, dtype=np.float64),
        rewards=np.asarray(rews, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Affine probe + metrics
# ---------------------------------------------------------------------------


def fit_affine_probe(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-6):
    """Least-squares affine map X -> Y.  Returns (W, b)."""
    Xa = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    A = Xa.T @ Xa + ridge * np.eye(Xa.shape[1])
    B = Xa.T @ Y
    coef = np.linalg.solve(A, B)
    return coef[:-1], coef[-1]


def probe_predict(W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ W + b


def r2_global(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def branch_keys(data: ProbeData, depth: int = 2) -> np.ndarray:
    """Branch id from the last ``depth`` visible tokens (base-4 code, -1 -> 3)."""
    t0 = np.where(data.tokens < 0, 3, data.tokens)
    if depth == 1:
        return t0
    t1 = np.where(data.prev_tokens < 0, 3, data.prev_tokens)
    return t0 * 4 + t1


def r2_fine(
    pred: np.ndarray, true: np.ndarray, branches: np.ndarray, min_branch: int = 50
) -> float:
    """Fine-structure R^2: residuals w.r.t. branch centroids of TRUE beliefs."""
    pred_r = np.empty_like(pred)
    true_r = np.empty_like(true)
    keep = np.zeros(len(true), dtype=bool)
    for br in np.unique(branches):
        m = branches == br
        if m.sum() < min_branch:
            continue
        c = true[m].mean(axis=0)
        pred_r[m] = pred[m] - c
        true_r[m] = true[m] - c
        keep[m] = True
    ss_res = float(((pred_r[keep] - true_r[keep]) ** 2).sum())
    ss_tot = float((true_r[keep] ** 2).sum())
    return 1.0 - ss_res / ss_tot


def evaluate_probe(
    train: ProbeData, test: ProbeData, branch_depth: int = 2
) -> dict:
    """Fit on train, report held-out global and fine R^2 (+ per-depth fine)."""
    W, b = fit_affine_probe(train.activations, train.beliefs)
    pred = probe_predict(W, b, test.activations)
    out = {
        "r2_global": r2_global(pred, test.beliefs),
        "r2_fine_depth1": r2_fine(pred, test.beliefs, branch_keys(test, 1)),
        "r2_fine_depth2": r2_fine(pred, test.beliefs, branch_keys(test, 2)),
        "n_train": len(train.beliefs),
        "n_test": len(test.beliefs),
        "probe": (W, b),
    }
    out["r2_fine"] = out[f"r2_fine_depth{branch_depth}"]
    return out


def within_branch_action_variance_fraction(data: ProbeData, depth: int = 2) -> float:
    """Fraction of total action variance that lives WITHIN branches."""
    br = branch_keys(data, depth)
    a = data.actions
    total = ((a - a.mean(axis=0)) ** 2).sum()
    within = 0.0
    for k in np.unique(br):
        m = br == k
        within += ((a[m] - a[m].mean(axis=0)) ** 2).sum()
    return float(within / total) if total > 0 else float("nan")
