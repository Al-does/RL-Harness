"""Uniform checkpoint access for every arm (RL and supervised paths).

A run directory (results/phaseK/<arm>/seed<S>/) contains:
  module_state_<envsteps:08d>.pt  log-spaced checkpoints, including 00000000
  module_state_final.pt
  blueprint.json                  arm + seed provenance
  progress.jsonl                  training metrics

``load_module`` reconstructs the exact RLModule (architecture from the
blueprint) and loads a checkpoint's weights.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


def load_blueprint_dict(run_dir: Path) -> dict:
    with open(run_dir / "blueprint.json") as f:
        return json.load(f)


def env_factory_from_blueprint(bp: dict):
    import importlib

    mod, cls = bp["env_entry"].split(":")
    env_cls = getattr(importlib.import_module(mod), cls)
    kw = dict(bp["env_kwargs"])
    if bp.get("scramble_tokens") in (True, "True"):
        kw["scramble_tokens"] = True
    return lambda: env_cls(dict(kw))


def module_from_blueprint(bp: dict):
    from envs.mess3.rlmodules import Mess3MLPRLModule, Mess3TransformerRLModule

    env = env_factory_from_blueprint(bp)()
    m = bp["model"]
    if m["kind"] == "transformer":
        cls = Mess3TransformerRLModule
        model_config = {
            "d_model": int(m["d_model"]),
            "n_layers": int(m["n_layers"]),
            "n_heads": int(m["n_heads"]),
            "context_len": int(m["context_len"]),
            "max_seq_len": 32,
        }
    else:
        cls = Mess3MLPRLModule
        hidden = m["mlp_hidden"]
        if isinstance(hidden, str):
            hidden = [int(x) for x in re.findall(r"\d+", hidden)]
        model_config = {"mlp_hidden": tuple(hidden)}
    obs_space = gym.spaces.Box(-np.inf, np.inf, env.observation_space.shape, np.float32)
    return cls(
        observation_space=obs_space,
        action_space=env.action_space,
        model_config=model_config,
    )


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Sorted (env_steps, path); 'final' resolves to its stored env_steps."""
    out = []
    for p in sorted(Path(run_dir).glob("module_state_*.pt")):
        payload = torch.load(p, map_location="cpu", weights_only=True)
        out.append((int(payload["env_steps"]), p))
    # Deduplicate identical step counts (final may coincide with the last log ckpt).
    seen, uniq = set(), []
    for steps, p in sorted(out):
        if steps in seen and "final" in p.name:
            continue
        seen.add(steps)
        uniq.append((steps, p))
    return uniq


def load_module(run_dir: Path, ckpt_path: Path):
    bp = load_blueprint_dict(run_dir)
    module = module_from_blueprint(bp)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    module.load_state_dict(payload["state_dict"])
    module.eval()
    return module


def read_progress(run_dir: Path) -> list[dict]:
    p = Path(run_dir) / "progress.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
