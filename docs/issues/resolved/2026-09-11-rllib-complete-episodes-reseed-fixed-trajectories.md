---
status: resolved
severity: high
area: harness/runners + RLlib SingleAgentEnvRunner
discovered: 2026-09-11
reproduction: confirmed
---

# PPO complete-episodes sampling replays the same seeded trajectories

## Context

- **Git revision / worktree:** `9a5fed83b71bdf3d63a1784556b9c9743b3bca7e`; clean before recording
- **Command:** see minimal reproduction below
- **Environment:** Python 3.14 via `uv`; Ray/RLlib `2.56.0`; Gymnasium `1.2.2`
- **Training context:** experiment `experiments.nonergodic_mess3_token_guess_cycle_1.ppo.experiment`; seed `42`; smoke and full; hardware profiles `cpu` and `rtx4090`
- **Related records:** `experiments/nonergodic_mess3_token_guess_cycle_1/ppo/results/20260911T003313Z-d1b8abe3/run_manifest.json`

## Expected behavior

With a fixed experiment seed, training should be reproducible while still
sampling fresh stochastic HMM episodes after the initial seeded reset. PPO
episode-return telemetry should estimate performance on newly sampled training
episodes, not repeated copies of the same finite trajectories.

## Observed behavior

RLlib's new `SingleAgentEnvRunner.sample()` calls `_sample(num_episodes=self.num_envs)`
when `batch_mode="complete_episodes"`. `_sample()` resets all vector envs when
`num_episodes is not None` and sets `_needs_initial_reset = True`, causing the
next call to reset with the same worker seed again. Gymnasium vector reset then
uses `[seed, seed + 1, ...]`, so every training iteration replays the same vector
env trajectories for each worker.

For the nonergodic MESS3 run, the historical config used 16 EnvRunners and 24
envs per runner, so each iteration trained on the same 384 seeded 127-step
episodes. Reported return rose on these fixed trajectories, peaking at 98.25/127
and maxing at 118/127, while independent held-out checkpoint evaluation stayed
near chance and ended at 32.55%.

The historical MESS3 run used complete episodes and reproduced exact sampled
episode reward sums, so the archived curve is not explained by compact-curve
postprocessing or by cross-episode reward addition.

## Minimal reproduction

From the experiment checkout with the editable sibling harness:

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run python - <<'PY'
from pathlib import Path
import hashlib, types
import numpy as np
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from experiments.nonergodic_mess3_token_guess_cycle_1.shared import build_config
from harness.context import RunContext
from harness.hardware import PROFILES

root = Path('/home/ubuntu/nonergodic-investigation/trace-training-episodes')
ctx = RunContext(
    experiment_dir=root,
    results_dir=root / 'results',
    artifacts_dir=root / 'artifacts',
    seed=42,
    smoke=True,
    hardware=PROFILES['cpu'],
)
config = build_config(ctx)
env_config = dict(config.env_config)
env_config['diagnostics'] = {'tokens': True, 'transitions': True}
config = config.environment(config.env, env_config=env_config).env_runners(
    env_runner_cls=SingleAgentEnvRunner,
    num_env_runners=0,
    num_envs_per_env_runner=4,
    rollout_fragment_length='auto',
    batch_mode='complete_episodes',
)
algo = config.build_algo()
runner = algo.env_runner_group.local_env_runner
sample_records = []
original = runner.sample

def wrapped(self, *args, **kwargs):
    episodes = original(*args, **kwargs)
    sample_records.append([
        hashlib.sha256(bytes(
            info['raw_token_before']
            for info in episode.get_infos()
            if 'raw_token_before' in info
        )).hexdigest()[:12]
        for episode in episodes
    ])
    return episodes

runner.sample = types.MethodType(wrapped, runner)
seen = set()
try:
    for iteration in range(1, 4):
        before = len(sample_records)
        result = algo.train()
        hashes = [h for sample in sample_records[before:] for h in sample]
        repeated = sorted(set(hashes) & seen)
        seen.update(hashes)
        print(iteration, result['env_runners']['episode_return_mean'], hashes, repeated)
finally:
    algo.stop()
PY
```

Observed hashes repeat across iterations; in local smoke with four vector envs,
the same four hashes recur while the reported return increases.

A second reproduction with four remote EnvRunners and four envs per runner
showed the same phenomenon at larger scale:

```text
1 mean 41.75 min 35 max 52
...
9 mean 80.00 min 54 max 93
11 mean 86.94 min 66 max 97
12 mean 57.31 min 40 max 63
```

Independent held-out rollouts from the same local checkpoints remained near
chance while RLlib training returns increased.

## Cause and scope

The immediate cause is the interaction between RLlib new-stack complete-episode
sampling, fixed worker seeds, and finite stochastic environments. This can make
PPO memorize a finite set of seeded training episodes and makes training
telemetry incomparable to held-out performance.

The scope includes seeded RLlib new-stack recipes using complete episodes with
stochastic finite environments. Continuing tasks using the custom
`ContinuingSingleAgentEnvRunner` are outside this exact failure mode.

## Resolution

Finite stochastic recipes that require complete episodes can select
`FreshEpisodeSingleAgentEnvRunner`. It applies the deterministic worker seed
only to the first vector reset after environment construction. Later sample
calls reset with `seed=None`, preserving reproducibility across runs while
advancing each environment's RNG stream instead of replaying trajectories.

The workaround uses RLlib 2.56's private reset hook and must be reviewed when
the pinned Ray version changes.

## Resolution history

- 2026-09-11 — Recorded after reproducing fixed trajectory hashes, exact reward
  recomputation, and training/held-out mismatch for nonergodic MESS3 PPO.
- 2026-09-11 — Added `FreshEpisodeSingleAgentEnvRunner`, which applies the
  deterministic worker seed only on the first vector reset after environment
  construction and uses the continuing RNG stream for later complete-episode
  sample calls. Regression coverage verifies that separate seeded runners
  reproduce the same sequence while successive sample calls do not replay it.
