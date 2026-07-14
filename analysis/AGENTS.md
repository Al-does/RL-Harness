# `analysis/` — reusable analysis tools

This folder contains domain-agnostic operations for consuming runs. Experiment
files supply model- and task-specific adapters.

## Appropriate contents

- run, result, metric, and checkpoint discovery;
- public RLlib/Tune checkpoint loading;
- generic rollout collection with injected adapters;
- affine/linear probes and train/test splitting;
- generic metrics such as R² and conditional residual R²;
- aggregation and reusable plotting primitives.

Do not place named experiment pipelines, thresholds, arm lists, phase reports,
or MESS3-specific semantics here.

## Probe boundaries

A reusable probe separates:

1. representation extraction from a model;
2. target extraction from environment data;
3. fitting;
4. evaluation metrics;
5. experiment-specific reporting.

Initially accept ordinary callables for representation and target extraction.
For example, an experiment may call `module.encode_step(...)`. Do not add a
formal model-representation protocol until real incompatible consumers require
one.

Compute activations on demand from a loaded model. Do not retain them in
rollout or replay storage by default.

Network representations and environment latent state are separate contracts.
The latter must come from public environment diagnostics or `info`, never
private fields such as `_s`, `_filter`, or `_obs_token`.

## Artifact boundary

Analysis should consume a small generic run/artifact interface exposing:

- manifest and resolved trial config;
- metric records;
- checkpoint discovery/loading;
- results and artifacts paths.

Do not reconstruct modules from obsolete Blueprint JSON. Do not hard-code
`results/phaseK`, metric namespaces, model class paths, or one checkpoint file
format.

## Current migration guidance

- Keep the affine fitting and broadly applicable metric math from
  `probe.py`.
- Parameterize rollout representation/target extraction.
- Move MESS3 belief branches, state access, and policy diagnostics to MESS3
  domain helpers or experiment leaves.
- Replace `checkpoints.py` Blueprint reconstruction with public RLlib/Tune
  artifact access.
- Move phase findings, null brackets, scrambled-input evaluation, and complete
  probe-report workflows into experiment leaves.
- Merge duplicate one-off drivers instead of growing parallel script copies.

Analysis-time CPU/NumPy conversion is acceptable where the computation is
genuinely offline. Never move such conversions into model or Learner hot paths.
