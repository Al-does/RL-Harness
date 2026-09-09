# `harness/` — generic execution mechanics

The harness runs experiments without knowing their scientific content.

## Contract

An experiment module exposes `run(context)`. The harness loads that module,
constructs an immutable `RunContext`, and invokes the function.

`RunContext` may contain only shared runtime concerns:

- experiment, results, and artifacts directories;
- `seed` (default `42`) or an explicit Tune-controlled seed policy;
- unique run ID;
- smoke mode;
- resume source;
- operational hardware/resource selection.

Do not add algorithm hyperparameters, model choices, environment behavior,
analysis settings, phases, gates, or arm metadata to `RunContext`.

## Responsibilities

- thin CLI and experiment-module loading;
- runtime directory creation;
- Ray/Torch/hardware setup;
- optional Tune and direct-Algorithm helpers;
- public checkpoint and resume operations;
- compact provenance manifests;
- generic result and artifact discovery;
- cleanup and generic post-run hooks.

The harness must never import a named experiment, MESS3, or a specific
Learner/loss.

## Ray lifecycle

Prefer Tune for ordinary execution, sweeps, stopping, checkpoint retention,
failure handling, and trial metadata. Do not require Tune: `run(context)` must
also support direct RLlib, supervised, offline, or custom workflows.

### Where the training loop lives

For a Tune run, the harness constructs a `Tuner` and calls `fit()`. RLlib's
`Algorithm` is the Tune Trainable, so Tune repeatedly invokes its training
iteration; repository code does not also call `algo.train()`.

For a direct RLlib run, a generic helper in `harness/runners.py` builds the
configured Algorithm and owns the ordinary loop:

```python
algo = config.build_algo()
try:
    while True:
        result = algo.train()
        record_result(context, result)
        if should_stop(result):
            return result
finally:
    algo.stop()
```

The experiment supplies the `AlgorithmConfig` and scientific stopping
condition. The harness owns iteration mechanics, cleanup, generic result
recording, manifests, and configured checkpoints. Do not copy the ordinary
`while algo.train()` loop into every experiment.

Use documented Ray/RLlib APIs. Avoid private paths such as
`algo.learner_group._learner`. Framework subclassing is appropriate only when
configuration, callbacks, connectors, or public composition cannot express
the behavior.

## Provenance and storage

An experiment source file describes intent; a run manifest records execution.
Record experiment-repo commit/dirty state, library commit/dirty/version,
lock/framework versions, command, runtime overrides, resolved seed, run ID,
timestamps, status, and hardware.

`results/` contains compact tracked outputs:

- `run_manifest.json`, `tune_summary.json`, B2 index files;
- optional experiment summaries and study-level digests.

Per-iteration metrics are written to ignored `artifacts/metrics.jsonl`. Experiment
repos decide which compact projections (for example `training_curves.jsonl`) belong
in `results/` for Git and agents.

`artifacts/` contains ignored trial trees, checkpoints, weights, verbose metrics,
raw data, and logs. Do not partially track checkpoint directories.

Remote artifact upload to Backblaze B2 is optional. When `B2_*` environment
variables are configured, the harness uploads ignored `artifacts/` trees at
run end and records URIs in `results/`. See `docs/artifact_storage.md`.

## Prohibited generic features

- experiment phases or approval gates;
- global experiment/arm registries;
- hard-coded metric namespaces;
- environment-specific argument rewriting;
- supervised target inference from action-space shape;
- scientific CLI override dictionaries;
- old Blueprint or checkpoint compatibility shims.

## Continuing-task sampling

`harness.env_runners.ContinuingSingleAgentEnvRunner` is an explicit opt-in via
`AlgorithmConfig.env_runners(env_runner_cls=..., batch_mode="truncate_episodes")`
for single-agent tasks that never terminate or truncate. RLlib 2.56 retains
ongoing episode chunks for eventual completed-episode metrics, including large
recurrent state outputs. This runner clears only those metrics references after
every sample (also on exceptions), independent of `get_metrics()` polling.
Returned batches, active episode lookback, model/connector state, and sampling
counters are unchanged. Do not use it for finite or naturally terminating tasks:
completed-episode metrics and callbacks needing previous chunks are unsupported;
episode-count sampling, complete-episode batch mode, and observed episode endings
raise errors instead of hanging or silently reporting partial returns. Selection
is not inferred from any environment's config keys.

Verify with `uv run pytest -q tests/test_env_runners.py tests/test_architecture.py`.
The tests reproduce upstream retention, check recurrent-array reclamation across
repeated samples with/without metrics polling, and preserve standard finite-task
metrics. Recheck this private-cache workaround when upgrading the pinned Ray.
