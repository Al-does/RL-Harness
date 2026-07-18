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

`results/` contains compact tracked outputs. `artifacts/` contains ignored
trial trees, checkpoints, weights, raw data, and logs. Do not partially track
checkpoint directories.

Remote artifact upload is deferred. Do not imply durability for artifacts on
an ephemeral machine.

## Prohibited generic features

- experiment phases or approval gates;
- global experiment/arm registries;
- hard-coded metric namespaces;
- environment-specific argument rewriting;
- supervised target inference from action-space shape;
- scientific CLI override dictionaries;
- old Blueprint or checkpoint compatibility shims.
