# Coding Agent Prompt: Generic RL Harness Refactor

You are joining a substantial refactor of an RLlib research repository. The
current repository was originally built around a specific MESS3 research
program. The goal is to turn it into a generic RL experimentation harness while
preserving MESS3 as a collection of independent example experiments.

Repository root:

`/Users/alex/Software/XOR/RLlib Harnes Beta`

Before changing code, inspect the repository carefully.

Read these documents in order:

1. `AGENTS.md`
2. `docs/generic_harness_overview.md`
3. `docs/generic_harness_refactor.md`
4. `harness/AGENTS.md`
5. `experiments/AGENTS.md`
6. `learners/AGENTS.md`
7. `losses/AGENTS.md`
8. `analysis/AGENTS.md`
9. `envs/AGENTS.md`
10. `scripts/AGENTS.md`

Then inspect the current implementation, especially:

- `scripts/train.py`
- `scripts/hardware.py`
- `blueprints/`
- `analysis/`
- `envs/mess3/`
- `learners/`
- `losses/`
- existing tests and `pyproject.toml`

Also inspect `git status` and the current diff. This refactor may already
contain in-progress work. Preserve unrelated user changes and do not discard or
overwrite them.

## Architectural goal

Separate reusable capabilities from experiment-specific scientific recipes.

Generic packages provide reusable:

- environments;
- model components and RLModules;
- Learner integrations and loss primitives;
- analysis operations;
- runtime, hardware, checkpoint, and artifact mechanics.

Experiments compose these pieces. Generic packages must never import named
experiments.

Prefer composition through small functions, PyTorch modules, validated
configuration, callbacks, wrappers, and cooperative mixins. Do not create
deeper inheritance hierarchies or pre-build every possible model × head × loss
× algorithm combination.

Use Ray, RLlib, and Tune facilities when they fit naturally. Subclass
documented extension points when necessary rather than reimplementing
framework behavior by default.

## Experiment contract

Each runnable experiment leaf has exactly one `experiment.py` exposing:

```python
def run(context):
    ...
```

The experiment file is the complete scientific recipe. It owns:

- the algorithm and fresh `AlgorithmConfig`;
- environment class and `env_config`;
- model/RLModule and Learner composition;
- losses and their configuration;
- training budget and stopping condition;
- fixed seed or Tune seed space;
- experiment-specific adapters and analysis wiring.

Do not recreate the Blueprint schema, global arm registry, research phase
system, or approval gates.

Avoid arbitrary scientific hyperparameter CLI overrides. Scientific changes
should normally be represented by a new or edited `experiment.py`.
Operational controls such as seed, smoke mode, resume, and hardware may be
passed at runtime and must be recorded.

The default seed is `42`. Users must also be able to pass different seeds to
separately provisioned machines. Tune-based seed sweeps must remain possible.

Related MESS3 experiments should be organized as separate leaves beneath a
family such as:

```text
experiments/
  mess3_belief_geometry_2026_07/
    shared.py
    reward_only/
      experiment.py
      results/
      artifacts/
    next_token_aux_0p1/
      experiment.py
      results/
      artifacts/
    ...
```

Supporting scripts and notes sit directly in each leaf rather than in nested
`scripts/` or `notes/` directories.

## Training lifecycle

Replace the current monolithic `scripts/train.py`.

The experiment supplies its configuration and scientific stopping condition.
Generic execution belongs in `harness/`.

For Tune runs, the harness constructs a `Tuner` and calls `fit()`. RLlib's
Algorithm is the Tune Trainable, so Tune owns repeated training iterations.

For direct RLlib runs, `harness/runners.py` should own the ordinary loop:

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

Experiments should not copy this loop.

PPO behavior such as rollout aggregation, train batch size, minibatch size,
and repeated optimization epochs belongs in `PPOConfig`, for example through:

- `rollout_fragment_length`;
- `train_batch_size_per_learner`;
- `minibatch_size`;
- `num_epochs`;
- the relevant EnvRunner and Learner resource settings.

Tune is preferred for standard run management but must not be mandatory.
Direct RLlib, supervised, offline, and custom workflows must remain possible
through `run(context)`.

## Runtime and artifacts

Create a small immutable `RunContext` containing only shared runtime concerns:

- experiment, results, and artifacts directories;
- optional seed, defaulting to `42`;
- unique run ID;
- smoke mode;
- resume source;
- operational hardware/resource selection.

Do not add scientific configuration to `RunContext`.

Generic helpers should handle:

- runtime directory creation;
- hardware and Ray setup;
- Tune and direct-Algorithm execution;
- generic result recording;
- public RLlib checkpoint APIs;
- cleanup;
- provenance manifests.

Track compact findings, summaries, tables, and figures under `results/`.

Place Tune trial directories, full RLlib checkpoints, `.pt` files, raw
rollouts, event files, and large logs under ignored `artifacts/`.

Do not preserve old Blueprint/checkpoint compatibility. Existing experiments
and checkpoints will be reproduced after the refactor.

Durable remote artifact upload is deferred. Artifacts on destroyed remote
machines may be lost. Remote workflows should produce required compact outputs
before teardown.

## Environment and analysis boundaries

Keep reusable environments under `envs/`, including `envs/mess3/`.

Environment behavior is configured through `env_config`. Environments must
not own training loops, result paths, algorithms, phases, or gates.

Move the current MESS3 supervised trainer out of the environment package.

Expose evaluation diagnostics such as true latent state or belief through
public accessors or `info` fields. Analysis must not read private fields such
as `_s`, `_filter`, or `_obs_token`.

Generic analysis should contain reusable:

- checkpoint/result access;
- rollout collection;
- affine and linear probes;
- train/test splitting;
- generic metrics;
- plotting primitives.

Experiments provide small model-representation and task-target callables. Do
not add a formal representation protocol yet. Compute probe activations on
demand rather than storing them in rollout or replay data.

## Learner and loss boundaries

Keep reusable PyTorch components and RLlib integrations generic.

One-experiment model and Learner leaf compositions belong in `experiment.py`.
Promote them only when they have independent reuse.

Keep reusable tensor loss math in `losses/`. Cooperative auxiliary loss mixins
must call `super()` correctly and remain device-native.

Do not assume that every objective is algorithm-agnostic. Algorithm-specific
primary objectives may require algorithm-specific Learner integration.

Remove MESS3 observation-layout assumptions from generic losses. Keep domain
target extraction in MESS3 or the relevant experiment.

Never add `.cpu()`, `.numpy()`, or `.item()` calls to model or loss hot paths.

## How to structure the work

Before implementation, produce a concise staged work plan based on the actual
dependency graph. Identify which foundations must land before dependent code
can move.

A likely order is:

1. Establish normal package imports and baseline tests.
2. Introduce `harness/` runtime context, artifacts, hardware, runners, and CLI.
3. Add focused unit tests for the harness contracts.
4. Create initial MESS3 experiment-family structure and shared helpers.
5. Move experiment recipes out of Blueprints.
6. Remove Blueprint and gate dependencies.
7. Separate generic and MESS3 analysis.
8. Clean environment ownership and public diagnostics.
9. Finish learner/loss genericity and move experiment-only leaves.
10. Remove obsolete scripts and path hacks.
11. Rebuild integration tests and documentation.

Adjust this order if inspection reveals a safer dependency sequence. Explain
any substantial deviation.

Do not perform one enormous blind rewrite. Work in coherent stages and verify
each stage before continuing.

## Intermediate testing checkpoints

Establish the existing test baseline first and record failures that predate
your work.

At minimum, add or run checkpoints for:

- package imports without `sys.path` mutation;
- `RunContext` defaults, especially seed `42`;
- direct runner stopping and guaranteed `algo.stop()` cleanup;
- Tune single-trial construction;
- artifact/results directory separation;
- provenance manifest creation;
- environment configuration and diagnostics;
- learner/model composition;
- active auxiliary-loss composition;
- experiment module import/build smoke tests;
- one tiny direct RLlib PPO run;
- one tiny Tune-managed PPO run;
- one representative supervised path;
- the complete fast test suite.

Use small smoke budgets; do not launch full research runs.

Do not claim tests passed if they were skipped or unavailable. Report exact
commands and outcomes.

## Final documentation pass

Only after implementation is complete and tests are passing, update all
relevant `AGENTS.md` files.

At that point, make them forward-looking:

- describe the architecture that actually exists;
- state required contracts and dependency directions;
- provide concise examples of correct extension patterns;
- remove migration notes, stale violations, transitional language, and
  references to deleted files;
- ensure future agents are guided toward configuration and composition before
  customization;
- ensure truly experiment-specific work remains in experiments;
- retain device-native hot-path guidance.

Also update the overview/refactor documentation where the final implementation
differs from the proposed migration.

Do not create a Git commit unless explicitly requested.
