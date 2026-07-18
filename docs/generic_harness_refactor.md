# Generic RL Harness Refactor

## Purpose

Turn the current MESS3-oriented repository into a reusable RL research
harness. Generic packages should implement reusable RL, runtime, environment,
and analysis concepts. Experiment folders should contain complete scientific
recipes and any irreducibly experiment-specific code.

> **Multi-repo note (2026-07):** Named experiment trees now live in personal
> experiment repositories (for example `alex-rl-experiments`). This library
> repo no longer packages `experiments/`. See `docs/multi_repo.md`. Layout
> diagrams below that show `experiments/` describe the logical composition
> root, which may be a separate git repository.

This is a hard cutover. Existing checkpoints, run manifests, result layouts,
and import paths do not require compatibility shims. Reproduce the experiments
after the refactor instead of preserving the old execution format.

## Implementation status

The hard cutover described here is implemented. The repository now has the
generic `harness/` runtime, independent MESS3 experiment leaves, generic
analysis adapters, experiment-owned task target extraction, and
experiment-local model/Learner leaf compositions. The Blueprint registry,
phase launchers, environment-owned trainers, and path-mutation scripts have
been removed. This document remains the architectural rationale and boundary
reference; current usage starts in the root `README.md`.

## Settled decisions

1. Use Ray, RLlib, and Tune facilities when they fit cleanly. Subclass or
   replace them when an experiment genuinely requires different behavior.
2. Delete the `Blueprint`, `PPOSpec`, and global arm-registry architecture.
   They duplicate `AlgorithmConfig` and couple the generic launcher to MESS3.
3. Every runnable experiment leaf owns one `experiment.py` with one required
   entry point:

   ```python
   def run(context):
       ...
   ```

   `run(context)` may build an RLlib algorithm, create a Tune run, invoke a
   supervised helper, or implement another workflow.
4. An experiment file is the complete scientific recipe. Hyperparameters,
   model and Learner classes, environment config, training budget, seed
   policy, Tune search space, and analysis choices belong there.
5. Avoid arbitrary scientific hyperparameter overrides from the CLI. A
   scientific change should normally produce a new or edited
   `experiment.py`. Runtime controls such as seed, smoke mode, resume, and
   hardware are valid CLI inputs and must be recorded.
6. The default seed is `42`. A user may pass another seed to one machine, or
   an experiment may define a Tune seed sweep. The resolved seed is recorded
   for every run.
7. Do not implement research phases, arm registries, approval files, or
   programmatic gates in the generic harness. An experiment may implement its
   own orchestration script locally.
8. Keep reusable environments under `envs/<name>/`, not under experiments.
   Environments accept standard `env_config` values supplied through
   `AlgorithmConfig.environment(...)`.
9. Keep compact results in Git and large/raw artifacts out of Git.
10. Do not introduce a formal model-representation API yet. Generic probes
    accept an experiment-supplied extraction callable and compute activations
    on demand.

## Target layout

```text
AGENTS.md
analysis/
  AGENTS.md
  checkpoints.py
  rollouts.py
  probes/
envs/
  AGENTS.md
  mess3/
harness/
  AGENTS.md
  context.py
  artifacts.py
  hardware.py
  runners.py
  cli.py
experiments/
  AGENTS.md
  mess3_belief_geometry_2026_07/
    shared.py
    reward_only/
      experiment.py
      analyze.py
      findings.md
      results/
      artifacts/               # generated, ignored
    next_token_aux_0p1/
      experiment.py
      results/
      artifacts/
learners/
  AGENTS.md
losses/
  AGENTS.md
devops/
tests/
```

Experiment and family directory names must be valid Python package names.
Use underscores rather than hyphens and do not begin package names with a
date.

Each experiment leaf contains exactly one `experiment.py`. Supporting scripts,
notes, and findings sit directly in the leaf; do not create `scripts/` or
`notes/` subdirectories for small experiment-local collections. Family-level
`shared.py` modules may remove repetition, but they must not become arm
registries or hidden recipes.

## Dependency direction

```text
experiment.py
  ├── Ray / RLlib / Tune
  ├── harness/
  ├── learners/ and losses/
  ├── envs/
  └── analysis/

harness/   must not import experiments or a named environment
analysis/  must not import experiments or assume MESS3
learners/  must not import experiments, environments, or analysis
losses/    must not import experiments or environments
envs/      must not import harness, learners, losses, or experiments
devops/    may invoke generic harness entry points, not named experiments
```

The experiment layer is the composition root. It is allowed to import all
lower layers and to define small custom subclasses, adapters, or callbacks.

## Experiment entry point and runtime context

Do not replace `Blueprint` with another field-heavy experiment schema. The
Python module and its `run(context)` function are the contract.

A small immutable `RunContext` carries execution mechanics:

```python
@dataclass(frozen=True)
class RunContext:
    experiment_dir: Path
    results_dir: Path
    artifacts_dir: Path
    seed: int | None = 42
    run_id: str = ...
    smoke: bool = False
    resume_from: Path | None = None
```

Additional fields require evidence that they are shared runtime concerns.
Do not add algorithm hyperparameters, model choices, environment behavior, or
analysis settings to `RunContext`.

Seed behavior:

- A direct run uses `context.seed`, defaulting to `42`.
- A manually distributed run passes a different `--seed` to each machine.
- A Tune experiment may ignore the fixed seed and define a seed search space. You should put a comment/warning saying as much.
- Every resolved trial seed belongs in its trial config and run manifest.
- Supervised/custom runners must seed their environment, NumPy, and framework
  RNGs consistently; do not assume RLlib will do this for non-RLlib code.

## Ray, RLlib, and Tune

`AlgorithmConfig` is the source of truth for RLlib algorithm configuration.
Do not mirror its fields in local dataclasses merely to copy them back later.

Tune is the preferred default for standard run and sweep lifecycle concerns:

- trial execution and parallelism;
- stop criteria;
- checkpoint retention;
- metrics and trial configs;
- resume and failure handling;
- trial storage layout.

Tune is not mandatory. `run(context)` deliberately permits a direct
`Algorithm` loop, supervised training, offline workflows, or custom algorithm
subclasses. Use the smallest standard mechanism that satisfies the experiment.

Subclass RLlib at documented extension points when configuration, callbacks,
connectors, or composition cannot express the desired behavior. Do not
subclass merely to rename or mirror existing configuration.

## Provenance

`experiment.py` records scientific intent, but it is mutable and cannot record
the resolved facts of a particular execution. Generic run helpers should write
a compact manifest containing:

- experiment module path and source hash;
- Git commit and dirty-worktree status;
- dependency-lock hash and key framework versions;
- command and runtime overrides;
- resolved seed and trial/run ID;
- start/end timestamps and completion status;
- hardware/resource summary;
- artifact URI and content hash when durable storage is implemented.

Tune/RLlib checkpoints and trial configs remain authoritative for framework
state. Do not attempt to JSON-serialize every Python class or callable in an
`AlgorithmConfig` as a substitute for source control.

## Results and artifacts

Each experiment leaf has two output classes:

### `results/` — compact and tracked

- findings and research notes;
- summary JSON/CSV;
- aggregate metrics and tables;
- figures;
- compact resolved-config or run-summary records.

### `artifacts/` — generated and ignored

- Tune trial directories;
- full RLlib checkpoints;
- module `.pt` exports;
- raw rollout payloads;
- TensorBoard/event data;
- verbose logs and temporary data.

In future work we will create a location to upload them to object storage and record the URI/hash in results/. For now, ignore the entire artifacts tree:

```gitignore
experiments/**/artifacts/
```

Ignoring only `*.pt` is insufficient because RLlib checkpoints are directory
trees with several file formats. Tracking only their metadata creates broken,
partial checkpoints.

Local artifacts persist on the local filesystem but are not versioned.
Ephemeral remote artifacts have no durability guarantee in the first version
of this harness. Remote experiments must produce any desired summaries and
figures in `results/` before teardown. Durable object-storage upload plus a
recorded URI/content hash is explicitly deferred.

A small `RunArtifacts` facade may normalize access to manifests, trial
configs, metrics, checkpoints, and directories. It must not contain
experiment-specific metric names or analysis behavior.

## Analysis and probing

Separate reusable analysis operations from experiment adapters.

Generic analysis may provide:

- checkpoint and result discovery;
- rollout iteration with injected policy/representation adapters;
- affine and linear probe fitting;
- train/test splitting;
- global R² and conditional residual R²;
- generic aggregation and plotting primitives.

Experiments or environment-domain helpers provide:

- which model activation to extract;
- which environment value is the target;
- MESS3 belief, token-branch, or hidden-state semantics;
- custom comparisons, thresholds, and figures.

Do not add probe activations to rollout storage or a replay buffer by default.
Post-hoc probes should load a model and compute activations on demand. A
generic probe initially accepts a callable such as:

```python
def extract_representation(module, observation, state):
    return module.encode_step(observation, state)
```

Introduce a formal representation protocol only after multiple incompatible
models demonstrate that callbacks are insufficient.

Environment latent state and network hidden representations are different:

- a network representation comes from the model or an extraction callable;
- a true environment state/belief comes from a public diagnostic accessor or
  `info` field.

Analysis must not read private fields such as `env._s` or `env._filter`.

## Environment contract

An environment package owns simulation and reusable domain logic:

- Gymnasium environment classes;
- validated environment config and defaults;
- filters, solvers, wrappers, and reusable analytic baselines;
- public diagnostic data useful for evaluation;
- environment-focused tests.

An environment must not own:

- a training loop;
- an RLModule or Learner;
- experiment budgets or result paths;
- phase/gate behavior;
- a specific analysis pipeline.

Move the current `envs/mess3/supervised.py` training loop out of the
environment package. Keep exact finite-HMM belief operations generic and
MESS3-specific solvers with the domain. Expose true latent state and belief
diagnostics through a public evaluation interface without adding them to
policy observations.

## Promotion triage for new functionality

When an experiment needs custom behavior:

1. Use existing RLlib, Ray, Tune, harness, model, loss, environment, and
   analysis components through configuration.
2. If a reusable underlying RL concept is missing, implement that concept in
   the appropriate generic package.
3. Prefer small pure functions, `nn.Module` components, wrappers, callbacks,
   and explicit composition. Use mixins where an orthogonal cooperative RLlib
   extension point genuinely benefits from them.
4. If only a small adapter is specific, keep the generic operation shared and
   define the adapter in the experiment or environment-domain package.
5. Keep truly idiosyncratic code in the experiment. Promote it only after a
   second use or a clearly stable abstraction appears.

Do not create a speculative configuration language to make every one-off
customization look generic.

## Implemented migration

The cutover followed these stages. Keep them as a reference for the intent
behind the resulting package boundaries, not as pending work.

### 1. Establish packaging and runtime foundations

- Make repository packages importable normally from `pyproject.toml`.
- Remove `sys.path.insert(...)` from executable files.
- Create `harness/context.py`, `harness/artifacts.py`, and generic run helpers.
- Move `scripts/hardware.py` to `harness/hardware.py`.
- Provide a thin CLI that loads an experiment module and calls `run(context)`.
- Default the CLI seed to `42`.

### 2. Remove the Blueprint system

- Delete `Blueprint`, `PPOSpec`, `ModelSpec`, registry side effects, and
  `blueprints/mess3_arms.py` after their recipes have moved.
- Build fresh `AlgorithmConfig` objects directly in experiment files.
- Remove `env_factory_from_blueprint`, `module_from_blueprint`, and every
  serialized `blueprint.json` assumption.
- Do not add compatibility adapters for old blueprint or checkpoint schemas.

Typed model-component configs may remain where they validate reusable model
construction. The problem is the parallel experiment schema, not typed
component configuration.

### 3. Replace `scripts/train.py`

Extract generic concerns:

- context and output directory creation;
- optional Tune execution;
- public RLlib checkpoint APIs;
- generic manifest and result handling;
- hardware/runtime setup.

Remove rather than generalize:

- phase gate checks;
- MESS3 supervised target inference;
- `scramble_tokens` special handling;
- hard-coded `next_token_aux/*` metric extraction;
- global `results/phaseK/...` paths;
- direct imports of MESS3 code;
- private `learner_group._learner` checkpoint access.

Move vast.ai teardown and remote lifecycle behavior behind `devops/` hooks or
generic post-run callbacks.

### 4. Create MESS3 experiment leaves

Convert the current arm matrix and phase scripts into independent leaves, for
example:

```text
experiments/mess3_belief_geometry_2026_07/
  shared.py
  operating_point_sweep/
    experiment.py
  passive_probe_validation/
    experiment.py
  reward_only/
    experiment.py
  no_delay/
    experiment.py
  next_token_aux_0p1/
    experiment.py
  next_token_aux_0p5/
    experiment.py
  prediction_only/
    experiment.py
  oracle_observation/
    experiment.py
  belief_observation/
    experiment.py
  stack_02/
    experiment.py
  stack_04/
    experiment.py
  stack_08/
    experiment.py
  stack_16/
    experiment.py
  scrambled_training/
    experiment.py
  scrambled_evaluation/
    experiment.py
```

This list is a migration example, not a registry. Each file contains a
complete runnable recipe. Cross-experiment synthesis may be another runnable
leaf or a family-level script, but the harness does not enforce dependencies
between them.

### 5. Separate generic and MESS3 analysis

- Replace `analysis/checkpoints.py` blueprint reconstruction with generic
  `RunArtifacts` and public RLlib/Tune checkpoint loading.
- Split `analysis/probe.py` into generic probe math, generic parameterized
  rollout collection, and MESS3 adapters.
- Parameterize reusable simplex plotting; keep MESS3 labels and figure
  composition local.
- Move `probe_arm.py`, phase findings, null-bracket, and scrambled-input
  workflows to the relevant experiment leaves.
- Merge or delete one-off duplicate drivers such as `probe_ckpt_fig.py`.

### 6. Clean environment and generic learning packages

- Keep `envs/mess3` as a reusable environment/domain package.
- Move its supervised trainer into an experiment or a promoted generic
  supervised helper.
- Move RLModule tests out of `envs/mess3/tests`.
- Remove MESS3 observation-layout assumptions from generic loss primitives;
  inject task target extraction where needed.
- Keep generic encoder/head/loss pieces shared, but place one-experiment leaf
  model and Learner compositions in that experiment's `experiment.py`.
- Do not create the combinatorial cross-product of every model, head, loss,
  and algorithm in `learners/`.

### 7. Rebuild tests and documentation

- Generic unit tests must use inline fixtures, not named MESS3 experiments.
- Environment tests cover environment behavior and diagnostics.
- Experiment smoke tests verify that each `experiment.py` can build its
  configured workflow.
- Integration tests cover representative RLlib and supervised runs.
- Update the root README to describe the harness first and link experiments
  as examples.
- Keep folder-level `AGENTS.md` files aligned with the final code; remove
  stale “known violation” lists once violations are fixed.

## Refactor completion criteria

- A new unrelated experiment can be added without editing generic packages.
- A normal experiment consists primarily of one `experiment.py` using
  existing components and configs.
- Generic runtime code imports no named experiment or environment.
- Generic analysis contains no MESS3 field names, state counts, or arm names.
- No scientific phase or approval gate exists in the harness.
- Seed `42` works by default; explicit per-machine seeds and Tune seed sweeps
  are both supported and recorded.
- Results are compact and tracked; artifacts and complete checkpoints are
  isolated under ignored `artifacts/`.
- No old blueprint/checkpoint compatibility code remains.
- All hot-path model and loss operations remain device-native.

## Deferred work

- Durable remote artifact upload to object storage.
- Artifact URI/hash recording and retrieval.
- A formal network-representation protocol.
- A generic supervised backend beyond the helpers demanded by real
  experiments.
- Cross-experiment orchestration or gate frameworks.
