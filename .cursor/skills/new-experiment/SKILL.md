---
name: new-experiment
description: Plans, implements, smoke-tests, and runs a new experiment in a personal experiment repo against the shared rl-harness library. Use when adding an experiment recipe, a new study or condition, experiment-specific analysis, or reusable functionality discovered while executing an experiment.
---

# New experiment

Use this workflow when creating or running an experiment. The experiment repo is
the composition root; do not make the shared library know about named studies.
See `docs/multi_repo.md`.

## 1. Read and inspect before coding

Read, in order:

1. Library `AGENTS.md` and `docs/multi_repo.md`
2. `docs/generic_harness_overview.md`
3. Experiment-repo `experiments/AGENTS.md`
4. The `AGENTS.md` for every generic package you may change
5. `docs/checkpoint_strategy.md` if training state or longitudinal analysis matters

Then:

- inspect `git status` and preserve existing changes;
- inspect one structurally similar experiment, but do not copy its domain
  assumptions;
- search existing Ray/RLlib/Tune, harness, environment, model, loss, and
  analysis components before inventing another abstraction;
- identify the smallest smoke test that exercises the real extension points.

## 2. Specify the scientific recipe first

Before implementation, write down:

- hypothesis and primary comparison;
- environment, observation/action semantics, and reward;
- algorithm, model, Learner, and objectives;
- controls and baselines;
- fixed seed policy or Tune seed search space;
- training budget and stopping metric;
- success/failure metrics and analysis;
- required checkpoints and compact outputs;
- smoke budget;
- intended hardware and whether artifacts must survive a remote box.

Ask the user when one of these choices is material and unspecified. Do not hide
scientific decisions behind defaults merely to make the code run.

## 3. Decide where each piece belongs

Use this priority order:

1. Configure an existing framework or repository component.
2. If a reusable underlying concept is missing, add it to its generic package.
3. Keep the small task adapter in the environment domain or experiment family.
4. Keep one-off behavior in the experiment. Promote it only after reuse reveals
   a stable abstraction.

Ownership:

| Concern | Location |
|---|---|
| Complete recipe, budget, seed policy, condition-specific classes/adapters | `experiments/` |
| Gymnasium simulation, validated env config, domain solvers/targets | `envs/` |
| Reusable `nn.Module`, encoder, head, RLModule | `learners/` |
| Reusable tensor objective or cooperative Learner mixin | `losses/` |
| Generic checkpoint, rollout, probe, metric, aggregation, plotting operation | `analysis/` |
| Runtime context, artifacts, hardware, Ray setup, generic runners, CLI | `harness/` |
| Remote execution and infrastructure | `devops/` |

Dependencies point from experiments into generic packages, never backward.

### Escalate changes into the shared library

The default edit scope for this skill is the personal experiment repo's
`experiments/` tree. Before changing the sibling `rl-harness` library
(`envs/`, `learners/`, `losses/`, `analysis/`, `harness/`, `tests/`, `docs/`,
or `devops/`), stop and flag the user down.
Explain:

- the exact folder and files that would change;
- why configuration or an experiment-local adapter is insufficient;
- why the proposed behavior is genuinely reusable;
- which generic contracts and tests the change could affect.

Wait for explicit approval before crossing the boundary. A `harness/` change is
the highest escalation because it affects runtime behavior for every
experiment. Never modify `harness/` merely to make one recipe convenient; call
out the architectural impact and obtain explicit confirmation first.

## 4. Create the experiment leaf

Use a valid importable package:

```text
experiments/<study_name>/
  __init__.py
  shared.py                 # only stable family-level repetition
  <condition_name>/
    __init__.py
    experiment.py           # exactly one
    analyze.py              # optional local workflow
    findings.md             # optional durable interpretation
    results/                # generated compact outputs
    artifacts/              # generated, ignored
```

Every runnable leaf exposes:

```python
def run(context):
    ...
```

`experiment.py` owns the complete scientific recipe: fresh configuration,
environment, model/Learner composition, objectives, hyperparameters, budget,
seed policy, stopping condition, and analysis wiring.

Do not create a registry, Blueprint replacement, phase schema, or parallel
configuration hierarchy. `shared.py` may remove repetition but must not become
a hidden arm registry or mutable global recipe.

## 5. Use `RunContext` correctly

`RunContext` is immutable operational input:

- `experiment_dir`
- `results_dir`
- `artifacts_dir`
- `seed` (default `42`)
- `run_id`
- `smoke`
- `resume_from`
- `hardware`

Do not add scientific hyperparameters to it. Put scientific choices in
`experiment.py`.

Use `context.seed` directly. Never write `context.seed or 42`: seed `0` is
valid. If `None` is unsupported, reject it explicitly.

The runner configures Ray, but the recipe must still apply the selected
hardware profile to its `AlgorithmConfig` resources.

## 6. Build a fresh RLlib configuration

Adapt this shape; do not copy placeholder science:

```python
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune

TOTAL_ENV_STEPS = ...
SMOKE_ENV_STEPS = ...
ENV_CONFIG = {...}
MODEL_CONFIG = {...}


def build_config(context: RunContext) -> PPOConfig:
    profile = context.hardware or PROFILES["cpu"]
    config = (
        PPOConfig()
        .environment(MyEnv, env_config=ENV_CONFIG)
        .training(
            # Explicit scientific PPO settings.
            ...
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=MyModule,
                model_config=MODEL_CONFIG,
            )
        )
        .debugging(seed=context.seed)
    )
    return (
        config
        .env_runners(
            num_env_runners=(
                0
                if context.smoke
                else resolve_env_runners(profile, default=4)
            ),
            num_envs_per_env_runner=(
                1 if context.smoke else profile.num_envs_per_env_runner
            ),
            num_gpus_per_env_runner=(
                0 if context.smoke else profile.num_gpus_per_env_runner
            ),
        )
        .learners(
            num_gpus_per_learner=(
                1 if profile.learner_device == "cuda" else 0
            )
        )
    )


def run(context: RunContext):
    target = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    return run_tune(
        build_config(context),
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            ),
        },
    )
```

Return a new `AlgorithmConfig` from every `build_config` call. Define custom
RLModule/Learner leaves and Tune-serialized callables at module scope so Ray
workers can import them.

Use `run_algorithm` for a direct RLlib loop. Do not copy an
`algorithm.train()` loop into the experiment. Supervised, offline, and custom
workflows may implement their own loop in the experiment family.

## 7. Results, artifacts, and checkpoints

Write:

- compact JSON/CSV summaries, findings, tables, and figures to
  `context.results_dir`;
- Tune trees, full checkpoints, `.pt` files, raw rollouts, event files, and
  large logs to `context.artifacts_dir`.

Use `RunArtifacts` for standard paths and records. Use public RLlib checkpoint
APIs; never reach through private Learner or module attributes.

If analysis needs learning-time checkpoints, follow
`docs/checkpoint_strategy.md`. A final-only checkpoint is insufficient for
N-init or training curves, while `num_to_keep` alone cannot create a step-zero
checkpoint.

Remote artifacts are ephemeral. A self-destructing run must produce every
required compact result before teardown.

## 8. Analysis boundaries

Generic analysis receives callables/adapters for:

- representation extraction;
- target extraction from public environment `info` or accessors;
- action semantics;
- report-independent metrics.

Keep task labels, thresholds, state meanings, comparisons, and figure
composition in the experiment. Never read private environment fields such as
`_s`, `_filter`, or `_obs_token`.

Compute probe activations on demand. Do not add them to training rollout or
replay storage unless exact training-time activations are scientifically
required.

## 9. Verify before a real run

At minimum:

1. Import the experiment normally.
2. Build two configs and confirm they are distinct.
3. Assert seed/resource/smoke settings.
4. Run focused component tests with inline fixtures.
5. Run one real smoke path through the actual extension point.
6. Run the relevant package tests, then the fast suite.

Typical commands:

```bash
source .venv/bin/activate

rl-harness experiments.<study>.<condition>.experiment \
  --smoke --hardware-profile cpu

pytest -q -m "not slow"
```

Do not launch a full research run as a test.

For remote GPUs, read `.cursor/skills/vast-provisioning/SKILL.md`. The Vast
`--run` command executes inside the activated, pre-synced `.venv`; do not
prefix it with `uv run`.

After Tune, inspect `tune_summary.json` and verify every trial status. Do not
infer trial success solely from the outer process exit status.

## Common gotchas

- Do not import a named experiment or environment from `harness/`.
- Do not put a trainer, result path, algorithm, phase, or gate in `envs/`.
- Do not add arbitrary scientific CLI flags; edit or create a recipe.
- Do not mutate a module-level `AlgorithmConfig`; build a fresh one.
- Do not mirror RLlib configuration in a new dataclass or DSL.
- Do not create every model × head × loss × algorithm combination in
  `learners/`; keep one-off leaf compositions in `experiment.py`.
- Cooperative Learner mixins come before the RLlib base class and must call
  `super()` (except `ConfigurableOptimizerMixin`, which replaces the base
  optimizer setup and must not call `super().configure_optimizers_for_module`).
- Non-default optimizers: compose `ConfigurableOptimizerMixin` and set
  `optimizer/type` / `optimizer/kwargs` on `learner_config_dict` (see
  `learners/AGENTS.md`).
- Keep forward/loss hot paths on-device: no `.cpu()`, `.numpy()`, or `.item()`.
  Offline analysis and checkpoint serialization may transfer to CPU.
- Domain target extraction does not belong in a generic loss.
- Generic tests use inline fixtures. Named experiment execution belongs in
  experiment smoke/integration tests.
- `--resume-from` is interpreted by the recipe: a Tune experiment directory,
  Algorithm checkpoint, or analysis input are different things. Document the
  expected kind.
- Tune checkpoint pruning can silently remove the history needed for analysis.
- Full checkpoints and raw rollouts never belong under tracked `results/`.
- A remote self-destruct discards ignored artifacts; analyze before teardown.
- Avoid `sys.path` mutation. Install/import the project normally.
- Repository paths may contain spaces; quote paths in shell commands.
