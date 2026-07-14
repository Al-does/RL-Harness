# `experiments/` — complete scientific recipes

Experiments are the repository's composition root. They may import generic
harness, environment, learner, loss, analysis, and infrastructure packages.
Generic packages must never import experiments.

## Layout

Group related work in a valid Python package, then create one leaf per runnable
experiment:

```text
experiments/study_name_2026_07/
  shared.py
  condition_name/
    experiment.py
    analyze.py
    findings.md
    results/
    artifacts/
```

Each leaf has exactly one `experiment.py`. Keep its few supporting scripts and
notes directly in the leaf; do not add `scripts/` or `notes/` subdirectories.
The generated `artifacts/` directory is ignored and need not be committed.

Family-level helpers may remove repeated setup. They must not become a hidden
arm registry, mutable global config, or second experiment schema.

## Recipe contract

Every `experiment.py` exports:

```python
def run(context):
    ...
```

The file contains the full scientific recipe:

- fresh `AlgorithmConfig` or other training construction;
- environment class and `env_config`;
- model/RLModule and Learner classes;
- loss/objective configuration;
- training budget and stopping policy;
- fixed seed or Tune seed search space;
- model- and task-specific adapters;
- optional analysis wiring.

Use `context.seed` for a direct run; it defaults to `42` and may be supplied
per machine. Tune experiments may define multiple seeds and must record each
resolved trial seed.

Avoid arbitrary scientific CLI overrides. If a hyperparameter change is a new
scientific condition, create or edit an experiment leaf so the recipe remains
readable and reviewable.

## Custom code and promotion

First configure existing components. If a missing abstraction is a reusable RL
concept, add it to the appropriate generic package and keep only a small
adapter here. If the behavior is genuinely idiosyncratic, define it beside or
inside `experiment.py`.

One-off model and Learner leaf compositions belong here. Promote a composition
to `learners/` only after it is independently reusable; do not populate the
generic package with every model × head × loss × algorithm combination.

## Results and artifacts

Track under `results/`:

- findings, summary data, tables, and figures;
- compact resolved-config and run summaries.

Ignore under `artifacts/`:

- Tune directories and full RLlib checkpoints;
- `.pt` exports, raw rollouts, event files, and large logs.

Remote artifacts are currently ephemeral. Generate required compact outputs
before remote teardown. Object-storage upload and URI/hash recording are
future work.

## Orchestration and analysis

An experiment may include custom orchestration or gates, but no generic
harness behavior depends on them. Prefer an explicit local script that invokes
other experiment entry points.

Wire generic analyses with small callables that select model representations
and task targets. Do not force experiment semantics into `analysis/` merely to
reduce a few local lines.
