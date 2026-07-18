# RL Harness

This repository is a composition-first harness for reproducible RL research. 
- The primary goal is to afford rapid code contribution from coding agents but in a way that thoughtfully controls the slop and affords human review. Many agents.md files guide our beloved slop cannons to do things in ways that don't pollute the codebase and in fact even afford composition of pre-existing functionality to make work easier and more trusted as it can be built on what has already been tested.
- The secondary goal is careful tracking of experimental results and ensuring reproducibility. The writing, running, and saving of experiments is thoughtfully advised down to commit and seed number tracking. 

It's build on top of RLlib and therefore inherits its scalability and pre-built features. However, it also affords easy composition of most RL concepts for rapid experiment design and prototyping for anything one might want to add on top.

Vast.ai integration is also available for those who have an API key. This dev op affords faster and parallelized experimentation for cheap. 

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13 or newer.

```bash
uv sync --group dev
source .venv/bin/activate
```

## Run an experiment

Pass the dotted module path of a leaf `experiment.py`:

```bash
# Minimal RLlib wiring check; seed defaults to 42.
rl-harness \
  experiments.mess3_belief_geometry_2026_07.reward_only.experiment \
  --smoke

# Analytic workflow with an explicit seed.
rl-harness \
  experiments.mess3_belief_geometry_2026_07.operating_point_sweep.experiment \
  --seed 7 --smoke
```

Runtime-only options include `--seed`, `--smoke`, `--resume-from`,
`--hardware-profile`, and output-directory overrides. Scientific
hyperparameters live in the recipe rather than in a generic CLI override
schema.

Each run writes compact, reviewable records under the experiment's
`results/<run-id>/` and large generated data under ignored
`artifacts/<run-id>/`. The run manifest records source, Git, dependency,
runtime, seed, and hardware provenance.

## Architecture

- `harness/` — immutable runtime context, provenance, artifacts, hardware,
  direct-RLlib and Tune runners, and the CLI.
- `experiments/` — complete scientific recipes and study-specific adapters.
- `learners/` — reusable RLModules and on-device PyTorch components.
- `losses/` — reusable objective primitives and cooperative Learner mixins.
- `analysis/` — generic checkpoint, rollout, probe, metric, and plot tools.
- `envs/` — reusable Gymnasium environments and domain logic.
- `devops/` — remote execution and infrastructure mechanics.

Dependencies point from experiments into generic packages. Generic runtime,
learning, and analysis code does not import named experiments.

See [the harness overview](docs/generic_harness_overview.md) for design
guidance and [the refactor specification](docs/generic_harness_refactor.md)
for the detailed boundaries. The
[random-stream guide](docs/seed_streams.md) documents root-to-child seed
derivation and its compatibility rules.

## Add an unrelated experiment

Create a valid Python package path such as:

```text
experiments/my_study/baseline/
  experiment.py
  results/
  artifacts/       # generated and ignored
```

The only required entry point is:

```python
def run(context):
    ...
```

Build a fresh RLlib `AlgorithmConfig`, Tune workflow, supervised loop, or
offline analysis in that module. Use `context` only for shared operational
inputs such as seed, smoke mode, paths, resume source, and hardware. No
registry or Blueprint update is required.

## Included MESS3 examples

`experiments/mess3_belief_geometry_2026_07/` contains independent recipes for:

- analytic operating-point and action-lattice sweeps;
- reward, delay, auxiliary-loss, observation, memory, and scrambling
  conditions;
- supervised validation, checkpoint probing, scrambled evaluation, and study
  synthesis.

Reusable finite-HMM mechanics and the single generic Gymnasium environment
live under `envs/hmm/`. MESS3 supplies probability models, explicit
action/reward tasks, and analytic solvers under `envs/mess3/`; see
`docs/env_architecture.md`.

The top-level `results/` tree is a historical pre-cutover archive. Current
recipes neither write to it nor reconstruct runs from its old Blueprint
records.

## Tests

```bash
# Unit, architecture, recipe-construction, and short integration tests.
pytest -q -m "not slow"

# Also run long Monte Carlo checks.
pytest -q
```

Remote RTX 4090 execution is documented in
[`devops/vast/README.md`](devops/vast/README.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
