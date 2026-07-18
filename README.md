# RL Harness

Shared RLlib research library for reproducible experiment composition.

- Rapid, reviewable contribution from coding agents (package-level `AGENTS.md`
  files keep generic code generic).
- Provenance for every run (experiment-repo commit, library commit, seed,
  hardware, lockfile).
- Optional vast.ai tooling for cheap parallel GPU runs.

Personal experiment recipes do **not** live here. **Entry point for
researchers:** fork
[`rl-experiments`](https://github.com/Al-does/rl-experiments), clone your fork,
run `./scripts/bootstrap_local.sh` (clones this library beside it). See
[docs/multi_repo.md](docs/multi_repo.md).

## Setup (library development)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13 or newer.

```bash
uv sync --group dev
source .venv/bin/activate
```

## Run an experiment

From your personal experiment repo (after `uv sync` there):

```bash
uv run rl-harness \
  experiments.mess3_belief_geometry_2026_07.reward_only.experiment \
  --smoke
```

The CLI imports a dotted module path; the experiment package must be installed
in that environment (the personal repo packages `experiments*`).

Runtime-only options include `--seed`, `--smoke`, `--resume-from`,
`--hardware-profile`, and output-directory overrides. Scientific
hyperparameters live in the recipe.

Each run writes compact records under the experiment leaf's
`results/<run-id>/` and large data under ignored `artifacts/<run-id>/`.

## Architecture

- `harness/` — immutable runtime context, provenance, artifacts, hardware,
  direct-RLlib and Tune runners, and the CLI.
- `learners/` — reusable RLModules and on-device PyTorch components.
- `losses/` — reusable objective primitives and cooperative Learner mixins.
- `analysis/` — generic checkpoint, rollout, probe, metric, and plot tools.
- `envs/` — reusable Gymnasium environments and domain logic.
- `devops/` — remote execution and infrastructure mechanics.

Dependencies point from experiment repos into this library. Generic packages
never import named experiments.

See [the harness overview](docs/generic_harness_overview.md) for design
guidance and [the refactor specification](docs/generic_harness_refactor.md)
for detailed boundaries.

## Contribute a reusable change

```bash
git checkout -b alex/my-change
# edit learners/, losses/, harness/, …
uv run pytest -q -m "not slow"
git push -u origin HEAD
gh pr create
```

Idiosyncratic science stays in the experiment repo until reuse proves an
abstraction worth promoting here.

## Included domains

Reusable finite-HMM mechanics and the Gymnasium environment live under
`envs/hmm/`. MESS3 supplies probability models and wrappers under `envs/mess3/`.
Concrete MESS3 study recipes live in `alex-rl-experiments`.
