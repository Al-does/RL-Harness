# `scripts/` — transitional executable entry points

This folder currently mixes generic runtime code with MESS3 experiment
pipelines. During the refactor, shrink it to thin executable wrappers or remove
it in favor of package entry points.

## Target ownership

- Generic execution mechanics move to `harness/`.
- Environment/domain code stays in `envs/`.
- MESS3 training, probing, findings, nulls, and evaluation scripts move beside
  the relevant `experiments/.../<leaf>/experiment.py`.
- Remote lifecycle logic stays in `devops/` or uses generic harness hooks.

A surviving script should parse arguments and call an importable function. It
must not contain a second implementation of environment resolution,
checkpointing, probe fitting, plotting, or result aggregation.

## Remove, do not generalize

- phase gates and approval artifacts;
- global Blueprint/arm lookup;
- `results/phaseK` paths;
- MESS3 supervised target inference;
- `scramble_tokens` rewriting;
- hard-coded auxiliary metric names;
- `sys.path.insert(...)`;
- subprocess calls used only to invoke another Python module in this repo.

Do not preserve compatibility with old Blueprint JSON or checkpoint layouts.

## Scientific arguments

Generic CLIs may accept operational controls such as experiment module, seed
(default `42`), smoke mode, resume source, and hardware. Do not add arbitrary
scientific hyperparameter overrides; those choices belong in
`experiment.py`.
