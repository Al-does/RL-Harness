# Repository architecture

This repository is a generic RL research harness. Read
`docs/generic_harness_refactor.md` before making structural changes.

## Ownership

- `harness/`: execution mechanics, runtime context, artifacts, hardware, and
  thin CLIs. It must not know any named experiment or environment.
- `experiments/`: complete scientific recipes and experiment-specific code.
  This is the composition root.
- `learners/`: reusable RLlib models, Learners, and PyTorch components.
- `losses/`: reusable objective primitives and cooperative Learner extensions.
- `analysis/`: reusable checkpoint, rollout, probe, metric, and plotting tools.
- `envs/`: reusable Gymnasium environments and domain logic.
- `devops/`: infrastructure and remote execution mechanics.

Dependencies flow from experiments into generic packages, never the reverse.

## Experiment contract

Every runnable experiment leaf has exactly one `experiment.py` and exposes:

```python
def run(context):
    ...
```

The experiment file owns all scientific choices: algorithm, environment
config, model, Learner, losses, hyperparameters, budget, seed policy, search
space, and analysis wiring.

Do not recreate `Blueprint`, arm registries, phase schemas, or a parallel
configuration hierarchy. Build fresh RLlib `AlgorithmConfig` objects directly.

The default runtime seed is `42`. Runtime seed, smoke, resume, and hardware
controls may come from the CLI and must be recorded. Avoid arbitrary CLI
overrides for scientific hyperparameters; edit or create an experiment recipe.

## Choosing where new code belongs

Use this order:

1. Configure an existing Ray, RLlib, Tune, harness, model, loss, environment,
   or analysis component.
2. Implement a missing reusable underlying concept in the appropriate generic
   package.
3. Keep only the small task adapter in the environment-domain or experiment
   package.
4. Keep truly idiosyncratic code in the experiment and promote it after reuse
   demonstrates a stable abstraction.

Do not build speculative configuration DSLs. Prefer ordinary Python,
validated component configs, pure functions, `nn.Module` composition,
callbacks, and documented RLlib extension points. Use mixins only for
cooperative, orthogonal framework hooks.

## Runtime and artifacts

- Prefer Ray/RLlib/Tune lifecycle tools when they fit without distortion.
- Research gates and phase ordering are never harness requirements.
- Track compact findings, summaries, and figures under an experiment's
  `results/`.
- Put checkpoints, `.pt` files, Tune trial trees, raw rollouts, and large logs
  under ignored `artifacts/`.
- Remote artifact durability is deferred. Ephemeral machines must produce
  required compact results before teardown.
- Use public RLlib checkpoint APIs; do not reach through private component
  attributes.

## Analysis

Generic analysis accepts adapters for model representations and task targets.
It must not inspect private environment fields or assume MESS3 beliefs,
tokens, state counts, or result paths.

Compute probe activations on demand. Do not add them to replay or rollout
storage unless the experiment explicitly requires exact training-time data.

## Performance and tests

- Keep forward, loss, and other hot-path operations on-device. Do not add
  `.cpu()`, `.numpy()`, or scalar synchronization to training paths.
- Generic tests use inline fixtures rather than named experiments.
- Environment tests cover environment/domain behavior.
- Experiment smoke tests verify recipe construction and minimal execution.
- After changing an extension point, test both isolated composition and one
  representative RLlib integration path.

## Cursor Cloud specific instructions

Setup, standard commands, and architecture are documented in `README.md`.
Notes below are only the non-obvious gotchas for this environment.

- Dependencies are managed by `uv` (see `README.md`). The startup update
  script runs `uv sync --group dev`. Run everything through `uv run ...` or
  activate `.venv` first; the system `python3` is 3.12 and does not satisfy the
  `>=3.13` requirement, so `uv` provisions its own interpreter (3.14) into
  `.venv`.
- No linter is configured. `pytest` is the automated gate. The fast suite
  (`uv run pytest -q -m "not slow"`) is ~2 min / 116 tests; drop `-m "not
  slow"` to include the long Monte Carlo checks.
- This is a batch CLI harness, not a service: there is no web server, DB, or
  daemon to start. "Running the app" means invoking a leaf experiment via
  `rl-harness` (or `uv run rl-harness`).
- For RLlib/Tune recipes (e.g. `reward_only`), Ray is started in-process by the
  harness and shut down at the end; no external Ray cluster is needed and CPU
  is fine for `--smoke`.
- `--smoke` bounds training budget, but analytic sweeps (e.g.
  `operating_point_sweep`) still run for several minutes and print nothing to
  stdout — check `results/<run-id>/run_manifest.json` (`status: completed`) and
  emitted figures/CSVs to confirm success.
- Smoke/dev runs write throwaway outputs under each experiment's
  `results/<run-id>/` and `artifacts/<run-id>/`; do not commit these.
