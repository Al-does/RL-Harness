# Repository architecture

This repository is the shared **rl-harness** library. Personal experiment
recipes live in separate researcher repos (for example
`alex-rl-experiments`) that editable-depend on a sibling checkout of this
library. Read `docs/generic_harness_refactor.md` before making structural
changes, and `docs/multi_repo.md` for the two-repo workflow.

## Ownership

- `harness/`: execution mechanics, runtime context, artifacts, hardware, and
  thin CLIs. It must not know any named experiment or environment.
- `learners/`: reusable RLlib models, Learners, and PyTorch components.
- `losses/`: reusable objective primitives and cooperative Learner extensions.
- `analysis/`: reusable checkpoint, rollout, probe, metric, and plotting tools.
- `envs/`: reusable Gymnasium environments and domain logic.
- `devops/`: infrastructure and remote execution mechanics.

Dependencies flow from experiment repos into this library, never the reverse.
Do not add an `experiments/` package here.

## Experiment contract

Every runnable experiment leaf (in a personal experiment repo) has exactly one
`experiment.py` and exposes:

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
2. Implement a missing reusable underlying concept in this library (PR here).
3. Keep only the small task adapter in the researcher's experiment repo.
4. Keep truly idiosyncratic code in the experiment repo and promote it after
   reuse demonstrates a stable abstraction.

Do not build speculative configuration DSLs. Prefer ordinary Python,
validated component configs, pure functions, `nn.Module` composition,
callbacks, and documented RLlib extension points. Use mixins only for
cooperative, orthogonal framework hooks.

## Runtime and artifacts

- Prefer Ray/RLlib/Tune lifecycle tools when they fit without distortion.
- Research gates and phase ordering are never harness requirements.
- Compact findings live in the experiment repo under `results/`.
- Put checkpoints, `.pt` files, Tune trial trees, raw rollouts, and large logs
  under ignored `artifacts/` in the experiment repo.
- Remote artifact durability is deferred. Ephemeral machines must produce
  required compact results before teardown.
- Use public RLlib checkpoint APIs; do not reach through private component
  attributes.
- Run manifests record both the experiment-repo commit and this library's
  commit (and package version when tagged).

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
- Experiment smoke tests live in the researcher's experiment repo.
- After changing an extension point, test both isolated composition and one
  representative RLlib integration path.

## Cursor Cloud specific instructions

Cloud agent VMs are CPU-only Ubuntu machines. Use them for code changes, fast
tests, and smoke runs. Full GPU training belongs on vast.ai via
`devops/vast/` (see `.cursor/skills/vast-provisioning/SKILL.md`).

Setup, standard commands, and architecture are documented in `README.md`.
Notes below are only the non-obvious gotchas for this environment.

- Dependencies are managed by `uv` (see `README.md`). The startup update
  script runs `uv sync --group dev` (also configured in
  `.cursor/environment.json`). Run everything through `uv run ...` or activate
  `.venv` first; the system `python3` is 3.12 and does not satisfy the
  `>=3.13` requirement, so `uv` provisions its own interpreter (3.14) into
  `.venv`.
- No linter is configured. `pytest` is the automated gate. The fast suite
  (`uv run pytest -q -m "not slow"`) is the library gate. Named experiment
  recipes and their tests live in personal experiment repos.
- This is a batch CLI harness, not a service: there is no web server, DB, or
  daemon to start. "Running the app" means invoking a leaf experiment via
  `rl-harness` from an experiment repo that depends on this library.
- For RLlib/Tune recipes, Ray is started in-process by the harness and shut
  down at the end; no external Ray cluster is needed and CPU is fine for
  `--smoke`.
- Smoke/dev runs write throwaway outputs under the experiment leaf's
  `results/<run-id>/` and `artifacts/<run-id>/`; do not commit artifacts.
- The experiment repo's editable dep is `rl-harness = { path = "../rl-harness"
  }`, but the cloud checkout of this library is `RL-Harness` (Linux is
  case-sensitive) and `bootstrap_local.sh` only auto-links `RL Harness` /
  `rl-harness-src`. So before `uv sync`/smoke-testing from
  `../alex-rl-experiments`, create the sibling link once (needs sudo since the
  repos root is root-owned): `sudo ln -sfn RL-Harness <repos-parent>/rl-harness`.
  The library's own `uv sync --group dev` needs no symlink.

### Verify changes

Run these before opening a PR to this library:

```bash
# Fast unit, architecture, and integration tests.
uv run pytest -q -m "not slow"
```

If you also changed a personal experiment recipe, smoke it from that repo:

```bash
cd ../alex-rl-experiments
uv run rl-harness \
  experiments.mess3_belief_geometry_2026_07.reward_only.experiment \
  --smoke
```

### Environment variables and secrets

Cloud agents do **not** read secrets from your laptop's shell or from `.env`
files in the repo. Values you add in **Cursor Dashboard → Cloud Agents →
Secrets** are injected into the cloud VM as normal environment variables when
an agent starts (and during the `install` step in `.cursor/environment.json`).

If you already created `GITHUB_TOKEN`, `VAST_API_KEY`, or similar entries
there, the agent sees them the same way as `echo $GITHUB_TOKEN` in a local
terminal — no extra wiring in this repository is required.

Cursor supports three secret types on the dashboard:

- **Environment Variable** — visible to the agent in chat and tool output; use
  for non-sensitive config (public URLs, feature flags).
- **Runtime Secret** — still exported as an env var, but redacted as
  `[REDACTED]` in transcripts, commits, and most agent-visible output; use for
  API keys and tokens.
- **Build Secret** — only available during Docker image build (for private
  package registries); not passed to the running agent.

Secrets can be workspace-wide or scoped to a specific saved environment. If an
agent cannot see a variable you expect, confirm you are on the same Cursor
account/team, that the secret name matches exactly, and restart the agent after
adding it.

For this repo, dashboard secrets are **optional** for most tasks. Add them only
when an agent session needs to:

- `GITHUB_TOKEN` — push results branches from vast self-destruct flows.
- `VAST_API_KEY` — rent remote GPU boxes from a cloud agent session.

Typical code changes and `pytest` runs do not need either.
