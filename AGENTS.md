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
