# `envs/` — reusable environments and domain logic

Each `envs/<name>/` package implements a reusable environment or benchmark
domain. Environments do not belong inside individual experiments.

## Contract

- Implement the Gymnasium API.
- Accept validated behavior through `env_config` supplied by
  `AlgorithmConfig.environment(...)`.
- Keep defaults explicit and deterministic under a supplied seed.
- Expose correct observation and action spaces.
- Keep reusable filters, wrappers, solvers, and analytic baselines with their
  domain.

An environment must not import the harness, experiments, learners, losses, or
analysis pipeline.

## Configuration

Environment configuration describes simulation behavior, not training:

- dynamics and observation variants;
- episode horizon;
- action constraints;
- optional diagnostic instrumentation.

Algorithm, model, loss, training budget, checkpoint, result-path, phase, and
gate settings do not belong in an environment config.

Avoid top-level special cases in the harness. For example, input scrambling is
an environment config or wrapper selected by an experiment, not a Boolean
field that generic checkpoint or training code interprets.

## Diagnostics

Evaluation may need true latent state, beliefs, or other privileged values.
Expose these through documented public accessors or `info` fields, optionally
behind a diagnostic config when computation is expensive.

Diagnostic data must not silently enter the policy observation. Generic
analysis must not read private attributes such as `_s` or `_filter`.

## Boundaries and tests

Environment packages may contain environment-focused tests and reusable
domain solver tests. Move model, RLModule, Learner, probe-pipeline, and training
tests to their owning packages.

Training loops do not belong under `envs/`. Move the current MESS3 supervised
trainer to an experiment leaf initially; promote a generic supervised helper
only after another experiment demonstrates reuse.
