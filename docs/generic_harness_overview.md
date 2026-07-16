# Generic RL Harness Overview

## Goal

Build a research harness where new experiments are assembled from reusable RL
capabilities without forcing unrelated research into one inheritance hierarchy
or one global experiment schema.

The repository has two broad layers:

- **Generic capabilities** implement reusable models, losses, environments,
  analysis tools, and execution mechanics.
- **Experiments** compose those capabilities into complete scientific recipes
  and contain any code that is genuinely specific to one study.

The dependency direction is one-way: experiments may use generic packages;
generic packages never know about named experiments.

## Composition over hierarchy

Prefer small pieces that combine cleanly:

- pure PyTorch components for encoders and heads;
- thin RLlib model and Learner integrations;
- cooperative mixins for orthogonal model outputs or auxiliary objectives;
- pure tensor functions for reusable loss math;
- wrappers, callbacks, and small adapters for task-specific behavior;
- ordinary configuration for choices already supported by a component.

A new capability should normally be a new focused file or component, not
another level in a deep base-class tree. Concrete experiment classes compose
the required pieces at the leaf. The generic library should not pre-build the
combinatorial cross-product of every model, head, loss, and algorithm.

Mixins are one useful mechanism, not the default answer to every abstraction.
Use them where framework hooks must cooperate through `super()`. Prefer plain
functions and contained modules when inheritance provides no clear benefit.

## Generic packages

- `learners/` contains reusable neural components, RLModules, and Learner
  integrations. Model dimensions and behavior are selected through validated
  configuration rather than experiment-specific subclasses.
- `losses/` contains reusable objective math and composable Learner
  extensions. Domain-specific target extraction remains outside generic loss
  primitives.
- `envs/` contains reusable Gymnasium environments and their domain logic.
  Environment behavior is selected through `env_config`. Finite discrete HMM
  simulation lives in one generic environment, while domain tasks own action
  and reward semantics as described in `docs/env_architecture.md`.
- `analysis/` contains reusable operations such as checkpoint access,
  rollout collection, probes, metrics, and plotting primitives. Experiments
  provide representation and target adapters.
- `harness/` contains runtime context, artifact handling, hardware setup, and
  thin execution helpers. It does not contain scientific choices.

Use Ray, RLlib, and Tune facilities where they fit naturally. Extend or
subclass their documented interfaces when configuration and composition are
insufficient, rather than rebuilding framework behavior by default.

## Experiments as complete recipes

Each experiment lives in its own leaf folder and has one `experiment.py` with
a `run(context)` entry point. That file fully defines the scientific recipe:

- algorithm and environment;
- model, Learner, and losses;
- hyperparameters and training budget;
- seed policy or search space;
- experiment-specific adapters and analysis wiring.

Supporting scripts, notes, and custom code live beside `experiment.py`.
Compact findings, summary data, and figures live in `results/`; large
checkpoints and raw outputs live in ignored `artifacts/`.

Scientific changes should normally be made in the experiment recipe, not
hidden behind arbitrary CLI overrides. Runtime controls such as seed, smoke,
resume, and hardware remain external and are recorded. The default seed is
`42`.

An experiment should be understandable and reproducible from its source plus
its compact run record. The harness does not impose research phases, arm
registries, or approval gates. Experiments may implement their own local
orchestration when needed.

## Deciding where custom work belongs

1. Configure an existing generic or framework component.
2. If a reusable underlying RL concept is missing, add that concept to the
   appropriate generic package.
3. Keep the small task-specific adapter in the environment domain or
   experiment.
4. Keep irreducibly idiosyncratic work in the experiment.
5. Promote experiment code only when reuse reveals a stable abstraction.

This keeps the shared harness broadly useful without preventing researchers
from moving quickly when an experiment needs something unusual.
