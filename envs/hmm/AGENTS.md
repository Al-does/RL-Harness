# `envs/hmm/` — stable generic finite-HMM machinery

This package is shared infrastructure. It owns validated HMM data, exact
belief updates, simulation, history, observations, diagnostics, and the
Gymnasium lifecycle. Domain and experiment semantics do not belong here.

## Default extension path

- For a new action/reward objective, add one file under
  `envs/<domain>/tasks/<task_name>.py`.
- Keep action interpretation, controlled dynamics, reward calculation, and
  task-local state together in that file.
- Reuse an existing domain model factory. If genuinely new probability data is
  required, add a domain model factory rather than changing `HMMModel`.
- Select the model and task from the experiment using primitive import-path
  configuration; do not add a registry.
- The normal change should touch the task, its tests, and the experiment
  recipe—not `envs/hmm/`.

```python
class MyTask:
    requires_belief = False

    def __init__(self, *, model, ...):
        self.action_space = ...
        self.action_observation_space = ...

    def reset(self) -> None: ...

    def resolve_action(self, action, state, model) -> ActionDecision:
        return ActionDecision(
            requested_action=...,
            executed_action=...,
            transition_matrix=...,
        )

    def reward(self, event, decision) -> tuple[float, dict[str, float]]: ...

    def encode_action(self, executed_action) -> np.ndarray: ...
```

## Protect the generic boundary

- Do not add task-name branches, modes, one-off config fields, domain rewards,
  domain transition rules, or experiment assumptions to `HMMEnv`.
- Do not change generic behavior merely to make one experiment shorter.
- Keep RLlib config values primitive and serializable; tasks are loaded via
  `package.module:Class`.
- Read `docs/env_architecture.md` before changing this package.

Bug fixes belong at the layer where the defect actually lives. Fix generic HMM
code when its contract or implementation is wrong, even if one experiment
revealed the bug. Promote a new generic capability only when it is
domain-independent, has a clear contract, and is covered by generic tests.
