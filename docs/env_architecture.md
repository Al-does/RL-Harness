# HMM Environment Architecture

## Goal

Provide one small reusable Gymnasium environment for finite discrete HMMs.
Domain packages supply probability data and one task object. The generic
environment owns simulation, history, observations, optional belief tracking,
and diagnostics.

Avoid separate runtime, controller, observation-view, and reward-component
frameworks. Keep each concrete task in its own file so its action and reward
semantics can be read together.

## Implemented layout

```text
envs/hmm/
  model.py                 # validated HMM probability data
  belief.py                # exact Bayesian measurement and prediction
  env.py                   # HMMEnv, history buffer, Gym API, task contract

envs/mess3/
  model.py                 # MESS3 transition and emission definitions
  tasks/
    occupancy_control.py   # continuous tilt control and occupancy reward
    passive.py             # fixed dynamics; actions do not affect transitions
    state_guess.py         # guess the current hidden state
    future_state_guess.py  # delayed reward for predicting a future state
  solvers/                 # MESS3-specific analytic baselines
```

Add another task by adding another file under the domain's `tasks/` package.
Do not add a new mode branch to `HMMEnv`.

This layout is implemented as a hard cutover. The former MESS3-specific Gym
environments and the separate runtime, controller, observation-view, and
reward-component layers have been removed.

## Generic HMM model

`HMMModel` is immutable probability data:

```python
initial_distribution       # P(s_0)
transition_matrix          # P(s_{t+1} | s_t) before task control
emission_matrix            # P(raw_token_t | s_t)
```

The model validates dimensions, non-negativity, and normalization. State and
token cardinalities follow from the array shapes.

The model does not know about:

- Gym actions;
- controllable transitions;
- rewards;
- exponential tilting;
- transition KL;
- observation history;
- diagnostics.

Use unambiguous names. `initial_distribution` and `transition_matrix` are
different concepts; neither should be overloaded as `CONTROL_TRANSITION_MATRIX`.

## Exact belief

`belief.py` provides pure Bayesian operations and a small optional stateful
tracker.

An exact agent belief uses:

- the previous belief;
- the token actually visible to the agent and its likelihood;
- the transition matrix that was actually executed.

It must not use the true hidden state. Hidden state is used only to simulate
the process and, when requested, to evaluate calibration.

For delay zero, update in this order:

```text
predict through U_t -> measure visible token from s_{t+1}
```

For delay one:

```text
measure the newly delivered token from s_t -> predict through U_t
```

Belief tracking is disabled unless requested by the policy observation, a
task, or diagnostics.

## Generic HMM environment

`HMMEnv` is the only generic Gym environment. It owns:

- the current hidden state and raw token;
- state-transition, emission, and presentation RNG streams;
- the internal history buffer;
- optional exact belief;
- Gymnasium `reset()` and `step()`;
- policy observation construction;
- diagnostic `info` construction;
- one attached task object.

Simulation samples:

```text
s_{t+1} ~ transition_matrix[s_t]
raw_token_{t+1} ~ emission_matrix[s_{t+1}]
```

Belief prediction uses matrix multiplication, but sampled state transition
does not.

## Observation and history configuration

The internal history buffer stores enough decision records to satisfy the
largest requested offset. `ObservationConfig` independently selects:

- a visible-token history window, normally offset zero and depth one;
- an executed-action history window, normally offset zero and depth one;
- exact agent belief;
- explicitly privileged hidden state.

Set either history window to `None` to omit it. Offset zero means the token or
executed action available at the current decision. Features are flat and
grouped as newest-first token one-hots, newest-first encoded actions, belief,
then hidden-state one-hot. A task exposes `action_observation_space` so the
generic environment can construct exact per-feature bounds.

Observation delay shifts token delivery. At time zero, unavailable history is
zero-padded.

Presentation scrambling applies only to the token and previous-action features
shown to the policy. It must not mutate:

- the raw emitted token;
- the action executed by the task;
- the hidden trajectory.

Use a separate presentation RNG so enabling scrambling does not alter states
or emissions under the same seed and actions.

## Diagnostics

Diagnostics are returned through Gymnasium `info`; they are not silently added
to the policy observation. Flags select fields such as:

- reward components;
- current hidden state;
- agent-conditioned belief;
- raw-emission belief;
- raw and visible tokens;
- original and executed transition matrices;
- requested and executed actions.

Use explicit timing names:

- `state_before`;
- `state_after`;
- `state_current`;
- `raw_token_before`;
- `raw_token_after`;
- `visible_token_current`.

This prevents a reward calculated from `s_t` from being confused with a
returned state describing `s_{t+1}`.

## Task contract

A task owns action and reward semantics together, but they execute in two
phases because some rewards require the sampled next state.

Conceptually, every task provides:

```python
class HMMTask:
    action_space: gym.Space
    action_observation_space: gym.spaces.Box
    requires_belief: bool

    def reset(self) -> None:
        ...

    def resolve_action(
        self,
        action,
        state,
        model,
    ) -> ActionDecision:
        """Clip/interpret action and select the transition matrix."""

    def reward(
        self,
        event,
        decision,
    ) -> tuple[float, dict[str, float]]:
        """Score the completed before/after transition."""

    def encode_action(self, executed_action) -> np.ndarray:
        """Encode previous action when it is part of the observation."""
```

`ActionDecision` is a small record containing:

- requested and executed action;
- transition matrix to execute;
- optional task metadata.

The environment passes a completed event containing explicit before/after
states and tokens to `reward()`.

`requires_belief` requests agent-belief snapshots on that event. A stateful
task may additionally define `on_truncation()` for episode-boundary cleanup;
the generic environment invokes it after the final reward is evaluated.

Keep this as a small structural contract or protocol. Do not introduce a task
registry or a declarative task DSL.

## MESS3 task ownership

### Occupancy control

`tasks/occupancy_control.py` owns:

- continuous action space and clipping;
- exponential transition tilting;
- selected occupancy-reward states;
- optional transition-KL calculation;
- optional subtraction of transition KL from reward;
- transition-KL diagnostic metrics.

The reference transition law defaults to the HMM model's original transition
matrix but may be supplied explicitly by the task.

Transition KL is a task metric or control cost, not an HMM property and not
necessarily a Learner loss. If neither reward nor diagnostics request it, the
task may skip reporting it.

### Passive

`tasks/passive.py` always executes the model's original transition matrix. Its
action space and reward are explicit rather than hidden behind a `passive_mode`
branch.

### Current-state guess

`tasks/state_guess.py` uses a discrete action to guess `state_before`. The
action does not modify transitions.

### Future-state guess

`tasks/future_state_guess.py` owns the pending prediction queue and scores a
guess when its configured future state becomes available. It defines what
happens to unresolved predictions at episode truncation. The implemented task
discards them when truncation occurs.

## Step flow

At reset:

1. Seed independent RNG streams.
2. Sample `s_0` from `initial_distribution`.
3. Sample the raw token emitted by `s_0`.
4. Initialize delay/history and optional belief.
5. Return the first policy observation and selected diagnostics.

At step `t`:

1. Capture the current decision state aligned to `s_t`.
2. Ask the task to resolve `a_t` into an executed action and transition matrix
   `U_t`.
3. Sample `s_{t+1}` from `U_t[s_t]`.
4. Sample the next raw token from `emission_matrix[s_{t+1}]`.
5. Advance token delivery, history, and optional belief.
6. Give the completed transition event to the task's reward method.
7. Record the executed action and reward in history.
8. Build the policy observation aligned to `s_{t+1}`.
9. Return reward, truncation state, and explicitly timed diagnostics.

## Construction

An experiment selects:

- an `HMMModel` factory;
- one concrete task class;
- observation/history configuration;
- diagnostic configuration;
- episode length, optional first-episode desynchronization, and seed.

`HMMEnv` constructs and owns the task instance. Config values passed through
RLlib should remain primitive and serializable. Avoid lambdas, live component
instances, global arm registries, and hidden experiment schemas.

With multiple vector environments, equal fixed horizons make every environment
truncate on the same sampler step. Set `randomize_first_episode_length` to
sample the first horizon uniformly from `1..episode_length`; every later
episode uses the configured full length. This assigns each environment a
persistent phase offset while preserving the long-run episode definition. The
episode-length RNG is separate from state, emission, and presentation RNGs, so
enabling desynchronization does not change a seeded within-episode trajectory.
These children use explicit stable spawn keys rather than ordered
`SeedSequence.spawn()` calls. Add a new unique key for a new concern; never
renumber or repurpose the existing state, emission, presentation, or
episode-length keys.

Factories and task classes use ordinary import paths rather than a registry:

```python
env_config = {
    "model": {
        "factory": "envs.mess3.model:control_model",
        "kwargs": {"alpha": 0.85},
    },
    "task": {
        "class": (
            "envs.mess3.tasks.occupancy_control:"
            "OccupancyControlTask"
        ),
        "kwargs": {
            "action_limit": 5.0,
            "transition_kl_beta": 4.0,
        },
    },
    "observation": {
        "token": {"offset": 0, "depth": 1},
        "action": {"offset": 0, "depth": 1},
        "belief": False,
        "hidden_state": False,
        "token_scrambling": "none",
        "action_scrambling": "none",
    },
    "diagnostics": {
        "state": False,
        "belief": False,
        "raw_belief": False,
        "tokens": False,
        "rewards": False,
        "transitions": False,
    },
    "delay": 1,
    "episode_length": 1024,
    "randomize_first_episode_length": True,
    "seed": 42,
}
```

`transitions` diagnostics include explicit before/after values, the model's
original and executed transition matrices, an optional task reference matrix,
and the requested and executed actions.

## Performance

- Keep individual finite-HMM environments on CPU.
- Do not construct beliefs or copy diagnostic arrays when disabled.
- Avoid per-step callback chains and unnecessary intermediate objects.
- Keep one transition event only where timing clarity requires it.
- Fuse task calculations that share intermediates, such as tilted transitions
  and transition KL.
- Benchmark end-to-end RLlib sampling before adding lower-level complexity.

## Future composition

HMM composition is not required now. Keep `HMMModel` as ordinary validated
arrays so a future pure factory can produce product or otherwise composed
models without changing the task and environment contracts.