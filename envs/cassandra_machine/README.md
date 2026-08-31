# Cassandra machine maintenance

This package implements Anthony Cassandra's canonical four-component
machine-maintenance POMDP. It is the `machine.POMDP` benchmark from the
[POMDP Example Domains](https://pomdp.org/examples/index.html), not an Apache
Cassandra database simulator.

## Canonical model

The hidden state is the Cartesian product of four independently evolving
components. Each component has four ordered conditions:

```text
0 broken < 1 bad < 2 fair < 3 good
```

The original model has 256 states, 4 actions, 16 observations, a fixed
all-good start state, and discount `0.999`.

| Action | Transition and observation | Reward |
|---|---|---|
| `operate` | Each non-broken component degrades one level with probability `0.03`; emits `15` when the resulting product passes and `0` when it fails. Per-component pass probabilities are `[0, 0.75, 0.95, 1]`, and every component must pass. | Product of the expected post-degradation component pass values `[0, 0.7275, 0.944, 0.9985]` |
| `inspect` | State is unchanged; emits one noisy binary reading per component, with positive probabilities `[0.02, 0.05, 0.80, 0.97]` | `-1` |
| `repair` | Each bad or fair component improves one level with probability `0.8`; broken components remain broken | `-3` |
| `replace` | Restores every component to good | `-15` |

The source file names actions numerically. The semantic names above follow
their transition, observation, and reward behavior. Joint state and
observation indices use component 0 as the least-significant base-four digit
or binary bit.

The implementation constructs the joint model from the one-component factors
instead of checking in a 10,000-line expanded probability table.

## Gymnasium use

```python
from envs.cassandra_machine import CassandraMachineEnv

env = CassandraMachineEnv(
    {
        "episode_length": 1000,
        "observation_mode": "symbol",
        "action_scope": "global",
        "diagnostics": False,
        "seed": 42,
    }
)
```

`action_scope="global"` is the canonical default. Set
`action_scope="global_aliases"` for a 10-action cardinality control containing
`operate`, `inspect`, four exact aliases of canonical global repair, and four
exact aliases of canonical global replacement. Each repair alias affects all
components and costs `3`; each replacement alias restores all components and
costs `15`.

Set `action_scope="targeted"` for a 10-action variant containing `operate`,
`inspect`, four component-specific repairs, and four component-specific
replacements. A targeted repair costs `0.75` and improves a bad or fair
component one level with probability `0.8`; good and broken components are
unchanged. A targeted replacement costs `3.75` and deterministically restores
its selected component to good from every condition.

The canonical initial state is all-good. Set
`initial_state_distribution="uniform"` to sample each component independently
and uniformly from `(broken, bad, fair, good)` at reset. The exact initial
Bayesian belief is then uniform over all 256 joint states.

`observation_mode="symbol"` returns the canonical `Discrete(16)` observation.
`operate` emits only `0` or `15`, `inspect` may emit any symbol, and the two
maintenance actions emit `0`. As in any POMDP, a policy using this mode needs
memory of prior observations and actions.

Set `observation_mode="state"` for the fully observable diagnostic MDP. The
policy receives the current joint component state as a `Discrete(256)` value
before every decision, using the same state encoding documented above. This
single flag composes with either 10-action comparison:

```python
# Fully observable global-alias control.
{"action_scope": "global_aliases", "observation_mode": "state"}

# Fully observable targeted maintenance.
{"action_scope": "targeted", "observation_mode": "state"}
```

Transitions, rewards, action costs, and the selected action scope are
unchanged; only the information available to the policy differs. This makes
the mode useful as a training diagnostic: failure to learn a strong policy
cannot be attributed to partial observability or memory alone. It does not,
by itself, prove that the achieved return is globally optimal.

`observation_mode="belief"` returns the exact Bayesian belief as a `Box(256)`.
`observation_mode="factored_belief"` returns its exact four component
marginals as a flat `Box(16)`. The global product-quality observation couples
components, so the marginals are a compact, information-losing view rather
than a factorization of the joint posterior. This corresponds to the
decomposed-belief representation studied by Wiering and Schmidhuber in
[Reinforcement Learning Using Approximate Belief States](https://papers.nips.cc/paper_files/paper/1999/file/158fc2ddd52ec2cf54d3c161f2dd6517-Paper.pdf).

Set `diagnostics=True` to place privileged component states, state indices,
beliefs, and reward decomposition in `info`. Diagnostics never enter the
policy observation implicitly.
