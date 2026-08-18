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
`action_scope="targeted"` for a 10-action variant containing `operate`,
`inspect`, four component-specific repairs, and four component-specific
replacements. A targeted repair costs `0.75` and improves a bad or fair
component one level with probability `0.8`; good and broken components are
unchanged. A targeted replacement costs `3.75` and deterministically restores
its selected component to good from every condition.

`observation_mode="symbol"` returns the canonical `Discrete(16)` observation.
`operate` emits only `0` or `15`, `inspect` may emit any symbol, and the two
maintenance actions emit `0`. As in any POMDP, a policy using this mode needs
memory of prior observations and actions.

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
