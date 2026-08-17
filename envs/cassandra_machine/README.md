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
| `operate` | Each non-broken component degrades one level with probability `0.03`; emits the null symbol | Product of per-component quality values `[0, 0.7275, 0.944, 0.9985]` |
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
        "diagnostics": False,
        "seed": 42,
    }
)
```

`observation_mode="symbol"` returns the canonical `Discrete(16)` observation.
Only `inspect` can emit a nonzero symbol. As in any POMDP, a policy using this
mode needs memory of prior observations and actions.

`observation_mode="factored_belief"` returns the exact four component
marginals as a flat `Box(16)` observation. Factorization is exact for this
model, reducing the full 256-state belief to 16 values. This is the
decomposed-belief representation evaluated by Wiering and Schmidhuber in
[Reinforcement Learning Using Approximate Belief States](https://papers.nips.cc/paper_files/paper/1999/file/158fc2ddd52ec2cf54d3c161f2dd6517-Paper.pdf).

Set `diagnostics=True` to place privileged component states, state indices,
beliefs, and reward decomposition in `info`. Diagnostics never enter the
policy observation implicitly.
