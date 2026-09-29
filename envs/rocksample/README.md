# RockSample

This package implements the RockSample POMDP introduced by Smith and Simmons
(UAI 2004). The rover crosses an `n × n` grid, checks the hidden Good/Bad
quality of known rocks through a distance-dependent noisy sensor, samples
valuable rocks, and exits through the east edge.

The default is the fixed RockSample[5,7] benchmark: `n=5`, `k=7`,
`randomize_train_layout=False`. The benchmark discount is `0.95`; experiments
should set that discount on their algorithm config.

## Configuration

```python
from envs.rocksample import RockSampleEnv

env = RockSampleEnv(
    {
        "n": 5,
        "k": 7,
        "randomize_train_layout": False,
        "evaluation": False,
        "episode_length": 100,
        "eval_layout_seed": 0,
        "diagnostics": False,
        "seed": None,
    }
)
```

Training uses the fixed canonical layout by default. Set
`randomize_train_layout=True` to sample held-out training layouts. Evaluation
environments should set `evaluation=True`; they use the fixed evaluation
layout regardless of `randomize_train_layout`.

Every randomized training episode samples `k` distinct cells uniformly,
excluding the fixed start cell. The evaluation layout is rejected even if its
cells are drawn in a different rock-index order. Evaluation is therefore on a
held-out layout rather than a layout that training could memorize.

Calling `reset()` advances the existing random streams rather than reseeding
them, so consecutive episodes receive fresh qualities, layouts when enabled,
and sensor draws. Passing `reset(seed=...)` explicitly restarts those streams.

The default step cap is `episode_length=100`. Reaching it truncates the
episode. Exiting east from the last column terminates the episode and pays
`+10`.

## Evaluation layouts

The standard layouts below use the original ZMDP rock order. The layouts
through [10,10] match the Smith and Simmons 2004 benchmark definitions;
[11,11] is the standard DESPOT extension. Coordinates are `(x, y)`, North is
`+y`, and `d0` is the sensor's half-efficiency distance.

| Instance | Start | Rocks in index order | `d0` |
|---|---:|---|---:|
| [4,4] | (0,2) | (3,1), (2,1), (1,3), (1,0) | ln 2 |
| [5,5] | (0,2) | (2,4), (0,4), (3,3), (2,2), (4,1) | 4 |
| [5,7] | (0,2) | (1,0), (2,1), (1,2), (2,2), (4,2), (0,3), (3,4) | 20 |
| [7,8] | (0,3) | (2,0), (0,1), (3,1), (6,3), (2,4), (3,4), (5,5), (1,6) | 20 |
| [10,10] | (0,5) | (0,3), (0,7), (1,8), (3,3), (3,8), (4,3), (5,8), (6,1), (9,3), (9,9) | 20 |
| [11,11] | (0,5) | (0,3), (0,7), (1,8), (2,4), (3,3), (3,8), (4,3), (5,8), (6,1), (9,3), (9,9) | 20 |

For any non-standard `(n, k)`, the start is `(0, floor(n/2))`, `d0=20`, and
the fixed evaluation layout is drawn without replacement from all non-start
cells using `eval_layout_seed`. Its default value is `0`; record any override
with results.

## Actions and observations

Actions are indexed as:

```text
0 North, 1 South, 2 East, 3 West, 4 Sample, 5+i Check_i
```

The sensor efficiency for `Check_i` is `2 ** (-distance / d0)`. It reports the
true quality with probability `(1 + efficiency) / 2`. Sampling a Good rock
pays `+10` and makes it Bad; sampling a Bad rock pays `-10`.

The policy observation is a flat `float32` vector:

```text
[rover_x, rover_y,
 rock_0_x, rock_0_y, ..., rock_(k-1)_x, rock_(k-1)_y,
 is_Good, is_Bad, is_None,
 prev_North, prev_South, prev_East, prev_West, prev_Sample,
 prev_Check_0, ..., prev_Check_(k-1)]
```

Coordinates are divided by `n-1`. Rock positions are always included, in
rock-index order, during randomized training, fixed-layout training, and
evaluation. The previous action is one-hot encoded; its block is all zeros on
reset. Move and Sample actions emit `None`; only Check actions emit `Good` or
`Bad`.

## Illegal actions and evaluation diagnostics

This implementation follows RockSample.jl rather than ZMDP for illegal
actions. North, South, or West into a wall leaves the rover in place with
reward `0`. Sampling an empty cell also leaves the state unchanged with reward
`0`. The episode continues. These choices do not change the optimal value
because an illegal action is always worse than proceeding east, but they are
easier for a learning agent than ZMDP's `-100` and termination.

Every step's `info` contains `illegal_action`, `illegal_action_count`, and
`illegal_action_rate`. On the final evaluation step,
`illegal_action_rate` is the episode fraction of wall bumps plus empty-cell
Sample actions and should be reported alongside return.

Set `diagnostics=True` to also expose hidden `rock_qualities` in `info`.
Privileged qualities never enter the policy observation.
