# Checkpoint strategy for longitudinal analysis

## The problem

MESS3 analyses need more than a final trained policy:

- **N-init** measures probe geometry before any optimization.
- **Training curves** show when representation geometry appears relative to
  reward learning.
- **Null brackets** compare trained representations with initialization and
  trained-on-noise baselines.

The current Tune recipes typically checkpoint every ten iterations, retain
only three checkpoints, and save at the end. This is adequate for basic
recovery but not for longitudinal analysis:

1. Tune pruning removes early checkpoints.
2. Tune does not normally create a step-zero checkpoint.
3. Raising `num_to_keep` cannot recover a checkpoint that was never written.
4. Keeping every full Algorithm checkpoint can consume significant temporary
   disk and checkpoint time because optimizer and framework state are included.
5. On self-destructing remote machines, ignored artifacts disappear before
   later analysis can use them.

## Recommended fix

Use two stages.

### 1. Measure the cost with one representative run

Temporarily checkpoint every training iteration and disable pruning:

```python
tune.CheckpointConfig(
    checkpoint_frequency=1,
    num_to_keep=None,
    checkpoint_at_end=True,
)
```

Record:

- bytes per full checkpoint;
- checkpoint wall time;
- expected checkpoint count;
- peak disk use and safe free-space margin.

A 10-million-step run with a 32,768-step train batch is roughly 306 training
iterations, so keeping every full checkpoint may be acceptable or may be
wasteful depending on Algorithm state size. Remote teardown removes durability
cost, but not disk exhaustion or save-time overhead during the run.

### 2. Adopt a log-spaced analysis schedule

For routine research runs, retain checkpoints at:

```text
step 0, iterations 1, 2, 4, 8, 16, ... , final
```

This reproduces the useful early-training resolution with logarithmic storage.
The experiment decides the schedule because checkpoint cadence is part of its
analysis plan. A generic helper or callback may implement the mechanics.

Requirements:

- create the step-zero checkpoint immediately after Algorithm initialization
  and before the first `train()` call;
- use public RLlib checkpoint APIs such as `algorithm.save_to_path(...)`;
- never access private Learner/module internals;
- record iteration and sampled environment steps beside every checkpoint;
- keep checkpoints under `context.artifacts_dir`;
- keep at least one standard Tune/final checkpoint suitable for resume.

Tune's fixed `checkpoint_frequency` cannot express a log-spaced schedule by
itself. Implement the schedule through a documented RLlib/Tune callback or a
small generic runner extension, while keeping the chosen cadence in the
experiment recipe.

## Remote execution order

Ephemeral boxes must finish analysis before teardown:

1. train and write checkpoints under `artifacts/`;
2. probe the ordered checkpoint sequence;
3. write compact `probe_curve.json`, null-bracket summaries, tables, and
   figures under `results/`;
4. verify the compact outputs;
5. push results and destroy the machine.

Do not self-destruct immediately after training when a later checkpoint-based
analysis is required.

## Validation

Add coverage for:

- a real step-zero checkpoint;
- ordered log-spaced checkpoint labels and environment-step metadata;
- public checkpoint restoration;
- final resume behavior;
- compact analysis output produced before teardown;
- cleanup when training or analysis fails.

Until the log-spaced mechanism exists, `checkpoint_frequency=1` with no pruning
is the safest temporary setting for runs whose artifacts will be analyzed on
the same machine before destruction.
