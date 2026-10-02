---
status: open
severity: medium
area: Al-does/alex-rl-experiments/tests
discovered: 2026-10-02
reproduction: confirmed
---

# Experiment fast suite has 30 failures on its clean base

## Context

- Experiment revision: `68290b4c` (origin/main, clean detached worktree).
- Harness revision: `180bbcc` (origin/main, clean detached worktree).
- Environment: Python 3.14; locked experiment environment (Ray 2.56).
- Discovered during offline RockSample probing, which adds new adapters/tests
  and an environment-domain filter; it changes none of these old recipes.
- Related analysis PR: https://github.com/Al-does/alex-rl-experiments/pull/176.
  No training or remote infrastructure was started.

## Expected behavior

The existing fast-suite recipe and integration tests pass against the current
recipe definitions and shared runner behavior.

## Observed behavior

The complete experiment fast suite reports 875 passed, 30 failed, one skipped,
one deselected. All 30 failed test nodes also fail when imported from clean
experiment and harness base checkouts. The new RockSample adapter tests pass.

Failure clusters include:

- Cassandra and MESS3/two-factor model dictionary equality checks reject the
  added `grad_checkpointing: False` field.
- Gol recipe/runner assertions fail; including continuing-runner metrics
  storage assertions. Full failures need individual triage.
- Nonergodic MESS3 tests expect historical component parameters and a 10M-step
  budget, whereas recipes now contain different parameters and a 100M budget.
- The coarse-probe campaign finds two matching seed-42 result directories and
  raises `expected one run, got [...]`.

## Minimal reproduction

From a clean experiment checkout with the matching editable harness base:

```bash
uv run --group jax pytest -q -m 'not slow'
uv run pytest -q \
  tests/test_cassandra_best_critic_bptt64_250m.py \
  tests/test_mess3_reward_state_action_symmetry_cycle_6.py \
  tests/test_nonergodic_mess3_supervised.py \
  tests/test_nonergodic_mess3_token_guess_cycle_1.py \
  tests/test_belief_quotient_ladder_coarse_probes.py
```

For cross-worktree comparisons, activate the existing environment or invoke
its Python directly, with both clean worktrees on PYTHONPATH. `uv run --project`
pointing outside Ray's working directory introduces an unrelated worker-path
error; direct Python avoids that wrapper artifact.

## Suspected cause and scope

Several assertions demonstrably disagree with the current recipes; do not
blindly update tests, because this could conceal unintended recipe changes.
The recipe owner should decide which definitions are authoritative. Runner
failures and duplicate campaign inputs need separate investigation.
These failures do not involve RockSample or its Bayesian target update.

## Resolution history

- 2026-10-02 — Recorded after full feature-suite evaluation and reproduction
  of the same failed nodes on both clean base source trees. Left unchanged
  outside the belief-probing task.
