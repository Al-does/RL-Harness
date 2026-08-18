---
status: open
severity: high
area: harness/runners.py
discovered: 2026-08-18
reproduction: confirmed
---

# Tune runs retain only the terminal metric snapshot in compact results

## Context

- **Git revision / worktree:** `68c0b50`; implementation in progress on
  `cursor/cassandra-targeted-actions-51f1`
- **Command:** `rl-harness experiments.cassandra_belief_factoring_2026_08.ppo.experiment --seed 42 --hardware-profile cuda4090`
- **Environment:** Python 3.14.7; Ray 2.56.0; RLlib 2.56.0
- **Training context:** experiment
  `experiments.cassandra_belief_factoring_2026_08.ppo.experiment`; seed 42;
  full; `cuda4090`
- **Related records:**
  `experiments/cassandra_belief_factoring_2026_08/ppo/results/20260818T063416Z-bfc0c2df/tune_summary.json`

## Expected behavior

Tune-managed runs preserve compact per-iteration scalar metrics so reward and
training curves survive removal of the ignored Tune artifact tree.

## Observed behavior

`run_tune()` writes only `result.metrics`, the terminal Tune snapshot, to
`tune_summary.json`. The global 5M terminal iteration completed zero episodes,
so its episode-return fields are `null`; earlier non-null returns existed only
in the destroyed `artifacts/tune/**/progress.csv`.

## Minimal reproduction

Run a Tune-managed RLlib experiment for at least two iterations, then inspect
`results/<run-id>/`. `tune_summary.json` contains only the terminal metrics and
`progress.jsonl` is absent, while Tune's ignored `progress.csv` contains every
iteration.

## Suspected cause and scope

`harness.runners.run_algorithm()` calls the generic result recorder every
iteration, but `run_tune()` never exports `Result.metrics_dataframe`. This
affects every Tune-managed experiment that does not implement its own compact
history export.

## Resolution history

- 2026-08-18 — Recorded from the two Cassandra seed-42 5M runs. A generic
  scalar-history export and focused regression coverage are in progress.
- 2026-08-18 — Fixed on `cursor/cassandra-targeted-actions-51f1`: Tune now
  writes every scalar history row to compact `progress.jsonl`. Validated by the
  full 330-test fast suite and live Vast smoke `infra-live-20260818`, whose two
  reward points survived Git publication.
