---
status: resolved
severity: high
area: harness/runners.py
discovered: 2026-08-18
reproduction: confirmed
---

# Tune runs retain only the terminal metric snapshot in compact results

## Context

- **Git revision / worktree:** `68c0b50`; fixed on
  `cursor/cassandra-targeted-actions-51f1`
- **Command:** `rl-harness experiments.cassandra_belief_factoring_2026_08.ppo.experiment --seed 42 --hardware-profile cuda4090`
- **Environment:** Python 3.14.7; Ray 2.56.0; RLlib 2.56.0
- **Related record:**
  `experiments/cassandra_belief_factoring_2026_08/ppo/results/20260818T063416Z-bfc0c2df/tune_summary.json`

## Expected behavior

Tune-managed runs preserve compact per-iteration scalar metrics so reward and
training curves survive removal of the ignored Tune artifact tree.

## Observed behavior

`run_tune()` wrote only the terminal `result.metrics` snapshot. When the final
reporting window completed no episodes, compact episode-return fields were
`null`; earlier returns existed only in the ephemeral Tune tree.

## Minimal reproduction

Run a Tune-managed experiment for at least two iterations and compare the
ignored Tune `progress.csv` with compact `results/<run-id>/`.

## Resolution history

- 2026-08-18 — Recorded from two Cassandra seed-42 5M runs.
- 2026-08-18 — `run_tune()` now exports every scalar history row to compact
  `progress.jsonl`; validated by the fast suite and live Vast smoke.
- 2026-08-20 — Moved to `resolved/` after regression coverage and repeated
  durable full runs confirmed the compact history.
