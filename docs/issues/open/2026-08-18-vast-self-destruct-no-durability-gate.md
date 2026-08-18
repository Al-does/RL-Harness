---
status: open
severity: high
area: devops/vast
discovered: 2026-08-18
reproduction: confirmed
---

# Vast self-destruct permits full runs without durable artifacts

## Context

- **Git revision / worktree:** `68c0b50`; implementation in progress on
  `cursor/cassandra-targeted-actions-51f1`
- **Command:** `python -m devops.vast.provision up --branch cursor/cassandra-belief-probe-51f1 --self-destruct --run "rl-harness experiments.cassandra_belief_factoring_2026_08.ppo.experiment --seed 42 --hardware-profile cuda4090"`
- **Environment:** Vast RTX 4090; full non-smoke run
- **Training context:** Cassandra global and targeted PPO; seed 42; `cuda4090`
- **Related records:** Cassandra run IDs `20260818T063416Z-bfc0c2df` and
  `20260818T193443Z-e37dbc13`

## Expected behavior

A paid self-destructing run either verifies durable artifact publication before
teardown or requires an explicit operator acknowledgement that checkpoints and
raw Tune history will be discarded.

## Observed behavior

Vast accepted full self-destruct runs without `--forward-b2`. Compact Git
results survived, but all checkpoints, Tune histories, and raw logs vanished
when the boxes were destroyed. The run manifests contain no completed
`remote_artifacts` summary.

## Minimal reproduction

Provision any successful full run with `--self-destruct` and without
`--forward-b2`. After compact result push, the box is destroyed even though no
B2 durability manifest exists.

## Suspected cause and scope

B2 forwarding is opt-in and self-destruct gates only on compact Git publication.
This affects every Vast run where later checkpoint restoration or post-hoc
analysis is important.

## Resolution history

- 2026-08-18 — Recorded after checkpoint loss prevented final-policy
  reevaluation and reward-curve recovery. Required/compact-only durability
  modes and teardown verification are in progress.
