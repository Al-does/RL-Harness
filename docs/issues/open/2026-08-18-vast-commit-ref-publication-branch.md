---
status: open
severity: medium
area: devops/vast
discovered: 2026-08-18
reproduction: confirmed
---

# Vast can use a detached commit SHA as the results branch

## Context

- **Git revision / worktree:** `68c0b50`; implementation in progress on
  `cursor/cassandra-targeted-actions-51f1`
- **Command:** `python -m devops.vast.provision up --commit <sha> --self-destruct --run "<command>"`
- **Environment:** Vast RTX 4090
- **Related records:** Cassandra global seed-42 partial run
  `20260818T052440Z-7e95c8c2`

## Expected behavior

Self-destruct validates a real publication branch before renting. Detached
commit launches require an explicit `--results-branch`.

## Observed behavior

The publication destination defaults to `results_branch or ref`. When `ref` is
a full commit SHA, self-destruct attempts to push to a branch named after that
SHA. The first Cassandra remote run required manual result recovery.

## Minimal reproduction

Build a Vast environment with a 40-character commit ref, self-destruct enabled,
and no results branch. Inspect `VAST_PUBLISH_BRANCH`; it equals the commit SHA.

## Suspected cause and scope

`devops.vast.provision.build_env()` does not distinguish clone refs from valid
result branch destinations. Any detached-HEAD launch can publish to an
accidental SHA-named branch.

## Resolution history

- 2026-08-18 — Recorded with confirmed Cassandra publication evidence. A
  pre-rental branch validation guard and regression test are in progress.
