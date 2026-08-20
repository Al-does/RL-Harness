---
status: resolved
severity: medium
area: devops/vast
discovered: 2026-08-18
reproduction: confirmed
---

# Vast can use a detached commit SHA as the results branch

## Context

- **Git revision / worktree:** `68c0b50`; fixed on
  `cursor/cassandra-targeted-actions-51f1`
- **Command:** `python -m devops.vast.provision up --commit <sha> --self-destruct --run "<command>"`
- **Environment:** Vast RTX 4090

## Expected behavior

Self-destruct validates a real publication branch before renting. Detached
commit launches require an explicit `--results-branch`.

## Observed behavior

The publication destination previously defaulted to the clone ref, allowing a
40-character commit SHA to become an accidental branch destination.

## Minimal reproduction

Configure self-destruct with a detached commit and no valid publication branch.

## Resolution history

- 2026-08-18 — Recorded after manual Cassandra result recovery.
- 2026-08-18 — Added pre-rental branch validation and required explicit result
  branches for detached launches.
- 2026-08-20 — Added direct regression coverage for full-SHA rejection and
  valid branch acceptance; moved to `resolved/`.
