---
status: resolved
severity: high
area: devops/vast
discovered: 2026-08-18
reproduction: confirmed
---

# Vast self-destruct permits full runs without durable artifacts

## Context

- **Git revision / worktree:** `68c0b50`; fixed on
  `cursor/cassandra-targeted-actions-51f1`
- **Command:** `python -m devops.vast.provision up --branch <branch> --self-destruct --run "rl-harness <experiment>"`
- **Environment:** Vast RTX 4090
- **Related records:** Cassandra runs `20260818T063416Z-bfc0c2df` and
  `20260818T193443Z-e37dbc13`

## Expected behavior

A paid self-destructing run verifies durable artifact publication before
teardown or requires explicit `compact-only` acknowledgement.

## Observed behavior

Vast accepted full self-destruct runs without B2 forwarding. Compact Git
results survived, but checkpoints, Tune histories, and raw logs were destroyed.

## Minimal reproduction

Provision a successful full self-destruct run without B2 credentials, then
observe teardown without a completed durability manifest.

## Resolution history

- 2026-08-18 — Recorded after checkpoint loss prevented post-hoc evaluation.
- 2026-08-18 — Added required/compact-only modes, B2 preflight, and teardown
  verification; validated with a live durable smoke.
- 2026-08-20 — Hardened all defaults and missing/invalid mode handling to fail
  closed. Only explicit `compact-only` bypasses B2 verification. Durable smoke
  now rejects disabled upload. Moved to `resolved/`.
