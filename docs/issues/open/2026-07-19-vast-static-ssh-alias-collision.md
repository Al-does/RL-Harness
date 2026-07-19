---
status: open
severity: high
area: devops/vast/terminals.py
discovered: 2026-07-19
reproduction: confirmed
---

# Concurrent Vast launches overwrite static SSH aliases

## Context

- **Git revision / worktree:** `019ebf5`; clean at discovery
- **Command:** run two `python -m devops.vast.provision up -n 1 --no-open ...`
  operations from concurrent agent sessions
- **Environment:** macOS 15.6; shared `~/.ssh/config.d/vast.conf`
- **Related records:** MESS3 run instance `45324078` was later observed running
  an unrelated experiment command

## Expected behavior

Each live rental should retain a stable, unique connection target so concurrent
agents cannot accidentally inspect, stop, or replace another run.

## Observed behavior

Every `up` invocation names its first box `vast-1` and rewrites the entire
managed `vast.conf`. A later launch therefore redirects `ssh vast-1` from an
existing live rental to the new box. During the MESS3 run, a previously observed
instance path and process were replaced by an unrelated workload after shared
alias use; direct instance metadata still described the original MESS3 command.

## Minimal reproduction

1. Launch one box and record the target of `ssh -G vast-1`.
2. Launch a second one-box job from another session.
3. Run `ssh -G vast-1` again.
4. The alias now points only to the second box; the first target was removed
   from `vast.conf`.

## Suspected cause and scope

Aliases are assigned per invocation (`vast-1..N`) rather than per instance or
run, and `write_ssh_config()` replaces the complete shared file. All concurrent
local agents and worktrees share that namespace. Stable aliases containing the
instance id or run slug, plus merge/prune behavior, would prevent cross-run
control-plane mistakes.

## Resolution history

- 2026-07-19 — Recorded after a paper-scale run was interrupted and the shared
  alias resolved to a different workload.
