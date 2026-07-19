---
status: open
severity: high
area: agent orchestration / git worktree isolation
discovered: 2026-07-19
reproduction: confirmed
---

# Concurrent agent changed the active branch in a shared worktree

## Context

- **Git revision / worktree:** experiment worktree initially on
  `experiment/mess3-paper-replication`; no local changes before result import
- **Command:** `git cherry-pick origin/results`
- **Environment:** Cursor agents sharing one macOS worktree
- **Related records:** accidental local commit `5962e87` on
  `experiment/mess3-muon-comparison`; intended result commit `671a493` on
  `experiment/mess3-paper-replication`

## Expected behavior

An agent's assigned worktree and branch should remain stable for the duration of
its task, or concurrent mutation should be blocked before a write operation.

## Observed behavior

The branch changed from `experiment/mess3-paper-replication` to
`experiment/mess3-muon-comparison` between earlier status checks and a result
cherry-pick. Git therefore created the first imported-results commit on the
wrong branch. The commit was not pushed from that branch; the agent switched
back and applied the result to the intended branch.

## Minimal reproduction

1. Start in experiment worktree branch
   `experiment/mess3-paper-replication`.
2. Create and check out `fix/vast-agent-issues` in the sibling harness clone,
   push it, and record it as the active branch.
3. Use Cursor app control to move the agent root to the harness clone.
4. Run `git status` there. The harness clone is now on
   `experiment/mess3-paper-replication`, despite having been on the harness
   feature branch immediately before the root move.

## Suspected cause and scope

Multiple concurrent sessions appear able to mutate one worktree's `HEAD`.
Branch metadata in the agent UI does not lock Git state. Any check-then-write
sequence can therefore target a different branch. Separate worktrees per agent,
or a branch assertion immediately before mutating Git operations, would reduce
the risk.

## Resolution history

- 2026-07-19 — Recorded from the MESS3 result-import incident.
- 2026-07-19 — Reproduced during issue reporting: moving the agent root to the
  harness clone silently checked out the experiment repository's branch name.
