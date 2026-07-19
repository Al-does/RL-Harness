---
status: resolved
severity: high
area: devops/vast/self_destruct.py
discovered: 2026-07-19
reproduction: confirmed
---

# First push to a new Vast results branch is rejected

## Context

- **Git revision / worktree:** `019ebf5`; dirty fix in
  `devops/vast/self_destruct.py` and `tests/test_infra_safeguards.py`
- **Command:** `git push origin HEAD:results` from a detached Vast checkout when
  `refs/heads/results` does not yet exist
- **Environment:** Git on Ubuntu 22.04; Python 3.14.5
- **Related records:**
  `experiments/mess3_belief_geometry_2026_07/paper_supervised_replication/results/paper-sgd-compiled-retry-seed42-20260719/operations_summary.json`
  in `Al-does/alex-rl-experiments`

## Expected behavior

The first completed remote run should create the configured results branch and
push its compact result commit before teardown.

## Observed behavior

All six retries failed because Git could not infer the destination namespace:

```text
The destination you provided is not a full refname
Did you mean to create a new branch by pushing to
'HEAD:refs/heads/results'?
```

The result commit had to be pushed manually before the box self-destructed.

## Minimal reproduction

Create a detached commit, ensure the destination branch does not exist, then
run `git push origin HEAD:results`. Git rejects the abbreviated destination.
`git push origin HEAD:refs/heads/results` succeeds.

## Suspected cause and scope

`push_results()` used `HEAD:{branch}`. That form works when the remote branch
already exists but is ambiguous when creating it from a detached `HEAD`.
Every new results-branch deployment was affected.

## Resolution history

- 2026-07-19 — Recorded after the MESS3 result push failed six times.
- 2026-07-19 — Qualified the destination as `refs/heads/{branch}` and added a
  regression test for a missing remote branch.
