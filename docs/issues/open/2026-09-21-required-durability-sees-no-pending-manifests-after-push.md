---
status: open
severity: medium
area: devops/vast/self_destruct.py
discovered: 2026-09-21
reproduction: confirmed
---

# Required-durability check fails closed after per-seed result pushes

## Context

- **Git revision / worktree:** `79249be05ae4282885e1ecdc639654685ca0d0f3`; clean
- **Command:** `provision up` launched
  `python -m experiments.mess3_reward_state_action_symmetry_cycle_6.seed_queue
  --condition <leaf> --seeds 42 43 44 45 46 --target-agent-steps 10000000`
  with `--self-destruct --durability required --forward-b2`
- **Environment:** vast ondemand RTX 4090 boxes (`vast-51826844`,
  `vast-51827129`); Ray 2.56.0, Python 3.14.7
- **Training context:** full runs; both completed all seeds and pushed results
- **Related records:** `docs/issues/open/2026-09-12-vast-publication-test-ignore-mock.md`
  (same file, different defect — that one is test mocks)

## Expected behavior

After the run command exits 0, `self_destruct` verifies B2 durability and
destroys the box.

## Observed behavior

Both boxes printed:

```text
[self_destruct] required durability found no pending run manifest
[self_destruct] no new compact experiment results to push
[self_destruct] compact results pushed, but required B2 durability was not verified; preserving box for recovery until the max-age cap
```

and stayed alive despite every `run_manifest.json` recording
`status: completed` and `remote_artifacts.status: completed`, with
`remote_artifacts.json`/`durability_manifest.json` present on disk. Boxes were
destroyed manually after verification; cost was bounded by `--max-age`.

## Minimal reproduction

`seed_queue` calls `devops.vast.self_destruct.push_results` after each seed
(`--push-each`-style flow), which git-commits the results. When
`run_remote.sh` later invokes `self_destruct`, `pending_run_manifests()` lists
only *untracked* files under `experiments/**/results/**/run_manifest.json`.
All manifests are already committed, so the list is empty and
`required_durability_completed` returns `False` before ever checking
`remote_artifacts.status`.

Any experiment that pushes compact results itself before `self_destruct` runs
(seed queues, per-seed `--push-each`) hits this. Only runs that leave all
manifests uncommitted until teardown actually exercise the durability check.

## Suspected cause and scope

`required_durability_completed` conflates "pending" (untracked) with
"needs verification". When manifests were committed by an earlier
`push_results`, the check should either look at manifests changed/committed on
the launch ref for this run or treat "no pending manifests" as durable (nothing
left at risk) rather than not-verified. Scope: all self-destructing runs whose
run command commits results before teardown; B2 upload itself works fine.

## Resolution history

- 2026-09-21 — Recorded. Evidence on both launch-branch results and on-box
  `run.log`; manifests verified `completed`/`completed` for all 10 runs.
