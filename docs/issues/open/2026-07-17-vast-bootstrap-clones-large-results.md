---
status: open
severity: high
area: "devops/vast/bootstrap.sh and root results/"
discovered: 2026-07-17
reproduction: confirmed
---

# Vast bootstrap downloads 430 MiB of unrelated tracked results

## Context

- **Git revision / worktree:** `d91bdffc7aa6839a8bd45161e2b03920c270a5f5`
  on `MESS3-supervised`; dirty in pre-existing workspace files and the three
  issue records created during this run
- **Command:** `uv run --group devops python -m devops.vast.provision up -n 1
  --branch MESS3-supervised --run "rl-harness
  experiments.MESS3_supervised.experiment --seed 42 --hardware-profile
  cuda4090" --self-destruct --yes`
- **Environment:** local macOS provisioner; multiple Vast RTX 4090 Ubuntu/CUDA
  hosts
- **Training context:** experiment
  `experiments.MESS3_supervised.experiment`; seed `42`; full mode; hardware
  profile `cuda4090`; training never started before the workaround
- **Related records:**
  `docs/issues/open/2026-07-17-vast-cannot-exclude-bad-machines.md`;
  no run manifest existed yet because bootstrap had not completed

## Expected behavior

Remote bootstrap should transfer the source, lockfile, experiment recipes, and
other files needed to execute the selected Git ref. Historical result payloads
that are unrelated to the requested experiment should not consume most of the
bootstrap network and indexing budget.

## Observed behavior

The current revision contains 858 tracked files under root `results/` totaling
430.4 MiB:

```bash
git ls-tree -r --long HEAD results |
  awk '{sum += $4; count += 1}
       END {printf "%d files, %.1f MiB\n", count, sum/1048576}'
# 858 files, 430.4 MiB
```

`bootstrap.sh` runs an unfiltered `git clone` followed by
`git fetch --all --tags`. On two running Vast hosts, `git clone` or
`git index-pack` remained active for more than 15 minutes. A shallow clone
still had to materialize the 430 MiB current results tree and remained in
`index-pack` after several minutes.

On the retained host, replacing that operation with:

```bash
git clone --depth 1 --filter=blob:none --sparse --single-branch \
  --branch MESS3-supervised <repo-url> /root/RLLibHarnessBeta
git -C /root/RLLibHarnessBeta sparse-checkout set \
  .cursor analysis devops docs envs experiments harness learners losses tests
```

completed the clone and initial checkout in approximately six seconds. The
normal environment synchronization then began. `experiments/` remains in the
sparse worktree, so the existing result-push/self-destruct contract can still
commit compact experiment outputs.

## Minimal reproduction

1. Confirm the current payload:

   ```bash
   git ls-tree -r --long HEAD results |
     awk '{sum += $4; count += 1}
          END {printf "%d files, %.1f MiB\n", count, sum/1048576}'
   ```

2. Provision any clean Vast host through `devops.vast.provision up`.
3. Observe `/root/bootstrap.log` and
   `ps -eo pid,etime,cmd` while the unfiltered clone indexes the repository.
4. Compare with the partial sparse-clone commands above on the same host.

## Suspected cause and scope

The bootstrap assumes the entire Git tree and history are cheap to clone, but
the repository still tracks a large legacy root `results/` tree. This affects
every Vast experiment, even though current experiment outputs live under
`experiments/<leaf>/results/`.

The immediate infrastructure fix is a blob-filtered sparse clone that includes
all executable packages and `experiments/` while excluding root `results/`,
and that avoids a subsequent full `fetch --all --tags`. Independently, the
430 MiB legacy result tree should be reviewed against the repository's compact
results/artifact policy and moved to appropriate durable storage if those
payloads are not intended to remain in every source checkout.

## Resolution history

- 2026-07-17 — Recorded after a sparse blobless checkout reduced live-host
  clone time from repeated multi-minute stalls to approximately six seconds.
