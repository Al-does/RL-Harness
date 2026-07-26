---
name: runpod-provisioning
description: Launch, monitor, persist, cost-account, and reap on-demand RunPod Community Cloud Pods for RL experiments through devops/runpod/pods. Use when the user asks to run training on RunPod, provision a cloud GPU Pod, execute a remote GPU smoke run, retrieve RunPod checkpoints, inspect costs, or clean up RunPod Pods.
---

# RunPod Pods provisioning

Use `devops/runpod/pods/` for batch RL training on ordinary RunPod Pods. This
backend is separate from RunPod Serverless and does not require SSH. Read
`devops/runpod/pods/README.md` before changing or operating it.

## Safety rules

- Always run the identical launch with `--dry-run` before creating a Pod.
- Create only Community Cloud, on-demand Pods. Successful placement must report
  `podType=RESERVED`, `secureCloud=False`, and an allowed GPU.
- Keep a positive `--max-age`; never bypass provider `terminateAfter`.
- Use `--forward-b2` when checkpoints must survive teardown. Community Cloud
  cannot attach RunPod network volumes.
- Never print raw Pod metadata or credentials. Use `inspect`, which redacts
  environment secrets.
- Every job self-terminates on success and failure. Still confirm cleanup with
  `status`; use `reap --yes` for managed failed, orphaned, or over-age Pods.
- Do not use this workflow for Serverless endpoints. Serverless belongs under
  `devops/serverless/`.

## Prerequisites

Required process or Cursor Runtime Secrets:

- `RUNPOD_API_KEY`
- `GH_TOKEN`

For durable artifacts, also configure `B2_BUCKET`, `B2_ENDPOINT`,
`B2_APPLICATION_KEY_ID`, and `B2_APPLICATION_KEY`. Do not create or request an
SSH keypair.

The experiment and harness refs must exist on GitHub before launch. The public,
digest-pinned GHCR image contains the environment and runner only; experiment
and harness source are cloned at startup.

## Launch workflow

Run from the harness or its editable-dependent experiment checkout:

```bash
uv run python -m devops.runpod.pods.provision up \
  --commit EXPERIMENT_SHA \
  --library-commit HARNESS_SHA \
  --run-name RUN_NAME \
  --max-age 1 \
  --max-price 0.50 \
  --run "rl-harness experiments.study.condition.experiment --smoke --upload-artifacts --run-id RUN_NAME" \
  --forward-b2 --self-destruct --dry-run
```

Review the resolved refs, image digest, GPU request, hourly price, and hard-cap
estimate. If acceptable, repeat the same command with `--yes` instead of
`--dry-run`.

The runner validates pinned Ray/Torch/Gymnasium versions and
`torch.cuda.is_available()` before training. A Community host can occasionally
advertise a GPU without exposing CUDA; this must fail and self-terminate. Retry
on a new Pod rather than weakening the check.

## Observe and clean up

```bash
# Managed live Pods, completed Pods, and posted/estimated costs
uv run python -m devops.runpod.pods.provision status

# Safe redacted metadata
uv run python -m devops.runpod.pods.provision inspect POD_ID

# Terminate one known Pod
uv run python -m devops.runpod.pods.provision destroy --id POD_ID --yes

# Reap managed failed, orphaned, or over-age Pods
uv run python -m devops.runpod.pods.provision reap --yes
```

RunPod billing aggregation is delayed. Report posted actual cost separately
from the capped estimate for pending records, and run `status` again later.
Before finishing, verify that no managed live Pods remain.

## Success evidence

For a validation run, record:

1. `RESERVED`, Community placement and actual GPU.
2. Checked-out experiment and harness SHAs.
3. CUDA availability plus completed training steps.
4. A checkpoint uploaded to B2 and downloaded after Pod deletion.
5. MLflow tags for both SHAs and the immutable image digest.
6. Automatic Pod deletion and posted or clearly labeled estimated cost.
