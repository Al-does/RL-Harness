---
name: runpod-provisioning
description: Launch, monitor, persist, cost-account, and reap on-demand RunPod Community Cloud Pods for RL experiments through devops/runpod/pods. Use when the user asks to run training on RunPod, provision a cloud GPU Pod, execute a remote GPU smoke run, retrieve RunPod checkpoints, inspect costs, or clean up RunPod Pods.
---

# RunPod Pods provisioning

Use `devops/runpod/pods/` for batch RL training on ordinary RunPod Pods. This
backend is separate from RunPod Serverless. Batch jobs do not require SSH;
interactive profiling does. Read `devops/runpod/pods/README.md` and
`docs/runpod_execution.md` before changing or operating it.

Shared phased execution helpers live in `devops/runpod/execution/` (preflight,
resource contracts, clean-worktree results publication, B2 durability, portable
checkpoint guidance). Serverless may fall back here with `--fallback pods`.

## Safety rules

- Always run the identical launch with `--dry-run` before creating a Pod.
- Exact commit SHAs are verified fetchable from GitHub before Pod create.
- Create only Community Cloud, on-demand Pods. Successful placement must report
  `podType=RESERVED`, `secureCloud=False`, and an allowed GPU.
- Pods accept one exact GPU type per request. Use `--gpu-type` to choose a
  configured 24-GB equivalent (L4, A5000, or RTX 3090) when a 4090 is
  unavailable; never weaken memory, CUDA, price, or placement verification.
- Keep a positive `--max-age`; never bypass provider `terminateAfter`.
- Use `--forward-b2` when checkpoints must survive teardown. Community Cloud
  cannot attach RunPod network volumes. Compact JSON/plots are uploaded too;
  retrieve via `durability_manifest.json`.
- Never print raw Pod metadata or credentials. Use `inspect`, which redacts
  environment secrets.
- Every job self-terminates on success and failure. Still confirm cleanup with
  `status`; use `reap --yes` for managed failed, orphaned, or over-age Pods.
- An agent must remain responsible after submission: follow status/logs until
  verified durable success and cleanup or a diagnosed terminal failure. Never
  stop merely because the remote command is still running; recover interrupted
  monitoring with `status`.
- Interactive Pods are the exception to job-completion teardown: they wait for
  manual destroy while provider `terminateAfter` and the watchdog enforce a
  positive hard ceiling.
- Compact Git results publication overlays onto the `results` branch from a
  clean worktree; it never rebases experiment history and must not change
  workload success.
- Do not use this workflow for Serverless endpoints. Serverless belongs under
  `devops/serverless/`.

## Prerequisites

Required process or Cursor Runtime Secrets:

- `RUNPOD_API_KEY`
- `GH_TOKEN`

For durable artifacts, also configure `B2_BUCKET`, `B2_ENDPOINT`,
`B2_APPLICATION_KEY_ID`, and `B2_APPLICATION_KEY`.

Batch jobs need no SSH key. Interactive mode automatically uses
`~/.ssh/id_ed25519(.pub)` or `~/.ssh/id_rsa(.pub)`; override with `--ssh-key`
or `RUNPOD_SSH_KEY_PATH`. Only the public key is injected. Cursor Cloud
bootstrap creates an RSA pair, so do not put private keys in dashboard secrets.

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
estimate. If preflight rejects a nonexistent SHA, fix the ref before `--yes`.
If acceptable, repeat the same command with `--yes` instead of `--dry-run`.

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

# Snapshot or follow the documented v2 SSE logs
uv run python -m devops.runpod.pods.provision logs POD_ID --tail 100
uv run python -m devops.runpod.pods.provision logs POD_ID \
  --source container --follow

# Terminate one known Pod
uv run python -m devops.runpod.pods.provision destroy --id POD_ID --yes

# Reap managed failed, orphaned, or over-age Pods
uv run python -m devops.runpod.pods.provision reap --yes
```

RunPod billing aggregation is delayed. Report posted actual cost separately
from the capped estimate for pending records, and run `status` again later.
Before finishing, verify that no managed live Pods remain.

## Interactive CUDA workspace

Use interactive mode for terminal debugging and profiling. It does not accept
`--run`; connect and execute commands manually.

```bash
uv run python -m devops.runpod.pods.provision up \
  --interactive --max-age 2 --max-price 0.50 --dry-run

# Repeat with --yes, then:
uv run python -m devops.runpod.pods.provision ssh POD_ID
uv run python -m devops.runpod.pods.provision destroy --id POD_ID --yes
```

The Pod exposes only TCP 22, runs key-only SSH, and defaults to a 2-hour hard
ceiling. Launch and connect from the same Cursor Cloud Agent because its SSH
keypair is VM-specific.

## Success evidence

For a validation run, record:

1. `RESERVED`, Community placement and actual GPU.
2. Checked-out experiment and harness SHAs (preflight-verified when exact).
3. CUDA availability plus completed training steps.
4. Artifacts **and** compact results uploaded to B2 and hash-verified after
   Pod deletion.
5. MLflow tags for both SHAs and the immutable image digest.
6. Automatic Pod deletion and posted or clearly labeled estimated cost.
7. Publication status reported separately from workload success.
