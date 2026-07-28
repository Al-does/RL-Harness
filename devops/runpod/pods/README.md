# RunPod Pods backend

This backend runs ordinary long-lived training commands on **RunPod Pods**. It
does not use Serverless. Shared preflight, durability, and results-publication
helpers live in `devops/runpod/execution/`; see `docs/runpod_execution.md`.
It mirrors the Vast CLI lifecycle:

```text
up (default) | status | inspect | destroy | reap
```

Run it from either the harness or experiment checkout:

```bash
uv run python -m devops.runpod.pods.provision ...
```

The Vast backend remains unchanged at `devops.vast.provision`.

## Safety policy

Every created Pod is:

- explicitly `cloudType=COMMUNITY`;
- created with RunPod's on-demand-only `podFindAndDeployOnDemand` mutation;
- restricted to an RTX 4090 or the configured similar GPUs;
- rejected and immediately terminated unless the returned Pod metadata proves
  Community placement, `podType=RESERVED`, and an allowed GPU;
- given a positive wall-clock cap (default 5 hours, matching Vast; `0` is
  rejected);
- terminated from the container on success and failure;
- discoverable by the `rlh-runpod-` name prefix so `reap` can clean up even if
  local `state.json` was lost.

The current v2 beta Pods API does not expose an interruptibility field. This
backend therefore uses RunPod's GraphQL on-demand mutation for creation and
authoritative `podType`, `secureCloud`, and GPU verification. The v1 REST API
is retained for listing, termination, and billing. Migrating fully to v2 is
blocked until v2 can explicitly request and return on-demand status.

## Setup

Copy `.env.example` into your secrets manager, not into a populated repository
file. Required local variables:

```bash
export RUNPOD_API_KEY='...'
export GH_TOKEN='...'
```

For durable checkpoints on Community Cloud, configure the existing B2 backend:

```bash
export B2_BUCKET='...'
export B2_ENDPOINT='https://s3...backblazeb2.com'
export B2_APPLICATION_KEY_ID='...'
export B2_APPLICATION_KEY='...'
export B2_PREFIX='runpod'  # optional
```

Community Cloud cannot attach RunPod network volumes. The runner uploads
checkpoints to B2 before terminating the Pod. `--forward-b2` fails before
creation unless all required B2 settings are present.

Cursor Cloud Agents receive the same names from **Dashboard → Cloud Agents →
Secrets**. Use Runtime Secret for API keys/tokens and ordinary environment
variables for bucket/endpoint names.

Batch jobs need no SSH client or keypair. Provisioning, inspection, logs,
billing, and termination use RunPod APIs; training evidence comes back through
compact Git results and B2 artifacts. Opt-in interactive Pods use SSH as
described below.

## Image strategy

The default is the public, digest-pinned
`ghcr.io/al-does/rl-harness-runpod` image built from the adjacent `Dockerfile`.
It starts from digest-resolved `ubuntu:24.04`; the pinned Linux PyTorch wheel
supplies the CUDA 13 runtime libraries. The image bakes:

- Python 3.13;
- `ray[rllib]==2.56.0` (exactly pinned);
- `torch==2.12.1`;
- `gymnasium==1.2.2`;
- B2 and MLflow clients.

Dependency installation precedes the runner source `COPY`, so runner edits do
not invalidate the expensive framework layer. Experiment and harness source
are never baked; both repositories are cloned at refs passed in environment
variables and editable-installed with dependency resolution disabled.

The OCI source label links the image to this repository. GHCR must remain
public because image pull happens before Pod environment variables exist; no
registry credential is sent to RunPod. Public visibility exposes only the
environment and runner—no source checkout or secret is present in the image.

To validate a newly published image before changing the default, pass its
digest:

```bash
--image ghcr.io/OWNER/IMAGE@sha256:...
```

## Dry run first

Dry run resolves the image digest, refs, hard cap, and public price estimate,
but does not create a Pod:

```bash
uv run python -m devops.runpod.pods.provision up \
  --commit EXPERIMENT_COMMIT \
  --library-commit HARNESS_COMMIT \
  --run-name token-guess-smoke \
  --max-age 1 \
  --max-price 0.50 \
  --run "rl-harness experiments.mess3_token_guess_cycle_1.iqn_first_checkpoint_reproduction.experiment --smoke --upload-artifacts --run-id token-guess-smoke" \
  --forward-b2 --self-destruct --dry-run
```

The public RTX 4090 Community price is used for the pre-create estimate. The
authoritative create response is checked against `--max-price`; an over-limit
Pod is terminated immediately.

## Launch

Run the same command without `--dry-run` and add `--yes`:

```bash
uv run python -m devops.runpod.pods.provision up \
  --commit EXPERIMENT_COMMIT \
  --library-commit HARNESS_COMMIT \
  --run-name token-guess-smoke \
  --max-age 1 \
  --max-price 0.50 \
  --run "rl-harness experiments.mess3_token_guess_cycle_1.iqn_first_checkpoint_reproduction.experiment --smoke --upload-artifacts --run-id token-guess-smoke" \
  --forward-b2 --self-destruct --yes
```

`--self-destruct` retains the Vast meaning of pushing compact `experiments/`
results to `--results-branch` (default `results`). Pod termination itself is
unconditional: success and failure both terminate.

The runner records experiment and harness SHAs plus the immutable image digest
as MLflow tags. Its file-backed MLflow run is uploaded to
`s3://$B2_BUCKET/$B2_PREFIX/runpod/mlflow/<run-name>/`.

## List, inspect, destroy, and reap

```bash
# List all managed Pods, including untracked orphans, and collect posted billing
uv run python -m devops.runpod.pods.provision status

# Redacted metadata only; environment secrets are never printed
uv run python -m devops.runpod.pods.provision inspect POD_ID

# Read recent container and system logs, then return
uv run python -m devops.runpod.pods.provision logs POD_ID --tail 100

# Follow container logs until the Pod exits or you press Ctrl-C
uv run python -m devops.runpod.pods.provision logs POD_ID \
  --source container --follow

# Terminate specific tracked or explicitly named Pods
uv run python -m devops.runpod.pods.provision destroy --id POD_ID --yes

# Discover managed untracked, failed, exited, or over-age Pods and terminate them
uv run python -m devops.runpod.pods.provision reap --yes
```

`reap` only touches names beginning with `rlh-runpod-`; it will not take over
unrelated Pods. Actual charges are read from RunPod's Pod billing endpoint and
saved in the gitignored `cost_history.json` after billing posts.

## Interactive SSH and profiling

`--interactive` creates the same on-demand Community Cloud GPU, enables only
TCP port 22, prepares both repositories and the pinned CUDA environment, then
waits for SSH work. It does not accept `--run`. The Pod remains protected by
provider `terminateAfter` and the in-container watchdog; its default ceiling is
2 hours. Destroy it manually as soon as profiling is complete.

```bash
# Validate key, refs, image, price, and hard ceiling without creating a Pod
uv run python -m devops.runpod.pods.provision up \
  --interactive --max-age 2 --max-price 0.50 --dry-run

# Create it
uv run python -m devops.runpod.pods.provision up \
  --interactive --max-age 2 --max-price 0.50 --yes

# Connect after the runner reports the CUDA workspace ready
uv run python -m devops.runpod.pods.provision ssh POD_ID

# Always terminate it when finished
uv run python -m devops.runpod.pods.provision destroy --id POD_ID --yes
```

The launcher discovers `~/.ssh/id_ed25519(.pub)` and then
`~/.ssh/id_rsa(.pub)`, or accepts `--ssh-key PATH` /
`RUNPOD_SSH_KEY_PATH`. It injects only the public key through RunPod's
documented `SSH_PUBLIC_KEY` per-Pod override. The private key never leaves the
launching laptop or Cursor Cloud VM, and no key is written to the repository.

Create a local key only if neither pair exists:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
```

Cursor Cloud bootstrap already creates `~/.ssh/id_rsa(.pub)`, so Cloud Agents
need no dashboard SSH secret and can use the same `--interactive`, `logs`, and
`ssh` commands. Keys are VM-specific; launch and connect from the same Cloud
Agent. Adding a public key to RunPod account settings is optional because the
backend injects it per Pod.

## Secret handling

- API keys and tokens are read from process environment only.
- The account-level RunPod key is never forwarded in Pod creation metadata.
  RunPod supplies a Pod-scoped `RUNPOD_API_KEY` to the container for teardown.
- Git authentication uses an in-process extra header. Clone remotes remain
  token-free, so credentials are not written to `.git/config`.
- Do not pass credentials as command-line flags or print raw Pod metadata.
