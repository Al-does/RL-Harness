# Backblaze B2 artifact storage

Large experiment outputs live under ignored `artifacts/<run-id>/` trees:
checkpoints, Tune trial directories, `.pt` weights, raw payloads, and logs.
Compact findings stay in tracked `results/<run-id>/`.

When B2 credentials are configured, the harness uploads the entire
`artifacts/<run-id>/` tree at the end of a run (success or failure), records
URIs in `results/<run-id>/`, and leaves the local files in place.

## What you need to do in Backblaze

1. Sign in to [Backblaze B2](https://www.backblaze.com/b2/cloud-storage.html).
2. Create a **private** bucket (recommended). Note the bucket name.
3. Open **Application Keys** and create a key scoped to that bucket with
   **Read and Write** access.
4. Copy the **keyID** and **applicationKey** immediately; the secret is shown
   once.
5. Find your bucket's **S3 Endpoint** on the bucket details page. It looks
   like `https://s3.us-west-004.backblazeb2.com`.

Keep the bucket private. Share objects with time-limited pre-signed URLs when
you need to hand checkpoints to collaborators or analysis tools.

## Install the upload dependency

The harness uses `boto3` against B2's S3-compatible API. Install it in the
environment that runs experiments:

```bash
# In rl-harness (library development)
uv sync --extra storage

# In alex-rl-experiments (recommended default)
uv sync --extra storage
```

The personal experiment repo depends on `rl-harness[storage]` so a normal
`uv sync` there pulls in `boto3`.

## Configure credentials

### Cursor Cloud Agents

Add these as **Runtime Secrets** in **Cursor Dashboard → Cloud Agents →
Secrets** (scope them to your saved environment if you use one):

| Secret name | Type |
|-------------|------|
| `B2_BUCKET` | Environment Variable |
| `B2_ENDPOINT` | Environment Variable |
| `B2_APPLICATION_KEY_ID` | Runtime Secret |
| `B2_APPLICATION_KEY` | Runtime Secret |
| `B2_PREFIX` | Environment Variable *(optional)* |

### Local Mac (recommended)

Use the interactive setup script from your experiment repo:

```bash
cd /Users/alex/Software/XOR/alex-rl-experiments
./scripts/setup_b2_env.sh
```

That writes `~/.rl_harness_b2_env` with mode `600`. Both local `rl-harness`
runs and `devops.vast.provision` read that file automatically — you do not need
to export the vars manually each time.

Run experiments with secrets loaded:

```bash
./scripts/run_harness.sh experiments.mess3_belief_geometry_2026_07.reward_only.experiment --smoke
```

Rent vast boxes with the same credentials forwarded to the remote container:

```bash
./scripts/run_vast.sh up -n 1 \
  --run "rl-harness experiments.mess3_belief_geometry_2026_07.reward_only.experiment --seed 0 --smoke" \
  --self-destruct --yes
```

Optional: load secrets into your current shell only:

```bash
source scripts/load_env.sh
```

Optional: keep a repo-local override in `.env.local` (gitignored). Copy
`.env.local.example` as a starting point. `scripts/load_env.sh` sources
`~/.rl_harness_b2_env` first, then `.env.local`.

Optional: add this to `~/.zshrc` if you want every terminal session to inherit
the vars:

```bash
[ -f "$HOME/.rl_harness_b2_env" ] && set -a && source "$HOME/.rl_harness_b2_env" && set +a
```

`./scripts/setup_b2_env.sh` offers this by default (press `n` to opt out). To
install or remove it later:

```bash
./scripts/b2_shell_autoload.sh install   # add to ~/.zshrc
./scripts/b2_shell_autoload.sh remove    # remove managed block
./scripts/setup_b2_env.sh --shell-autoload-only
./scripts/setup_b2_env.sh --no-shell-autoload
```

### Manual environment variables

Export these wherever you run `rl-harness` if you prefer not to use the secrets
file:

| Variable | Example | Purpose |
|----------|---------|---------|
| `B2_BUCKET` | `alex-rl-artifacts` | Target bucket |
| `B2_ENDPOINT` | `https://s3.us-west-004.backblazeb2.com` | S3-compatible endpoint |
| `B2_APPLICATION_KEY_ID` | `004…` | Application key ID |
| `B2_APPLICATION_KEY` | `K004…` | Application key secret |
| `B2_PREFIX` | `prod` | Optional root prefix inside the bucket |

The harness also accepts standard AWS-style names as fallbacks:
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.

Override the secrets file location with `RL_HARNESS_B2_ENV_FILE` if needed.

For vast.ai boxes, provisioning injects the same `B2_*` variables into the
remote container env before training starts. Upload happens inside `rl-harness`
before the self-destruct step pushes compact `results/` to Git, so URIs land in
`run_manifest.json` on the results branch.

Treat the application key as a secret. Do not commit it.

## Object layout

Each run uploads under a stable prefix derived from the experiment path and
run id:

```text
{B2_PREFIX}/{experiment_path_relative_to_repo}/{run_id}/{artifact_relative_path}
```

Example:

```text
prod/experiments/mess3_belief_geometry_2026_07/reward_only/20260718T205806Z-1028e338/tune/PPO_.../checkpoint_000001/...
```

## What gets recorded locally

After upload, the harness writes:

- `results/<run-id>/remote_artifacts.json` — full per-file URIs, sizes, and
  SHA-256 digests.
- `results/<run-id>/run_manifest.json` — compact `remote_artifacts` summary
  pointing at the manifest file plus aggregate counts.

Example manifest summary:

```json
"remote_artifacts": {
  "backend": "b2-s3",
  "bucket": "alex-rl-artifacts",
  "endpoint": "https://s3.us-west-004.backblazeb2.com",
  "prefix": "prod/experiments/study/condition/20260718T120000Z-deadbeef",
  "base_uri": "s3://alex-rl-artifacts/prod/experiments/study/condition/20260718T120000Z-deadbeef/",
  "status": "completed",
  "file_count": 42,
  "total_bytes": 987654321,
  "manifest_file": "remote_artifacts.json"
}
```

Upload failures do not fail the scientific run. The manifest records
`status: "failed"` and an error message instead.

## CLI behavior

By default, upload runs automatically when all required B2 variables are set.

```bash
uv run rl-harness experiments.study.condition.experiment

# Force upload (errors if B2 is not configured)
uv run rl-harness experiments.study.condition.experiment --upload-artifacts

# Skip upload even when B2 is configured
uv run rl-harness experiments.study.condition.experiment --no-upload-artifacts
```

## Downloading a checkpoint later

Use any S3 client pointed at the same endpoint and credentials:

```bash
export AWS_ACCESS_KEY_ID="$B2_APPLICATION_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$B2_APPLICATION_KEY"
aws s3 cp \
  --endpoint-url "$B2_ENDPOINT" \
  "s3://alex-rl-artifacts/prod/experiments/.../checkpoints/module_state_final.pt" \
  ./module_state_final.pt
```

Or read `remote_artifacts.json` from the tracked results folder and copy the
`uri` field for the file you need.

## Lifecycle and cost notes

- B2 charges for stored bytes and download bandwidth; uploads are free.
- Consider a bucket lifecycle rule to expire old smoke-test prefixes if you
  upload `--smoke` runs during development.
- Git still tracks only compact `results/`; `artifacts/` remains ignored locally
  even after upload.
