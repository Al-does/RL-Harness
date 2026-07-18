# `devops/vast` — vast.ai provisioning toolkit

Find, rank, rent, bootstrap, and connect to vast.ai RTX 4090 boxes from your Mac,
with one CLI. Boxes clone the **experiment repo** (science + results push) and
this **library** as siblings, editable-install the library, `uv sync` the
experiment env, and (optionally) run a command in `tmux` — then optionally push
compact experiment `results/` back and self-destruct.

> **Cost warning:** rented boxes bill by the hour the moment they reach `running`,
> and storage bills from creation. **Always** `destroy` boxes you are done with.
> `state.json` + `destroy` are your backstop if a box fails to self-destruct.

## Prerequisites

- **vast API key.** Get one at <https://console.vast.ai/manage-keys/>. Resolution
  order: `VAST_API_KEY` env → `~/.vast_api_key` (`chmod 600` recommended) →
  the key stored by `vastai set api-key`.
- **SSH keypair** at `~/.ssh/id_rsa(.pub)`. The tool registers `id_rsa.pub` on
  your vast account so direct SSH works.
- **`gh` CLI** authed (for `--self-destruct` result pushes): token resolution is
  `--github-token` → `GITHUB_TOKEN` → `gh auth token`.
- The `devops` dependency group: `uv sync --group devops` (installs `vastai`
  locally only — it is never installed on the boxes).

## Usage

Run everything through the `devops` group so `vastai` stays out of the training env:

```bash
# Preview the ranked candidates without renting anything
uv run --group devops python -m devops.vast.provision up -n 2 --dry-run

# Rent 1 on-demand box, run a smoke train in tmux, auto-open a terminal tab
uv run --group devops python -m devops.vast.provision up -n 1 \
  --run "rl-harness experiments.mess3_belief_geometry_2026_07.reward_only.experiment --seed 0 --smoke" --yes

# Rent 3 interruptible (spot) boxes, each self-destructing after it finishes
uv run --group devops python -m devops.vast.provision up -n 3 --mode interruptible \
  --self-destruct --run-name sweepA \
  --run "rl-harness experiments.mess3_belief_geometry_2026_07.reward_only.experiment --seed 0"

# See what you have running
uv run --group devops python -m devops.vast.provision status

# Retry without a host that failed bootstrap
uv run --group devops python -m devops.vast.provision up -n 1 \
  --exclude-machine 140297 --yes

# Destroy any tracked box older than the max-age cap (local backstop; cron this)
uv run --group devops python -m devops.vast.provision reap --yes

# Tear everything down
uv run --group devops python -m devops.vast.provision destroy --all
```

`up` is the default subcommand, so `... provision -n 2 --dry-run` also works.

### `up` flags

| flag | meaning |
|------|---------|
| `-n/--count N` | number of boxes (across distinct hosts) |
| `--mode {ondemand,interruptible}` | rental type (default `ondemand`) |
| `--bid $/hr` | interruptible bid (default: auto = `min_bid * BID_MARGIN`) |
| `--disk GB` | local disk (default from `config.py`) |
| `--image IMG` | docker image (default from `config.py`) |
| `--branch` / `--commit` | experiment-repo ref to clone (default: local experiment `HEAD`) |
| `--library-branch` / `--library-commit` | rl-harness ref (default: `main`) |
| `--experiment-repo PATH` | local experiment repo used to resolve HEAD |
| `--run "CMD"` | run `CMD` in the activated, pre-synced project environment |
| `--max-price $/hr` | hard price cap |
| `--regions US,CA` | require these country codes (hard filter when set; default `HOME_REGIONS` remains tiebreak-only) |
| `--offer-id ID` | rent one exact displayed offer (requires `--count 1`) |
| `--exclude-machine ID [ID ...]` | omit known-bad provider machines |
| *(auto)* | destroy hosts that miss readiness, quarantine them locally, and try the next ranked offer |
| `--dry-run` | print ranked candidates, rent nothing |
| `--yes` | skip the rent confirmation |
| `--no-open` | do not auto-open terminal tabs |
| `--self-destruct` | inject teardown env + enable the training push+destroy hook |
| `--run-name NAME` | per-shot results subdir + commit label |
| `--results-branch NAME` | branch the box pushes results to (default `results`) |
| `--github-token TOK` | write token (else `GITHUB_TOKEN` / `gh auth token`) |
| `--teardown-on-error` | also push+destroy if the run raises (off by default) |
| `--max-age HOURS` | wall-clock lifetime cap (default `MAX_AGE_HOURS`=5; `0` disables) |

`destroy`: `--all` or `--id <id> ...` (`--yes` skips confirm). `reap`:
`--max-age HOURS` (override), `--yes`. `status`: shows live status of tracked boxes.

## Max-age cap (hard cost backstop)

Every box gets a wall-clock lifetime cap (`--max-age`, default `MAX_AGE_HOURS`=5;
`0` disables). This is a safety net against a forgotten box billing forever —
distinct from `--self-destruct`, which fires when the *run* finishes.

- **On-box watchdog (primary).** `bootstrap.sh` arms a detached `tmux` session
  (`watchdog`) that `sleep`s for the cap, then runs
  `self_destruct.py --max-age` to REST-destroy the box. It fires **even if your
  Mac is off** and **even if the run never finished**. It is armed *before*
  `uv sync`, so a box whose sync failed (and so lingers for debugging) is still
  reaped. If the box was launched with `--self-destruct`, the watchdog salvages
  compact experiment results before destroying; otherwise it destroys straight
  away.
- **Local `reap` (backstop).** `provision reap` destroys any *tracked* box whose
  `created_at` in `state.json` is past its cap. Cron/loop it to catch boxes
  whose on-box timer never fired (e.g. a `stopped` interruptible box).

The watchdog needs the vast API key on the box (to REST-destroy itself), so the
cap injects `VAST_API_KEY` into the container env — the same host-visibility
tradeoff already accepted for `--self-destruct` boxes. Pass `--max-age 0` to opt
out (not recommended).

## How "best" is chosen (`scoring.py`)

Offers expose only a coarse `geolocation` string, so there is no true geodistance.
Ranking prefers **reliable mid/upper-market hosts** over the absolute cheapest:

- **Hard gates** (drop the offer): `reliability2 >= MIN_RELIABILITY`,
  `verification == "verified"`, max rental `duration >= MIN_DAYS`,
  `disk_space >= disk + headroom`, `direct_port_count >= 1`,
  `cuda_max_good >= MIN_CUDA`, `cpu_cores_effective >= MIN_CPU_CORES`,
  `rentable`, and (optional) `effective_price <= --max-price`.
- **Price band** among gated distinct hosts:
  - if there are at least `PRICE_BAND_MIN_HOSTS` hosts, keep the upper inner
    quartile `[Q2, Q3]`;
  - otherwise keep `[floor, max(floor * PRICE_BAND_FLOOR_MULT, floor + PAD)]`.
- **Rank key inside the band** =
  `(-reliability2, -cpu_cores_effective, -inet_down, region_rank, price)`.
- **Distinct hosts:** the top N never include two offers on the same `machine_id`.
- **Local quarantine** (`quarantine.json`, gitignored): machines / public IPs that
  miss readiness are excluded for `QUARANTINE_TTL_S` (default 7 days).
- **Operator controls:** `--offer-id` / `--machine-id` pin one listing (and skip
  the price band), while `--exclude-machine` removes known-bad hosts before
  ranking. Explicit `--regions` is a hard country filter.

`effective_price` is `dph_total` (on-demand) or your bid (interruptible).
Created boxes are monitored concurrently; unready hosts are destroyed and
replaced from the remaining pool. On-box `uv sync` also fails after
`UV_SYNC_STALL_S` seconds with no log progress so fallthrough happens sooner.

## Fast remote checkout

Bootstrap uses a depth-one, blob-filtered sparse checkout containing the source,
tests, docs, and experiment tree. Per-leaf `experiments/**/results/` remains
available for training and self-destruct result pushes. An exact branch, tag, or
pushed commit can still be selected with `--branch` or `--commit`.

## Self-destruct on completion

With `--self-destruct`, each box is given a git identity + a token-authed
`origin`, and the remote runner's teardown hook fires when the run finishes:

1. Stage changes under `experiments/`. Each experiment's ignored `artifacts/`
   tree keeps checkpoints, raw payloads, and verbose logs out of Git.
2. Nothing new? Log "nothing to push" and succeed (no commit, no failure).
3. Otherwise commit and run a bounded **fetch → rebase --autostash → push**
   retry loop against `--results-branch` (default `results`, keeping `main`
   clean). Disjoint per-run folders + the retry loop let N concurrent boxes push
   the same branch without conflicts or non-fast-forward rejections.
4. Destroy the box in a `finally`, so a push hiccup still frees it.

A **crashed** run stays up for debugging unless `--teardown-on-error` is set.

Notes and tradeoffs:

- The teardown hook only exists in the cloned ref, so `--self-destruct` requires
  the ref you launch (`--branch`/`--commit`, default local `HEAD`) to already be
  pushed and to contain this code.
- Ray's `uv run` runtime-env hook can recreate the project environment for
  every worker. `run_remote.sh` activates the already-synced `.venv` and invokes
  the requested command directly, so Ray workers reuse that environment.
- Boxes report the host's core count (`nproc` can say 128) but are capped by a
  docker CPU quota (often ~16); size Ray workloads from
  `/sys/fs/cgroup/cpu.max` (or the cgroup-v1 quota files), not
  `os.cpu_count()`. `harness/hardware.py` uses that quota for Ray's logical CPU
  pool and experiment resource sizing, preventing phantom schedulable cores.
- The pinned `torch==2.12.1` PyPI wheels are CUDA-13 builds; hosts with older
  drivers (e.g. 570 / CUDA 12.8) import fine but `torch.cuda.is_available()`
  is False. `scoring.py` gates offers on `cuda_max_good >= MIN_CUDA` (13.0),
  and bootstrap refuses readiness unless torch can actually use the GPU.
- Bootstrap logs the cgroup CPU quota, host load, and current PCIe generation
  and width. `uv sync` has a bounded total timeout (`UV_SYNC_TIMEOUT_S`) so a
  pathological network does not consume the full max-age window silently.
- The instance id isn't known before creation, so the box resolves its own id by
  a unique injected label via the vast REST API at teardown time. The destroy
  call uses stdlib `urllib` (no `vastai` on the box), keeping the training env clean.
- Each self-destruct box holds a write-capable GitHub token and your
  `VAST_API_KEY`, both visible to the host. Neither token can delete repos.

## Files

| file | role |
|------|------|
| `config.py` | `VastConfig` defaults (GPU, disk, image, regions, gates, paths) |
| `quarantine.py` | local gitignored bad-host quarantine (machine id + public IP) |
| `vast_client.py` | thin `vastai` SDK wrapper: auth, search, create, poll, destroy |
| `redaction.py` | strips credentials from third-party exception text before logging |
| `scoring.py` | pure `build_query()` + `rank_offers()` (gates + ranking) |
| `bootstrap.sh` | remote setup: `uv`, clone@ref, `uv sync`, ready sentinel, `tmux` run |
| `run_remote.sh` | activate the synced environment, run the command, and trigger teardown |
| `self_destruct.py` | on-box: push compact experiment results + destroy (REST, stdlib only) |
| `terminals.py` | write `~/.ssh/config.d/vast.conf`, open iTerm2/Terminal tabs |
| `provision.py` | CLI orchestrator (`up`/`destroy`/`status`) |
| `state.json` | gitignored record of rented boxes (ids, labels, connection info) |
