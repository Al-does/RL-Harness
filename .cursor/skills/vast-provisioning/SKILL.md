---
name: vast-provisioning
description: Rent, bootstrap, connect to, and tear down vast.ai RTX 4090 GPU boxes for remote training via the repo's devops/vast toolkit. Use when the user wants to run training on a remote/cloud GPU, rent a vast.ai box, provision GPUs, run a sweep on rented machines, or push results back and self-destruct the box.
---

# vast.ai provisioning (`devops/vast`)

A local Mac CLI that finds, ranks, rents, bootstraps, and connects to vast.ai
RTX 4090 boxes, with optional push-results-then-self-destruct. Boxes install
`uv`, clone the personal **experiment repo** and this **library** as siblings,
editable-install the library, `uv sync` the experiment env, and (optionally)
run a command in `tmux`.

> **AUTO-DESTROY:** every box self-destroys after a wall-clock cap (default 5h,
> `--max-age`) via an on-box watchdog that fires even if this Mac is off. This is
> a safety net, **not** a substitute for cleaning up — still `destroy` boxes as
> soon as you're done. See [Max-age cap](#max-age-cap-hard-cost-backstop).

> **COST WARNING:** boxes bill hourly the moment they reach `running`, and
> storage bills from creation. **ALWAYS** `destroy` boxes when done. `state.json`
> + `destroy --all` is the backstop. Never leave this task without confirming
> no boxes remain (`status`, or check <https://console.vast.ai/instances/>).

> **DO NOT TAKE OVER ANOTHER AGENT'S BOX:** multiple Cursor agents (or git
> worktrees) can run on the same Mac against the same vast.ai account. Each
> library checkout has its **own** gitignored `devops/vast/state.json`, but
> `~/.ssh/config.d/vast.conf` and the vast account are **shared machine-wide**.
> An agent whose `state.json` is empty must **not** assume no boxes are running.
> **Never** SSH into, re-bootstrap, or run commands on a box you did not rent
> with `provision up` in **this** agent session and checkout. Aliases are
> `vast-<instance-id>` and are merged (not rewritten) into the shared SSH
> config — still connect only to the alias/`id` from **your** `provision up`
> output. **Never** use another checkout's `state.json` or console instance IDs
> to reach a running box. When asked to run on GPU, **always rent a fresh box**
> (`provision up … --dry-run` first). Only `destroy --id <your-id>`; never run
> `destroy --all` unless the user explicitly confirms no other agent or worktree
> session has active boxes.
>
> **Never dump raw instance metadata.** `vastai show instance --raw` includes
> plaintext `extra_env` secrets. Use `provision status` or `provision inspect
> <id>` (redacted). Pass `--forward-b2` only when the run needs B2 uploads.

## Prerequisites (already set up on this machine)

- `VAST_API_KEY` env → `~/.vast_api_key` → `vastai` stored key (resolved in that order).
- Local OpenSSH client (`ssh` on `PATH`) and keypair `~/.ssh/id_rsa(.pub)`
  (registered on the vast account automatically). Cursor Cloud images install
  `openssh-client` in the Dockerfile; bootstrap generates a key if missing.
  `provision up` refuses to rent when either is absent (avoids billed unready boxes).
- GitHub token (`--github-token` → `GITHUB_TOKEN` → `gh auth token`) when the
  experiment repo is private (needed for the initial clone) and/or when using
  `--self-destruct` (needed to push compact `experiments/` results).
- Always run through the `devops` group so `vastai` never enters the training env:
  `uv run --group devops python -m devops.vast.provision ...`

## Commands

Always **`--dry-run` first** to preview ranked candidates and price before renting.

```bash
# Preview ranked candidates, rent nothing
uv run --group devops python -m devops.vast.provision up -n 2 --dry-run

# Rent 1 on-demand box, run a smoke train in tmux, auto-open a terminal tab
uv run --group devops python -m devops.vast.provision up -n 1 \
  --run "rl-harness experiments.mess3_belief_geometry_2026_07.reward_only.experiment --seed 0 --smoke" --yes

# See tracked boxes + live status
uv run --group devops python -m devops.vast.provision status

# Redacted instance metadata (safe for logs)
uv run --group devops python -m devops.vast.provision inspect <INSTANCE_ID>

# Reap any tracked box older than the max-age cap (local backstop; cron-friendly)
uv run --group devops python -m devops.vast.provision reap --yes

# Tear down only boxes you rented (prefer --id over --all)
uv run --group devops python -m devops.vast.provision destroy --id <INSTANCE_ID> --yes
```

`up` is the default subcommand. Key `up` flags: `-n/--count`,
`--mode {ondemand,interruptible}`, `--bid`, `--disk`, `--image`,
`--branch`/`--commit` (experiment-repo ref; default = local experiment `HEAD`),
`--library-branch`/`--library-commit` (rl-harness ref; default `main`),
`--experiment-repo PATH`, `--run "CMD"`, `--max-price`,
`--regions US,CA` (hard country filter when set), `--dry-run`, `--yes`,
`--offer-id ID`, `--exclude-machine ID [ID ...]`, `--no-open`,
`--max-age HOURS` (lifetime cap; default 5, `0` disables),
`--forward-b2` (inject B2 credentials for artifact upload; off by default).
Self-destruct pushes compact `experiments/` changes from the **experiment**
repo: `--self-destruct`, `--run-name NAME`, `--results-branch NAME`,
`--github-token`, `--teardown-on-error`.
`destroy`: `--all` or `--id <id> ...` (`--yes` skips confirm).
`reap`: `--max-age HOURS` (override), `--yes`.
`inspect <id>`: redacted metadata (never use `vastai show instance --raw`).

## `--run` semantics

The command runs in the repo dir inside a detached `tmux` session named `run`.
The runner activates the pre-synced `.venv` first; do **not** prefix the command
with `uv run`, because Ray would otherwise recreate the uv environment for
worker processes. Example:
`--run "rl-harness experiments.mess3_belief_geometry_2026_07.reward_only.experiment --seed 0"`.

## Self-destruct (push results, then destroy)

`--self-destruct` makes each box push compact changes under `experiments/` to a
branch (default `results`, keeping `main` clean) and destroy itself when the run
finishes. Per-experiment `artifacts/` trees are ignored, so checkpoints and raw
payloads are not pushed. A **crashed** run stays up for debugging unless
`--teardown-on-error` is passed.

Requirement: the teardown hook only exists in the **cloned ref**, so the ref you
launch (`--branch`/`--commit`, default local `HEAD`) must already be pushed to
the remote and contain the current `devops/vast` runner.

## Max-age cap (hard cost backstop)

Independent of `--self-destruct` (which fires when the *run* ends), every box
gets a wall-clock lifetime cap (`--max-age`, default 5h; `0` disables). An on-box
`tmux` "watchdog" sleeps for the cap then REST-destroys the box — it fires **even
if this Mac is off** or the run never finished, and is armed *before* `uv sync`
so a failed-sync box still gets reaped. `provision reap` is the local backstop:
it destroys any tracked box past its cap (cron/loop it). The cap injects
`VAST_API_KEY` onto the box (host-visible, same tradeoff as self-destruct).

## Monitoring a run without SSH

Bootstrap output is tee'd to the container log and the tmux run's tail is
surfaced there on completion, so progress is visible even if SSH is unreachable:

```bash
uv run --group devops python -c "from vastai import VastAI; \
print(VastAI(api_key=open('$HOME/.vast_api_key').read().strip()).logs(<INSTANCE_ID>, tail=40))"
```

Readiness = `actual_status == running` **and** `/root/.vast_ready` exists (env
fully `uv sync`ed and torch CUDA validated). Bootstrap failures write
`/root/.vast_bootstrap_failed`; `provision up` returns nonzero if any created
box fails readiness. Sync is capped at 20 minutes by default to fail fast on
pathologically slow hosts.

## Gotchas (learned in practice)

- **Parallel agents / worktrees share one vast account.** `state.json` is per
  library checkout; another agent's box won't appear in yours. Empty local
  state does not mean the account is idle — check
  <https://console.vast.ai/instances/> if unsure. Rent your own box; connect
  only via the `vast-<instance-id>` alias printed by your `up` (aliases merge
  into the shared SSH config and are not reused across instances).
- **On-demand offers churn.** Top picks often return HTTP 410 (Gone) or would
  create a *stopped* (still-billed) box. The tool passes `cancel_unavail=True`
  and falls through to the next-best offer automatically — expect a few
  "offer … skipped" lines before one sticks.
- **Prefer reliable mid-market hosts.** Ranking keeps the upper inner price
  quartile `[Q2, Q3]` among gated hosts (small pools fall back to a modest
  floor-relative cap), then sorts by reliability / CPU before price.
- **Failed hosts are quarantined locally.** Re-run with
  `--exclude-machine <machine-id>` after a provider host stalls, or use
  `--offer-id` / `--machine-id` to pin a candidate. Explicit `--regions US,CA`
  is a hard country filter. Readiness failures destroy the box, write
  gitignored `devops/vast/quarantine.json` (machine + public IP, 7-day TTL),
  and try the next ranked offer. On-box `uv sync` also fails after ~8 minutes
  with no log progress so fallthrough is not stuck for the full 20 minutes.
- **Remote clones omit legacy bulk results.** Bootstrap uses a depth-one,
  blob-filtered sparse checkout, excluding root `results/` while retaining the
  complete `experiments/` tree needed for runs and compact result pushes.
- **Direct SSH port may be blocked** by the client network; the tool probes and
  falls back to the vast proxy (`sshN.vast.ai`). Some individual hosts also have
  flaky SSH key propagation — if a box never becomes reachable, `destroy` it and
  re-run to land on a different host.
- **torch/CUDA `uv sync` works** on `vastai/base-image:@vastai-automatic-tag`
  (torch's wheels bundle CUDA; only a compatible host driver is needed) — no
  custom torch index required. Bootstrap hard-fails if torch still cannot use
  CUDA despite the offer's `cuda_max_good` gate.
- Bootstrap logs cgroup CPU quota, host load, and PCIe link generation/width.
  `harness/hardware.py` caps Ray's logical CPU resources and experiment resource
  sizing to the same cgroup-aware CPU count.
- If a `provision up` process is interrupted, an instance may already be
  created; run `status` / `destroy --all` to be safe.

## Full reference

See `devops/vast/README.md` for the complete flag table, the scoring/gating
rules (price-band ranking with region tiebreak across distinct hosts), and the
self-destruct concurrency design.
