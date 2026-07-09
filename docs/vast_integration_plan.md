# Vast.ai Provisioning Integration — Build Plan

Build a `devops/vast/` toolkit that finds, ranks, rents, bootstraps, and connects to
vast.ai RTX 4090 boxes. Everything is a local Mac dev tool driven from one CLI.

## Build status

- [ ] **scaffold** — Create `devops/vast/` package + `config.py` defaults; add `vastai` to a `devops` dependency group in `pyproject.toml`; gitignore `state.json`.
- [ ] **client** — `vast_client.py` wrapper: auth, `search_offers`, SSH-key registration, `create_instance` (on-demand + bid), `wait_until_running`, connection info, `destroy_instance`.
- [ ] **scoring** — `scoring.py`: `build_query()` and `rank_offers()` with hard gates (reliability/verified/maxdur/disk/direct-port) and price-then-proximity ranking with tolerance band + distinct-host selection.
- [ ] **bootstrap** — `bootstrap.sh`: install `uv`, git clone at ref, `uv sync`, write readiness sentinel, optional `--run` command in `tmux`.
- [ ] **terminals** — `terminals.py`: write `~/.ssh/config.d/vast.conf` aliases and auto-open iTerm2 tabs (Terminal.app fallback), each SSH'd into a box; `--no-open` flag.
- [ ] **cli** — `provision.py` orchestrator: `up`/`destroy`/`status` subcommands and all flags; `state.json` tracking.
- [ ] **self-destruct** — `self_destruct.py` (`push_results` with nothing-to-push tolerance + fetch/rebase/retry push loop, `destroy_self`, `push_results_and_destroy`); inject teardown env in provision + bootstrap; opt-in teardown hook in `scripts/train.py`.
- [ ] **docs-verify** — `README.md` (API key, usage, cost/teardown warnings) and a live single-box dry-run + real run to confirm torch/CUDA `uv sync` and exact SDK arg names.

## Decisions locked in

- Code delivery: `git clone` of the public repo at a ref (default = current local `HEAD` sha; override with `--branch`/`--commit`).
- Setup scope: ready dev env (install `uv`, clone at ref, `uv sync`) and then idle — plus an optional `--run "CMD"` that launches your command in a persistent `tmux` session.
- Connect UX: auto-open one iTerm2 tab per box already SSH'd in, with a Terminal.app fallback (different command per box, so per-tab, no broadcast default).
- "Best box" gates: hard-filter on reliability, verified host, max rental duration, and disk; then rank.
- Self-destruct on completion: a training run on the box can, when finished, push new files under `results/` to the remote and then destroy its own instance. Must not fail when `results/` has nothing new. Push target defaults to a dedicated `results` branch (keeps `main` history clean; auto-created from the launched ref if absent), overridable via `--results-branch`. Uses a fetch+rebase+retry loop to survive concurrent boxes.
- Teardown-on-error default: a crashed run stays up for debugging; self-destroy on failure only when `--teardown-on-error` is passed.

## "Best" scoring (note the proximity nuance)

Offers only expose `geolocation` (2-letter country code) — there is no lat/long, so true
geodistance is impossible. Proximity becomes an ordered region-preference list (config
`HOME_REGIONS`, e.g. `["US","CA"]`). Ranking:

- Hard gates (drop offer if it fails): `reliability2 >= MIN_RELIABILITY`, `verified == true`, `duration >= MIN_DAYS`, `disk_space >= disk + headroom`, `direct_port_count >= 1`, `rentable == true`, optional `dph_total <= --max-price`.
- Rank: price-primary with a tolerance band, proximity as tiebreak. Sort key = `(round(effective_price / PRICE_TOLERANCE), region_rank, effective_price)`. So near-equal prices prefer closer regions. `effective_price` = `dph_total` (on-demand) or your bid (interruptible).
- Pick the top N across distinct hosts (avoid renting two offers on the same `machine_id`).

## Folder layout: `devops/vast/`

- `config.py` — dataclass of defaults: `GPU_NAME="RTX_4090"`, `NUM_GPUS=1`, `DISK_GB`, `IMAGE`, `HOME_REGIONS`, `MIN_RELIABILITY`, `MIN_DAYS`, `PRICE_TOLERANCE`, `REPO_URL`, `SSH_KEY_PATH`.
- `vast_client.py` — thin wrapper over the `VastAI` SDK: auth, `search_offers`, `create_instance`, `wait_until_running`, connection info, `destroy_instance`, SSH-key registration.
- `scoring.py` — `build_query()` + `rank_offers()` implementing the gates/scoring above (pure functions, easy to unit test).
- `bootstrap.sh` — remote script run on each box: install `curl/git` + `uv`, `git clone REPO_URL`, `git checkout <ref>`, `uv sync`, write a `/root/.vast_ready` sentinel, and if `--run` was given, start it in `tmux new-session -d -s run '<cmd>'`. When self-destruct is enabled it also configures git identity + a token-authed `origin` and writes the injected env for the teardown hook.
- `self_destruct.py` — runs on the box; exposes `push_results_and_destroy()` plus lower-level `push_results()` and `destroy_self()` (see Self-destruct section).
- `terminals.py` — write `~/.ssh/config.d/vast.conf` (Host aliases `vast-1..N` with HostName/Port/User/IdentityFile/`StrictHostKeyChecking accept-new`), then open tabs: detect iTerm2 (`/Applications/iTerm.app`) -> AppleScript via `osascript` opens N tabs each running `ssh vast-k`; fallback to Terminal.app; `--no-open` skips.
- `provision.py` — the main CLI orchestrator (entrypoint).
- `state.json` — gitignored record of rented instances for teardown.
- `README.md` — usage, prerequisites (API key), teardown, cost warnings.

## CLI (`python -m devops.vast.provision ...`)

- `up` (default): `-n/--count`, `--mode {ondemand,interruptible}`, `--bid` (interruptible; default = auto from `min_bid` * margin), `--disk`, `--image`, `--branch`, `--commit`, `--run "CMD"`, `--max-price`, `--regions US,CA`, `--dry-run` (print ranked candidates + scores, rent nothing), `--yes`, `--no-open`.
- Self-destruct flags on `up`: `--self-destruct` (inject teardown env + enable the training hook), `--run-name NAME` (per-shot results subdir + commit label), `--results-branch NAME` (push target; default = dedicated `results` branch, auto-created from the launched ref if absent), `--github-token` (or `GITHUB_TOKEN` env), `--teardown-on-error` (also push+destroy if the run raises; off by default so crashed runs stay up for debugging).
- `destroy`: `--all` or `--id ...`, reads `state.json`, confirms, calls `destroy_instance`.
- `status`: `show_instances` for tracked boxes.

## End-to-end flow

```mermaid
flowchart TD
  A["provision up -n N"] --> B["load config + VAST_API_KEY"]
  B --> C["ensure local SSH pubkey registered on vast account"]
  C --> D["search_offers(query, type=ondemand|bid)"]
  D --> E["rank_offers: gates then price/proximity"]
  E --> F{"--dry-run?"}
  F -->|yes| G["print ranked table, exit"]
  F -->|no| H["confirm, then create_instance xN (ssh+direct, onstart=bootstrap, label)"]
  H --> I["poll show_instances until running + ssh_host/port + /root/.vast_ready"]
  I --> J["write ~/.ssh/config.d/vast.conf, save state.json"]
  J --> K["open iTerm2 (fallback Terminal.app) tab per box, ssh vast-k"]
```

## Key implementation details

- Auth: read `VAST_API_KEY` env (or the key stored by `vastai set api-key`). `README` documents getting it from cloud.vast.ai/manage-keys.
- SSH keys: register the local public key on the vast account (SDK `create_ssh_key`/`attach_ssh`) so direct SSH works; use `--ssh --direct` and require `direct_port_count>=1`.
- Bootstrap delivery: pass a compact `onstart_cmd` that curls `bootstrap.sh` raw from the public repo (or inlines it) and execs it, so setup is fully unattended. Repo is public -> no clone creds needed.
- Readiness: box is "ready" only when `actual_status=="running"` AND `/root/.vast_ready` exists (polled over SSH), so tabs don't open into a half-installed env.
- Interruptible safety: note in output that outbid instances go to `stopped` (storage still billed); `destroy` cleans them up.
- Dependency isolation: add `vastai` to a new `[dependency-groups] devops` group in `pyproject.toml` so it never pollutes the training env; run via `uv run --group devops`.
- `.gitignore`: add `devops/vast/state.json`.

## Self-destruct on completion (push results, then destroy)

Goal: when a run on the box finishes, push new files under `results/` to the remote, then
destroy the box. Never fail when there is nothing new to push.

Env injected per box at provision time (via `create_instance(env=...)`): `VAST_API_KEY`,
`VAST_INSTANCE_ID` (the `new_contract` id), `GITHUB_TOKEN`, `VAST_RESULTS_BRANCH`,
`VAST_RUN_NAME`, `VAST_SELF_DESTRUCT=1`. Bootstrap also sets `git config user.{name,email}`
and rewrites `origin` to `https://x-access-token:$GITHUB_TOKEN@github.com/Al-does/RLLibHarnesBeta.git`.

`self_destruct.py` behavior:

- `push_results()`:
  - `git add -A results/` (respects `.gitignore`, so pngs / checkpoint dirs / pkl / tfevents are excluded; csv/json/npz/md/state `.pt` are kept).
  - If nothing staged (`git diff --cached --quiet` returns 0): log "nothing to push", return success — no commit, no failure.
  - Else commit `"results: <VAST_RUN_NAME> (vast <id>)"`, then a bounded retry loop (K attempts, jittered backoff): `git fetch origin <branch>` -> `git rebase --autostash origin/<branch>` -> `git push origin HEAD:<branch>`. Disjoint per-run folders make each rebase auto-apply, so the push effectively always succeeds even with N concurrent boxes.
- `destroy_self()`: `VastAI(api_key=VAST_API_KEY).destroy_instance(VAST_INSTANCE_ID)`, only when `VAST_SELF_DESTRUCT` is set (so local runs never self-destroy).
- `push_results_and_destroy()`: push (best-effort, logged) then destroy in a `finally`, so a push hiccup still frees the box.

Detached-HEAD note: default provisioning clones at a commit sha (detached HEAD, no branch),
so pushes always use `git push origin HEAD:<results-branch>` against an explicit branch
(default = dedicated `results` branch, keeping `main` clean).

Concurrency note: distinct per-run folders remove content/merge conflicts, but NOT push
rejections — `git push` races at the branch tip, so a second box pushing onto the same
branch is rejected non-fast-forward regardless of disjoint files. The fetch+rebase+retry
loop closes that gap. Unique `--run-name` per shot is recommended and wired into the subdir
+ commit label.

Training hook (opt-in, blueprints stay inert):

- Add a tail to `scripts/train.py`: when `VAST_SELF_DESTRUCT` is set (or `--vast-teardown` passed), after train + probe complete, call `devops.vast.self_destruct.push_results_and_destroy()`. Wrap the run body in try/finally; `--teardown-on-error` controls whether failures also trigger push+destroy. The hook lives in the launcher, so every blueprint benefits with no per-blueprint edits.

## Local environment (verified on this machine)

- vast API key: read from `~/.vast_api_key` (`chmod 600` recommended). Auth resolution order: `VAST_API_KEY` env -> `~/.vast_api_key` -> `vastai` stored key.
- SSH keypair: `~/.ssh/id_rsa(.pub)` exists; tool registers `id_rsa.pub` with vast (config `SSH_KEY_PATH` defaults to it).
- GitHub: `gh` CLI authed as `Al-does` with `repo` scope. Decision: use `gh auth token` for the results push (no PAT). Token resolution: `--github-token`/`GITHUB_TOKEN` -> fall back to `gh auth token`.
- iTerm2 installed; first tab-open triggers a one-time macOS Automation permission prompt.

## Caveats to verify during build

- `torch==2.12.1` + CUDA wheels: `uv sync` on the box may need a CUDA-matched torch index; pick a base `IMAGE` whose CUDA satisfies torch, or add a torch index to the bootstrap. Confirm by running `uv sync` on a first real box.
- Exact SDK method names for SSH-key attach and the `create_instance` bid arg (`price` vs `bid_price`) will be confirmed against the installed `vastai` version's `help()`.
- Self-destruct secrets: each box holds a write-capable GitHub token (via `gh auth token`, `repo` scope — read/write to all the user's repos) and the `VAST_API_KEY`, both visible to the host. Accepted tradeoff (see decision above); neither token can delete repos. `state.json` + manual `destroy` remain the backstop if a box fails to self-destruct.
- Confirm `create_instance` forwards a custom `env`/env-vars dict to the container so the teardown env actually lands on the box.
