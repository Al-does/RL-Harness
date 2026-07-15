#!/usr/bin/env bash
# Remote bootstrap, run once on each vast box via the container onstart hook.
#
# It installs uv, clones the repo at a pinned ref, `uv sync`s the training env,
# writes the /root/.vast_ready sentinel (so the local tool only opens terminals
# / considers the box ready once the env is fully installed), and — if --run was
# given — launches the command in a detached tmux session named "run".
#
# When self-destruct is enabled it also configures a git identity and a
# token-authed `origin` so the in-run teardown hook can push results back.
#
# Inputs (container env vars, injected at provision time):
#   VAST_REPO_URL         git URL to clone (public repo -> no creds needed)
#   VAST_GIT_REF          branch name or commit sha to check out
#   VAST_RUN_CMD          optional command run in the activated .venv in tmux
#   VAST_SELF_DESTRUCT    "1" to wire git identity + token origin for teardown
#   GITHUB_TOKEN          write token for the results push (self-destruct only)
#   VAST_RESULTS_BRANCH   branch the teardown hook pushes results to
#   VAST_RUN_NAME         per-shot run label
#   GIT_USER_NAME/GIT_USER_EMAIL  commit identity for the results push
#   VAST_REPO_SLUG        owner/repo, used to build the token origin URL
#   VAST_API_KEY          vast key (self-destruct and/or max-age watchdog REST destroy)
#   VAST_MAX_AGE_S        wall-clock lifetime cap in seconds; >0 arms the watchdog
#   VAST_UV_SYNC_TIMEOUT_S maximum total seconds allowed for uv sync

set -uo pipefail

REPO_DIR="/root/RLLibHarnesBeta"
READY_SENTINEL="/root/.vast_ready"
FAIL_SENTINEL="/root/.vast_bootstrap_failed"

log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }
fail() { log "ERROR: $*"; echo "$*" > "$FAIL_SENTINEL"; exit 1; }

log "starting; ref=${VAST_GIT_REF:-<none>} self_destruct=${VAST_SELF_DESTRUCT:-0}"
log "user=$(whoami) authorized_keys=$( [ -f "$HOME/.ssh/authorized_keys" ] && wc -l < "$HOME/.ssh/authorized_keys" || echo 0 ) line(s)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | sed 's/^/[bootstrap] gpu: /' || log "nvidia-smi not found"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current \
        --format=csv,noheader 2>/dev/null | sed 's/^/[bootstrap] pcie gen,width: /' || true
fi
if [ -r /sys/fs/cgroup/cpu.max ]; then
    log "cgroup cpu.max=$(cat /sys/fs/cgroup/cpu.max)"
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    log "cgroup cpu quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)/$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)"
fi
log "host load=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo unknown)"

export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -y >/dev/null 2>&1 || true
    apt-get install -y --no-install-recommends curl git ca-certificates tmux >/dev/null 2>&1 || \
        log "apt-get install had non-zero exit (image may already provide these)"
fi

# --- uv -----------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh || fail "uv install failed"
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || fail "uv not on PATH after install"

# --- clone at ref -------------------------------------------------------
if [ ! -d "$REPO_DIR/.git" ]; then
    log "cloning $VAST_REPO_URL"
    git clone "$VAST_REPO_URL" "$REPO_DIR" || fail "git clone failed"
fi
cd "$REPO_DIR" || fail "cannot cd $REPO_DIR"
git fetch --all --tags --quiet || true
if [ -n "${VAST_GIT_REF:-}" ]; then
    log "checking out $VAST_GIT_REF"
    git checkout --quiet "$VAST_GIT_REF" || fail "git checkout $VAST_GIT_REF failed"
fi

# --- self-destruct git wiring (before sync; cheap and idempotent) -------
if [ "${VAST_SELF_DESTRUCT:-0}" = "1" ]; then
    log "configuring git identity + token origin for results push"
    git config user.name "${GIT_USER_NAME:-vast-bot}"
    git config user.email "${GIT_USER_EMAIL:-vast-bot@users.noreply.github.com}"
    if [ -n "${GITHUB_TOKEN:-}" ] && [ -n "${VAST_REPO_SLUG:-}" ]; then
        git remote set-url origin \
            "https://x-access-token:${GITHUB_TOKEN}@github.com/${VAST_REPO_SLUG}.git"
    else
        log "WARNING: self-destruct set but GITHUB_TOKEN/VAST_REPO_SLUG missing; push will be skipped"
    fi
fi

# --- max-age watchdog (hard cost cap, machine-independent) --------------
# Armed BEFORE `uv sync` on purpose: a box whose sync fails (and so lingers
# for debugging) still gets reaped at the cap. self_destruct.py is stdlib-only,
# so `uv run` here just needs a resolvable env by the time the timer fires (5h
# later the normal sync has long since finished). Detached tmux so it outlives
# both bootstrap and the run.
if [ -n "${VAST_MAX_AGE_S:-}" ] && [ "${VAST_MAX_AGE_S}" -gt 0 ] 2>/dev/null; then
    log "arming max-age watchdog: destroy this box after ${VAST_MAX_AGE_S}s"
    tmux new-session -d -s watchdog \
        "sleep ${VAST_MAX_AGE_S}; cd $REPO_DIR && export PATH=\"$HOME/.local/bin:\$PATH\"; uv run python -m devops.vast.self_destruct --max-age 2>&1 | tee /root/watchdog.log"
else
    log "max-age watchdog disabled (VAST_MAX_AGE_S unset or 0)"
fi

# --- install training env ----------------------------------------------
SYNC_TIMEOUT="${VAST_UV_SYNC_TIMEOUT_S:-1200}"
log "uv sync (timeout=${SYNC_TIMEOUT}s; downloads python + torch/CUDA wheels)"
if command -v timeout >/dev/null 2>&1; then
    timeout --foreground "${SYNC_TIMEOUT}s" uv sync
    sync_rc=$?
    if [ "$sync_rc" -eq 124 ]; then
        fail "uv sync timed out after ${SYNC_TIMEOUT}s (host network too slow)"
    elif [ "$sync_rc" -ne 0 ]; then
        fail "uv sync failed (exit $sync_rc)"
    fi
else
    uv sync || fail "uv sync failed"
fi

# --- ready --------------------------------------------------------------
uv run python - <<'PY' || fail "torch CUDA validation failed"
import torch

print(
    "torch", torch.__version__,
    "built_cuda", torch.version.cuda,
    "cuda_available", torch.cuda.is_available(),
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable despite a GPU rental; host driver/runtime is incompatible")
print("gpu", torch.cuda.get_device_name(0))
PY
touch "$READY_SENTINEL"
log "env ready -> $READY_SENTINEL"

# --- optional run -------------------------------------------------------
if [ -n "${VAST_RUN_CMD:-}" ]; then
    log "launching run in tmux: $VAST_RUN_CMD"
    tmux new-session -d -s run \
        "cd $REPO_DIR && bash devops/vast/run_remote.sh"
    log "run started in tmux session 'run' (attach with: tmux attach -t run)"
    # Surface the run's tail + exit to container stdout when it ends, so the run
    # can be monitored with `vastai logs <id>` even without SSH reachability.
    (
        while tmux has-session -t run 2>/dev/null; do sleep 15; done
        echo "[bootstrap] === run finished; tail of /root/run.log ==="
        tail -n 40 /root/run.log 2>&1
    ) &
fi

log "bootstrap complete"
