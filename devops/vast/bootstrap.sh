#!/usr/bin/env bash
# Remote bootstrap, run once on each vast box via the container onstart hook.
#
# Clones the personal experiment repo (results push target) and the shared
# rl-harness library as siblings, editable-installs the library into the
# experiment env, writes /root/.vast_ready, and optionally launches --run in tmux.
#
# Inputs (container env vars, injected at provision time):
#   VAST_EXPERIMENT_REPO_URL   git URL for the science repo
#   VAST_EXPERIMENT_REPO_SLUG  owner/repo for token origin / results push
#   VAST_EXPERIMENT_GIT_REF    branch or sha for the experiment repo
#   VAST_LIBRARY_REPO_URL      git URL for rl-harness
#   VAST_LIBRARY_GIT_REF       branch or sha for the library (default: main)
#   VAST_RUN_CMD               optional command run in the activated .venv in tmux
#   VAST_SELF_DESTRUCT         "1" to wire git identity + token origin for teardown
#   GITHUB_TOKEN               write token for the results push (self-destruct only)
#   VAST_RESULTS_BRANCH        branch the teardown hook pushes results to
#   VAST_RUN_NAME              per-shot run label
#   GIT_USER_NAME/GIT_USER_EMAIL  commit identity for the results push
#   VAST_API_KEY               vast key (self-destruct and/or max-age watchdog)
#   VAST_MAX_AGE_S             wall-clock lifetime cap in seconds; >0 arms watchdog
#   VAST_UV_SYNC_TIMEOUT_S     maximum total seconds allowed for uv sync
#   B2_BUCKET/B2_ENDPOINT/B2_APPLICATION_KEY_ID/B2_APPLICATION_KEY/B2_PREFIX
#                              optional artifact upload credentials (injected by provision)
#
# Legacy aliases (still accepted):
#   VAST_REPO_URL / VAST_REPO_SLUG / VAST_GIT_REF -> experiment repo fields

set -uo pipefail

WORK_DIR="/root/work"
LIBRARY_DIR="$WORK_DIR/rl-harness"
READY_SENTINEL="/root/.vast_ready"
FAIL_SENTINEL="/root/.vast_bootstrap_failed"

EXPERIMENT_URL="${VAST_EXPERIMENT_REPO_URL:-${VAST_REPO_URL:-}}"
EXPERIMENT_SLUG="${VAST_EXPERIMENT_REPO_SLUG:-${VAST_REPO_SLUG:-}}"
EXPERIMENT_REF="${VAST_EXPERIMENT_GIT_REF:-${VAST_GIT_REF:-}}"
LIBRARY_URL="${VAST_LIBRARY_REPO_URL:-https://github.com/Al-does/RL-Harness.git}"
LIBRARY_REF="${VAST_LIBRARY_GIT_REF:-main}"
EXPERIMENT_NAME="$(basename "${EXPERIMENT_URL%.git}")"
EXPERIMENT_DIR="$WORK_DIR/${EXPERIMENT_NAME:-alex-rl-experiments}"

log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }
fail() { log "ERROR: $*"; echo "$*" > "$FAIL_SENTINEL"; exit 1; }

log "starting; experiment_ref=${EXPERIMENT_REF:-<none>} library_ref=${LIBRARY_REF} self_destruct=${VAST_SELF_DESTRUCT:-0}"
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

[ -n "$EXPERIMENT_URL" ] || fail "VAST_EXPERIMENT_REPO_URL (or VAST_REPO_URL) is required"

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

mkdir -p "$WORK_DIR"

# --- clone library ------------------------------------------------------
if [ ! -d "$LIBRARY_DIR/.git" ]; then
    log "cloning library $LIBRARY_URL"
    git clone --depth 1 "$LIBRARY_URL" "$LIBRARY_DIR" || fail "library git clone failed"
fi
cd "$LIBRARY_DIR" || fail "cannot cd $LIBRARY_DIR"
log "fetching library ref $LIBRARY_REF"
git fetch --depth 1 origin "$LIBRARY_REF" || fail "library git fetch $LIBRARY_REF failed"
git checkout --quiet --detach FETCH_HEAD || fail "library git checkout $LIBRARY_REF failed"

# --- clone experiment repo ----------------------------------------------
EXPERIMENT_CLONE_URL="$EXPERIMENT_URL"
if [ -n "${GITHUB_TOKEN:-}" ] && [ -n "$EXPERIMENT_SLUG" ]; then
    EXPERIMENT_CLONE_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${EXPERIMENT_SLUG}.git"
fi
if [ ! -d "$EXPERIMENT_DIR/.git" ]; then
    log "cloning experiment repo without historical result blobs"
    git clone --depth 1 --filter=blob:none --sparse --no-checkout \
        "$EXPERIMENT_CLONE_URL" "$EXPERIMENT_DIR" || fail "experiment git clone failed"
fi
cd "$EXPERIMENT_DIR" || fail "cannot cd $EXPERIMENT_DIR"
git sparse-checkout set --cone \
    experiments scripts tests pyproject.toml README.md .gitignore uv.lock \
    || fail "experiment sparse checkout configuration failed"
if [ -n "$EXPERIMENT_REF" ]; then
    log "fetching experiment ref $EXPERIMENT_REF"
    git fetch --depth 1 origin "$EXPERIMENT_REF" || fail "experiment git fetch $EXPERIMENT_REF failed"
    git checkout --quiet --detach FETCH_HEAD || fail "experiment git checkout $EXPERIMENT_REF failed"
else
    git checkout --quiet || fail "experiment git checkout default branch failed"
fi

# --- self-destruct git wiring (experiment repo is the push target) -------
if [ "${VAST_SELF_DESTRUCT:-0}" = "1" ]; then
    log "configuring git identity + token origin for results push"
    git config user.name "${GIT_USER_NAME:-vast-bot}"
    git config user.email "${GIT_USER_EMAIL:-vast-bot@users.noreply.github.com}"
    if [ -n "${GITHUB_TOKEN:-}" ] && [ -n "$EXPERIMENT_SLUG" ]; then
        git remote set-url origin \
            "https://x-access-token:${GITHUB_TOKEN}@github.com/${EXPERIMENT_SLUG}.git"
    else
        log "WARNING: self-destruct set but GITHUB_TOKEN/VAST_EXPERIMENT_REPO_SLUG missing; push will be skipped"
    fi
fi

# --- max-age watchdog ---------------------------------------------------
if [ -n "${VAST_MAX_AGE_S:-}" ] && [ "${VAST_MAX_AGE_S}" -gt 0 ] 2>/dev/null; then
    log "arming max-age watchdog: destroy this box after ${VAST_MAX_AGE_S}s"
    tmux new-session -d -s watchdog \
        "sleep ${VAST_MAX_AGE_S}; cd $EXPERIMENT_DIR && export PATH=\"$HOME/.local/bin:\$PATH\"; uv run python -m devops.vast.self_destruct --max-age 2>&1 | tee /root/watchdog.log"
else
    log "max-age watchdog disabled (VAST_MAX_AGE_S unset or 0)"
fi

# --- install training env (experiment repo + editable sibling library) ---
SYNC_TIMEOUT="${VAST_UV_SYNC_TIMEOUT_S:-1200}"
STALL_S="${VAST_UV_SYNC_STALL_S:-480}"
UV_LOG="/root/uv_sync.log"
: > "$UV_LOG"
log "uv sync in $EXPERIMENT_DIR (timeout=${SYNC_TIMEOUT}s stall=${STALL_S}s)"
cd "$EXPERIMENT_DIR" || fail "cannot cd $EXPERIMENT_DIR"
uv sync > >(tee -a "$UV_LOG") 2>&1 &
UV_PID=$!
START_TS=$(date +%s)
LAST_SIZE=0
LAST_CHANGE=$START_TS
while kill -0 "$UV_PID" 2>/dev/null; do
    NOW=$(date +%s)
    SIZE=$(wc -c < "$UV_LOG" 2>/dev/null | tr -d ' ' || echo 0)
    if [ "${SIZE:-0}" -gt "$LAST_SIZE" ]; then
        LAST_SIZE=$SIZE
        LAST_CHANGE=$NOW
    fi
    if [ $((NOW - START_TS)) -ge "$SYNC_TIMEOUT" ]; then
        kill "$UV_PID" 2>/dev/null || true
        wait "$UV_PID" 2>/dev/null || true
        fail "uv sync timed out after ${SYNC_TIMEOUT}s (host network too slow)"
    fi
    if [ $((NOW - LAST_CHANGE)) -ge "$STALL_S" ]; then
        kill "$UV_PID" 2>/dev/null || true
        wait "$UV_PID" 2>/dev/null || true
        fail "uv sync stalled for ${STALL_S}s (no log progress; host network too slow)"
    fi
    sleep 15
done
wait "$UV_PID"
sync_rc=$?
if [ "$sync_rc" -ne 0 ]; then
    fail "uv sync failed (exit $sync_rc)"
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
    # run_remote.sh lives in the library; execute with experiment cwd via env.
    tmux new-session -d -s run \
        "cd $EXPERIMENT_DIR && bash $LIBRARY_DIR/devops/vast/run_remote.sh"
    log "run started in tmux session 'run' (attach with: tmux attach -t run)"
    (
        while tmux has-session -t run 2>/dev/null; do sleep 15; done
        echo "[bootstrap] === run finished; tail of /root/run.log ==="
        tail -n 40 /root/run.log 2>&1
    ) &
fi

log "bootstrap complete"
