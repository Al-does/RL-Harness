#!/usr/bin/env bash
# Launch four 50M-step factored MESS3 runs on separate Vast boxes:
#   2-factor / 3-factor / 5-factor (64-dim) + 5-factor (120-dim paper width).
#
# On Cursor Cloud agents, dashboard secrets (VAST_API_KEY, GH_TOKEN) are injected
# into tmux sessions but not the direct shell. Run inside tmux:
#   tmux -f /exec-daemon/tmux.portal.conf new-session -d -s vast-quad-50m \
#     'bash devops/vast/launch_factored_mess3_quad_50m.sh | tee /tmp/vast_quad_50m.log'
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mkdir -p "$HOME/.ssh"
if [ ! -f "$HOME/.ssh/id_rsa" ]; then
    ssh-keygen -t rsa -b 4096 -N '' -f "$HOME/.ssh/id_rsa"
fi

if [ -z "${VAST_API_KEY:-}" ]; then
    echo "VAST_API_KEY is required (use a tmux session on Cloud agents)" >&2
    exit 1
fi

GITHUB_TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-$(gh auth token 2>/dev/null || true)}}"
COMMON=(
    --mode ondemand
    --branch main
    --library-branch cursor/factored-hmm-analysis-07b7
    --experiment-repo "${EXPERIMENT_REPO:-/alex-rl-experiments}"
    --max-age 48
    --yes
    --no-open
)
if [ -n "$GITHUB_TOKEN" ]; then
    COMMON+=(--github-token "$GITHUB_TOKEN")
fi

OVERLAY='bash /root/work/rl-harness/devops/vast/apply_experiment_overlay.sh factored-hmm-validation-07b7 &&'

launch_one() {
    local label="$1"
    local module="$2"
    echo "=== provisioning $label ==="
    uv run --group devops python -m devops.vast.provision up -n 1 \
        "${COMMON[@]}" \
        --run "${OVERLAY} rl-harness ${module} --seed 42"
}

launch_one "2-factor-64d-50m" \
    "experiments.factored_mess3_beliefs_2026_08.independent_ppo_64d_50m.experiment"
launch_one "3-factor-64d-50m" \
    "experiments.factored_mess3_beliefs_2026_08.independent_three_ppo_64d_50m.experiment"
launch_one "5-factor-64d-50m" \
    "experiments.factored_mess3_beliefs_2026_08.independent_five_ppo_64d_50m.experiment"
launch_one "5-factor-120d-50m" \
    "experiments.factored_mess3_beliefs_2026_08.independent_five_ppo_120d_50m.experiment"

echo "=== tracked boxes ==="
uv run --group devops python -m devops.vast.provision status
