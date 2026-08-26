#!/usr/bin/env bash
# Launch the 2-, 3-, and 5-factor factored MESS3 token experiments on separate
# on-demand Vast boxes. Provisions sequentially to avoid state.json races.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [ -z "${VAST_API_KEY:-}" ]; then
    echo "VAST_API_KEY is required" >&2
    exit 1
fi

GITHUB_TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-$(gh auth token 2>/dev/null || true)}}"
COMMON=(
    --mode ondemand
    --branch main
    --library-branch cursor/factored-hmm-analysis-07b7
    --experiment-repo "${EXPERIMENT_REPO:-/alex-rl-experiments}"
    --max-age 8
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

launch_one "2-factor" \
    "experiments.factored_mess3_beliefs_2026_08.independent_ppo.experiment"
launch_one "3-factor" \
    "experiments.factored_mess3_beliefs_2026_08.independent_three_ppo.experiment"
launch_one "5-factor" \
    "experiments.factored_mess3_beliefs_2026_08.independent_five_ppo.experiment"

echo "=== tracked boxes ==="
uv run --group devops python -m devops.vast.provision status
