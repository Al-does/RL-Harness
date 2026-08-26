#!/usr/bin/env bash
# Copy a pinned experiment-repo overlay from rl-harness onto the sparse clone.
set -euo pipefail

OVERLAY_NAME="${1:?usage: apply_experiment_overlay.sh <overlay-name>}"
LIBRARY_DIR="${VAST_LIBRARY_DIR:-/root/work/rl-harness}"
EXPERIMENT_DIR="${VAST_EXPERIMENT_DIR:-/root/work/alex-rl-experiments}"
OVERLAY_ROOT="$LIBRARY_DIR/devops/vast/experiment-overlays/$OVERLAY_NAME"

if [ ! -d "$OVERLAY_ROOT/experiments" ]; then
    echo "experiment overlay not found: $OVERLAY_ROOT" >&2
    exit 1
fi

echo "[overlay] applying $OVERLAY_NAME from $OVERLAY_ROOT -> $EXPERIMENT_DIR"
cp -a "$OVERLAY_ROOT/experiments/." "$EXPERIMENT_DIR/experiments/"
echo "[overlay] applied $(find "$OVERLAY_ROOT/experiments" -type f | wc -l | tr -d ' ') files"
