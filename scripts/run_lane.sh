#!/bin/zsh
# Sequential experiment lane: optionally wait for a PID, then run
# train+probe for each "<blueprint>:<seed>" argument in order.
#   scripts/run_lane.sh <wait_pid|-> <bp:seed> [<bp:seed> ...]
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

wait_pid=$1; shift
if [[ "$wait_pid" != "-" ]]; then
  while kill -0 "$wait_pid" 2>/dev/null; do sleep 20; done
fi

for spec in "$@"; do
  bp="${spec%%:*}"; seed="${spec##*:}"
  run="results/$($PY -c "import sys; sys.path.insert(0,'.'); from blueprints.base import get; print(f'phase{get(\"$bp\").phase}')")/$bp/seed$seed"
  if [[ ! -f "$run/module_state_final.pt" ]]; then
    echo "=== TRAIN $bp seed$seed ==="
    $PY scripts/train.py --blueprint "$bp" --seed "$seed" > "results/lane_${bp}_s${seed}.log" 2>&1 || { echo "TRAIN FAILED $bp:$seed"; continue; }
  fi
  echo "=== PROBE $bp seed$seed ==="
  $PY scripts/probe_arm.py --run "$run" --final-only > "results/lane_probe_${bp}_s${seed}.log" 2>&1 || echo "PROBE FAILED $bp:$seed"
done
echo "LANE_DONE $*"
