#!/usr/bin/env bash
# Config-variant benchmark for a vast.ai RTX 4090 box (run via provision --run).
#
# Runs short a_main slices (5 PPO iterations each) under different hardware
# profiles / env-runner counts and prints a per-variant summary that is
# visible in the container log (`vastai logs <id>`), ending with BENCH_DONE.
# Outputs go to /root/bench (outside the repo), so nothing lands in results/.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${BENCH_OUT:-/root/bench}"
STEPS="${BENCH_STEPS:-163840}"   # 5 iterations at train_batch=32768
mkdir -p "$OUT"

echo "[bench] nproc=$(nproc)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_variant() {
    name=$1; shift
    echo "[bench] === variant $name: $* ==="
    rm -rf "${OUT:?}/$name"
    uv run python scripts/train.py --blueprint a_main --seed 0 \
        --max-steps "$STEPS" --out "$OUT/$name" "$@" 2>&1 | tail -3
}

run_variant cuda_r8  --profile cuda4090
run_variant cuda_r4  --profile cuda4090 --env-runners 4
run_variant cpu_r8   --profile cpu --env-runners 8
if [ "$(nproc)" -ge 20 ]; then
    run_variant cuda_r16 --profile cuda4090 --env-runners 16
fi

echo "[bench] === SUMMARY (last iteration of each variant) ==="
for d in "$OUT"/*/; do
    echo "[bench] $(basename "$d"): $(tail -1 "$d/progress.jsonl" 2>/dev/null)"
done
echo "BENCH_DONE"
