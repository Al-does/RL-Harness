#!/usr/bin/env bash
set -uo pipefail

cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

set +e
eval "${VAST_RUN_CMD}" 2>&1 | tee /root/run.log
run_status=${PIPESTATUS[0]}
set -e
echo "EXIT=${run_status}" | tee -a /root/run.log

if [ "${VAST_SELF_DESTRUCT:-0}" = "1" ]; then
    if [ "$run_status" -eq 0 ] || [ "${VAST_TEARDOWN_ON_ERROR:-0}" = "1" ]; then
        python -m devops.vast.self_destruct 2>&1 \
            | tee -a /root/run.log
    else
        echo "run failed; leaving box up for debugging" | tee -a /root/run.log
    fi
fi

exit "$run_status"
