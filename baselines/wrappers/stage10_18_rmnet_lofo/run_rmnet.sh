#!/usr/bin/env bash
set -euo pipefail
trap 'echo "===== FAILED line=$LINENO code=$? time=$(date "+%F %T") ====="' ERR
ROOT=$(cd "$(dirname "$0")" && pwd); cd "$ROOT"; mkdir -p logs runs results
export PYTHONPATH="$ROOT/runner:$ROOT/vendor/RMNet:${PYTHONPATH:-}"
echo "===== START RMNet $(date '+%F %T') ====="
python -m compileall -q runner vendor/RMNet
python runner/preflight.py --config configs/LEVIR_MCI/rmnet.json
python runner/smoke_test.py --config configs/LEVIR_MCI/rmnet.json
python -u runner/train.py --config configs/LEVIR_MCI/rmnet.json --output runs/rmnet 2>&1 | tee logs/train.log
python -u runner/evaluate.py --checkpoint runs/rmnet/best.pt --output results/rmnet 2>&1 | tee logs/eval.log
python runner/summarize.py --input results/rmnet --output results/summary.md | tee logs/summary.log
touch RMNET_COMPLETE
echo "===== RMNET COMPLETE $(date '+%F %T') ====="
