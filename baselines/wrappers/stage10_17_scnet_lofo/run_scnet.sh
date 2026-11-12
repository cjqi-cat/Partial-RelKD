#!/usr/bin/env bash
set -euo pipefail
trap 'echo "===== FAILED line=$LINENO code=$? time=$(date "+%F %T") ====="' ERR
ROOT=$(cd "$(dirname "$0")" && pwd); cd "$ROOT"; mkdir -p logs runs results
export PYTHONPATH="$ROOT/runner:$ROOT/vendor/SCNet:${PYTHONPATH:-}"
echo "===== START SCNet $(date '+%F %T') ====="
python -m compileall -q runner vendor/SCNet/model
python runner/preflight.py --config configs/LEVIR_MCI/scnet.json
python runner/smoke_test.py --config configs/LEVIR_MCI/scnet.json
python -u runner/train.py --config configs/LEVIR_MCI/scnet.json --output runs/scnet 2>&1 | tee logs/train.log
python -u runner/evaluate.py --checkpoint runs/scnet/best.pt --output results/scnet 2>&1 | tee logs/eval.log
python runner/summarize.py --input results/scnet --output results/summary.md | tee logs/summary.log
touch SCNET_COMPLETE
echo "===== SCNET COMPLETE $(date '+%F %T') ====="
