#!/usr/bin/env bash
set -euo pipefail
ROOT=__PROJECT_ROOT__/stage10_authoritative_baselines
cd "$ROOT"
mkdir -p logs runs results
exec > >(tee -a logs/run_all.log) 2>&1
trap 'echo "===== FAILED line=$LINENO time=$(date "+%F %T") ====="' ERR
MODELS=(rsiccformer mccformer_d chg2cap sen rscama)
echo "===== START $(date '+%F %T') ====="
python -m compileall -q runner vendor
PREFLIGHT=()
for model in "${MODELS[@]}"; do PREFLIGHT+=(--config "configs/${model}.json"); done
python runner/preflight.py "${PREFLIGHT[@]}"
for model in "${MODELS[@]}"; do
  echo "===== SMOKE ${model} $(date '+%F %T') ====="
  python runner/smoke_test.py --config "configs/${model}.json"
  echo "===== TRAIN ${model} $(date '+%F %T') ====="
  mkdir -p "logs/${model}"
  python -u runner/train.py --config "configs/${model}.json" --output "runs/${model}" 2>&1 | tee "logs/${model}/train.log"
  echo "===== EVAL ${model} $(date '+%F %T') ====="
  python -u runner/evaluate.py --checkpoint "runs/${model}/best.pt" --conditions all --output "results/${model}" 2>&1 | tee "logs/${model}/eval.log"
  rm -f "runs/${model}/last.pt"
done
python runner/summarize.py --results results | tee logs/summary.log
du -sh "$ROOT"
echo "===== ALL FIVE BASELINES COMPLETED $(date '+%F %T') ====="
