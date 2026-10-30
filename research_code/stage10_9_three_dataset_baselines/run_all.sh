#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=__PROJECT_ROOT__/stage10_9_three_dataset_baselines
cd "$ROOT"; mkdir -p logs
trap 'echo "===== FAILED line=$LINENO time=$(date "+%F %T") ====="' ERR
echo "===== START $(date '+%F %T') ====="
python runner/make_configs.py
python runner/preflight.py
for dataset in SECOND_CC WHU_CDC; do
  models=(chg2cap mccformer_d rsiccformer sen)
  for model in "${models[@]}"; do
    echo "===== START $dataset $model $(date '+%F %T') ====="
    bash run_one.sh "$dataset" "$model" >>"logs/${dataset}_${model}.log" 2>&1
    echo "===== DONE $dataset $model $(date '+%F %T') ====="
  done
done
python runner/summarize.py
touch STAGE109_COMPLETE
echo "===== STAGE10.9 THREE-DATASET BASELINES COMPLETED $(date '+%F %T') ====="
