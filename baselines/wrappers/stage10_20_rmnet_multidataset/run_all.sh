#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=__PROJECT_ROOT__/stage10_20_rmnet_multidataset
cd "$ROOT"
trap 'code=$?; echo "===== FAILED line=$LINENO code=$code time=$(date "+%F %T") ====="; exit $code' ERR
bash run_dataset.sh SECOND_CC
bash run_dataset.sh WHU_CDC
python runner/summarize_all.py | tee logs/summary_all.log
touch RMNET_MULTIDATASET_COMPLETE
echo "===== RMNET SECOND_CC + WHU_CDC COMPLETE $(date '+%F %T') ====="
