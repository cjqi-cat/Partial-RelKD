#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=__PROJECT_ROOT__/stage10_19_scnet_multidataset
cd "$ROOT"
trap 'code=$?; echo "===== FAILED line=$LINENO code=$code time=$(date "+%F %T") ====="; exit $code' ERR
bash run_dataset.sh SECOND_CC
bash run_dataset.sh WHU_CDC
python runner/summarize_all.py | tee logs/summary_all.log
touch SCNET_MULTIDATASET_COMPLETE
echo "===== SCNET SECOND_CC + WHU_CDC COMPLETE $(date '+%F %T') ====="
