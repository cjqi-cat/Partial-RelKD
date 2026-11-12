#!/usr/bin/env bash
set -Eeuo pipefail
DATASET=${1:?Usage: bash run_dataset.sh SECOND_CC|WHU_CDC}
case "$DATASET" in SECOND_CC|WHU_CDC) ;; *) echo "Unknown dataset: $DATASET"; exit 2;; esac
ROOT=__PROJECT_ROOT__/stage10_20_rmnet_multidataset
BASE=__PROJECT_ROOT__/stage10_18_rmnet_lofo
CFG="$ROOT/configs/$DATASET/rmnet.json"
RUN="$ROOT/runs/$DATASET/rmnet"
RESULT="$ROOT/results/$DATASET/rmnet"
mkdir -p "$ROOT/logs/$DATASET" "$RUN" "$RESULT"
cd "$ROOT"
export PYTHONPATH="$BASE/runner:$BASE/vendor/RMNet:${PYTHONPATH:-}"
trap 'code=$?; echo "===== FAILED dataset='$DATASET' line=$LINENO code=$code time=$(date "+%F %T") ====="; exit $code' ERR
echo "===== START RMNet $DATASET $(date '+%F %T') ====="
python "$BASE/runner/preflight.py" --config "$CFG"
python "$BASE/runner/smoke_test.py" --config "$CFG"
if [[ ! -f "$RUN/TRAINING_COMPLETE" ]]; then
  resume=()
  if [[ -f "$RUN/last.pt" ]]; then resume=(--resume "$RUN/last.pt"); fi
  python -u "$BASE/runner/train.py" --config "$CFG" --output "$RUN" "${resume[@]}" 2>&1 | tee "$ROOT/logs/$DATASET/train.log"
  touch "$RUN/TRAINING_COMPLETE"
fi
if [[ ! -f "$RESULT/EVALUATION_COMPLETE" ]]; then
  python -u "$BASE/runner/evaluate.py" --checkpoint "$RUN/best.pt" --output "$RESULT" 2>&1 | tee "$ROOT/logs/$DATASET/evaluate.log"
  touch "$RESULT/EVALUATION_COMPLETE"
fi
python "$ROOT/runner/summarize.py" --dataset "$DATASET" --input "$RESULT" --output "$ROOT/results/$DATASET/summary.md" 2>&1 | tee "$ROOT/logs/$DATASET/summary.log"
touch "$ROOT/${DATASET}_COMPLETE"
echo "===== RMNET $DATASET COMPLETE $(date '+%F %T') ====="
