#!/usr/bin/env bash
set -Eeuo pipefail
DATASET=${1:?Usage: bash run_dataset.sh SECOND_CC|WHU_CDC}
case "$DATASET" in SECOND_CC|WHU_CDC) ;; *) echo "Unknown dataset: $DATASET"; exit 2;; esac
ROOT=__PROJECT_ROOT__/stage10_19_scnet_multidataset
BASE=__PROJECT_ROOT__/stage10_17_scnet_lofo
SOURCE_CFG="$ROOT/configs/$DATASET/scnet.json"
CFG="$ROOT/runtime_configs/$DATASET/scnet.json"
RUN="$ROOT/runs/$DATASET/scnet"
RESULT="$ROOT/results/$DATASET/scnet"
mkdir -p "$ROOT/logs/$DATASET" "$RUN" "$RESULT" "$(dirname "$CFG")"
cd "$ROOT"
trap 'code=$?; echo "===== FAILED dataset='$DATASET' line=$LINENO code=$code time=$(date "+%F %T") ====="; exit $code' ERR
echo "===== START SCNet $DATASET $(date '+%F %T') ====="
python "$ROOT/runner/prepare_config.py" --config "$SOURCE_CFG" --output "$CFG" 2>&1 | tee "$ROOT/logs/$DATASET/normalization.log"
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
echo "===== SCNET $DATASET COMPLETE $(date '+%F %T') ====="
