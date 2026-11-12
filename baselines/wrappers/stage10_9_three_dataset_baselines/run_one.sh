#!/usr/bin/env bash
set -Eeuo pipefail
DATASET=${1:?dataset}; MODEL=${2:?model}
ROOT=__PROJECT_ROOT__/stage10_9_three_dataset_baselines
CFG=$ROOT/configs/$DATASET/$MODEL.json
RUN=$ROOT/runs/$DATASET/$MODEL
RES=$ROOT/results/$DATASET/$MODEL
need(){ test -s "$1" || { echo "ERROR missing $1"; exit 1; }; }
mkdir -p "$RUN" "$RES" "$ROOT/logs/$DATASET"
cd "$ROOT"
if [ ! -f "$RUN/.complete" ]; then
  resume=(); test -s "$RUN/last.pt" && resume=(--resume "$RUN/last.pt")
  python -u runner/train.py --config "$CFG" --output "$RUN" "${resume[@]}"
  need "$RUN/best.pt"; touch "$RUN/.complete"; rm -f "$RUN/last.pt"
fi
if [ ! -f "$RES/.complete" ]; then
  python -u runner/evaluate.py --checkpoint "$RUN/best.pt" --conditions all --output "$RES"
  need "$RES/all_metrics.json"; touch "$RES/.complete"
fi
echo "RUN_OK $DATASET $MODEL"
