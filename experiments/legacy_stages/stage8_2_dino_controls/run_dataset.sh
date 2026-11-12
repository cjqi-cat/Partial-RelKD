#!/usr/bin/env bash
set -Eeuo pipefail

DATASET=${1:?Usage: bash run_dataset.sh LEVIR_MCI|SECOND_CC|WHU_CDC}
case "$DATASET" in LEVIR_MCI|SECOND_CC|WHU_CDC) ;; *) echo "Unknown dataset: $DATASET"; exit 2;; esac

ROOT=__PROJECT_ROOT__/stage8_2_dino_controls
STAGE66=__PROJECT_ROOT__/stage6_6_m2_backbones
CFG=$ROOT/configs/$DATASET
RUN=$ROOT/runs/$DATASET
RESULT=$ROOT/results/$DATASET
cd "$ROOT"
mkdir -p "$RUN" "$RESULT"
step(){ echo; echo "===== $(date '+%F %T') $DATASET START: $1 ====="; }
need(){ test -s "$1" || { echo "ERROR missing: $1"; exit 1; }; }
trap 'echo "===== FAILED dataset='$DATASET' line=$LINENO time=$(date "+%F %T") ====="' ERR

step "DINO-Clean 31-condition evaluation"
if [ ! -f "$RESULT/dino_clean/.complete" ]; then
  python -u "$STAGE66/runner/evaluate.py" --checkpoint "$RUN/dino_clean.pt" --conditions all --output "$RESULT/dino_clean"
  need "$RESULT/dino_clean/all_metrics.json"
  touch "$RESULT/dino_clean/.complete"
fi

step "DINO-Aug caption-CE-only training"
if [ ! -f "$RUN/dino_aug/.complete" ]; then
  python -u "$STAGE66/runner/train_m2.py" --config "$CFG/dino_aug.json" --variant m1 --output "$RUN/dino_aug"
  need "$RUN/dino_aug/best.pt"
  touch "$RUN/dino_aug/.complete"
  rm -f "$RUN/dino_aug/last.pt"
fi

step "DINO-Aug 31-condition evaluation"
if [ ! -f "$RESULT/dino_aug/.complete" ]; then
  python -u "$STAGE66/runner/evaluate.py" --checkpoint "$RUN/dino_aug/best.pt" --conditions all --output "$RESULT/dino_aug"
  need "$RESULT/dino_aug/all_metrics.json"
  touch "$RESULT/dino_aug/.complete"
fi

step "Control summary"
python -u runner/summarize.py --root "$ROOT" --dataset "$DATASET"
echo "===== $DATASET CONTROLS COMPLETED: $(date '+%F %T') ====="
touch "$ROOT/${DATASET}_COMPLETED"
