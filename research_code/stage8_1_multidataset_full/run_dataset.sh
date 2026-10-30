#!/usr/bin/env bash
set -Eeuo pipefail

DATASET=${1:?Usage: bash run_dataset.sh SECOND_CC|WHU_CDC}
case "$DATASET" in SECOND_CC|WHU_CDC) ;; *) echo "Unknown dataset: $DATASET"; exit 2;; esac

ROOT=__PROJECT_ROOT__/stage8_1_multidataset_full
PROJECT=__PROJECT_ROOT__
STAGE1=$PROJECT/dino_finetune_step1
STAGE4=$PROJECT/stage4_caption_dino_finetune
STAGE66=$PROJECT/stage6_6_m2_backbones
STAGE67=$PROJECT/stage6_7_partial_hybrid
META=$ROOT/metadata/$DATASET
DATA=$ROOT/data_views/$DATASET
CFG=$ROOT/configs/$DATASET
RUN=$ROOT/runs/$DATASET
RESULT=$ROOT/results/$DATASET
CA=$PROJECT/stage3_caption_retrain/vendor/Change-Agent/Multi_change

cd "$ROOT"
mkdir -p "$RUN" "$RESULT" "$ROOT/logs"
step(){ echo; echo "===== $(date '+%F %T') $DATASET START: $1 ====="; }
need(){ test -s "$1" || { echo "ERROR missing: $1"; exit 1; }; }
done_marker(){ touch "$1/.complete"; }
trap 'echo "===== FAILED dataset='$DATASET' line=$LINENO time=$(date "+%F %T") ====="' ERR

step "Change-Agent target-caption training"
if [ ! -f "$RUN/change_agent/.complete" ]; then
  python -u runner/run_ca.py --action train --stage66-runner "$STAGE66/runner" --config "$CFG/change_agent.json" --output "$RUN/change_agent"
  need "$RUN/change_agent/best.pt"; done_marker "$RUN/change_agent"
fi
if [ ! -f "$RESULT/change_agent/.complete" ]; then
  python -u runner/run_ca.py --action evaluate --stage66-runner "$STAGE66/runner" --checkpoint "$RUN/change_agent/best.pt" --conditions all --output "$RESULT/change_agent"
  need "$RESULT/change_agent/all_metrics.json"; done_marker "$RESULT/change_agent"
fi

step "Partial-last2 robust encoder"
if [ ! -f "$RUN/partial_encoder/.complete" ]; then
  resume=(); test -s "$RUN/partial_encoder/checkpoints/last.pt" && resume=(--resume "$RUN/partial_encoder/checkpoints/last.pt")
  python -u "$STAGE66/runner/run_dino_lofo.py" --source-train "$STAGE1/runner/train.py" --config "$CFG/partial_encoder.json" --output "$RUN/partial_encoder" --family geometry_quality "${resume[@]}"
  need "$RUN/partial_encoder/checkpoints/best.pt"; done_marker "$RUN/partial_encoder"
  rm -f "$RUN/partial_encoder/checkpoints/last.pt"
fi

step "Partial clean feature cache and caption teacher"
if [ ! -s "$RUN/partial_teacher/checkpoints/partial_last2_best.pt" ]; then
  python -u "$STAGE4/runner/prepare_features.py" --variant partial_last2 --stage1-root "$STAGE1" --checkpoint "$RUN/partial_encoder/checkpoints/best.pt" --dataset-root "$DATA" --list-root "$META" --output "$RUN/partial_cache"
  python -u "$STAGE4/runner/train_caption.py" --variant partial_last2 --cache "$RUN/partial_cache" --token-root "$META/tokens" --vocab "$META/vocab.json" --change-agent-root "$CA" --output "$RUN/partial_teacher"
  need "$RUN/partial_teacher/checkpoints/partial_last2_best.pt"
  rm -rf "$RUN/partial_cache"
fi

step "M2 core"
if [ ! -f "$RUN/m2/.complete" ]; then
  python -u "$STAGE66/runner/train_m2.py" --config "$CFG/m2.json" --variant m2 --output "$RUN/m2"
  need "$RUN/m2/best.pt"; done_marker "$RUN/m2"; rm -f "$RUN/m2/last.pt"
fi
if [ ! -f "$RESULT/m2/.complete" ]; then
  python -u "$STAGE66/runner/evaluate.py" --checkpoint "$RUN/m2/best.pt" --conditions all --output "$RESULT/m2"
  need "$RESULT/m2/all_metrics.json"; done_marker "$RESULT/m2"
fi

step "P3 partial relation"
if [ ! -f "$RUN/p3/.complete" ]; then
  resume=(); test -s "$RUN/p3/last.pt" && resume=(--resume "$RUN/p3/last.pt")
  python -u "$STAGE67/runner/train.py" --config "$CFG/p3.json" --output "$RUN/p3" "${resume[@]}"
  need "$RUN/p3/best.pt"; done_marker "$RUN/p3"; rm -f "$RUN/p3/last.pt"
fi
if [ ! -f "$RESULT/p3/.complete" ]; then
  python -u "$STAGE67/runner/evaluate.py" --checkpoint "$RUN/p3/best.pt" --conditions all --output "$RESULT/p3"
  need "$RESULT/p3/all_metrics.json"; done_marker "$RESULT/p3"
fi

step "Summary"
python -u runner/summarize.py --root "$ROOT" --dataset "$DATASET"
du -sh "$ROOT"
echo "===== $DATASET COMPLETED: $(date '+%F %T') ====="
touch "$ROOT/${DATASET}_COMPLETED"
