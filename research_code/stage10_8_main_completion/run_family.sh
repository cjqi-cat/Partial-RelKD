#!/usr/bin/env bash
set -Eeuo pipefail
FAMILY=${1:?atmospheric|noise_loss}
case "$FAMILY" in atmospheric|noise_loss) ;; *) echo "invalid family"; exit 2;; esac
ROOT=__PROJECT_ROOT__/stage10_8_main_completion
PROJECT=__PROJECT_ROOT__
STAGE1=$PROJECT/dino_finetune_step1
STAGE4=$PROJECT/stage4_caption_dino_finetune
STAGE66=$PROJECT/stage6_6_m2_backbones
STAGE67=$PROJECT/stage6_7_partial_hybrid
CA=$PROJECT/stage3_caption_retrain/vendor/Change-Agent/Multi_change
DATA=__DATA_ROOT__/LEVIR-MCI-dataset
META=$CA/data/LEVIR_MCI
CFG=$ROOT/configs/$FAMILY; RUN=$ROOT/runs/$FAMILY; RES=$ROOT/results/lofo/$FAMILY
mkdir -p "$RUN" "$RES"
need(){ test -s "$1" || { echo "ERROR missing $1"; exit 1; }; }
mark(){ touch "$1/.complete"; }
step(){ echo "===== $(date '+%F %T') $FAMILY: $1 ====="; }
trap 'echo "===== FAILED family='$FAMILY' line=$LINENO ====="' ERR

step masked_encoder
if [ ! -f "$RUN/masked_encoder/.complete" ]; then
 resume=(); test -s "$RUN/masked_encoder/checkpoints/last.pt" && resume=(--resume "$RUN/masked_encoder/checkpoints/last.pt")
 python -u "$STAGE66/runner/run_dino_lofo.py" --source-train "$STAGE1/runner/train.py" --config "$CFG/masked_encoder.json" --output "$RUN/masked_encoder" --family "$FAMILY" "${resume[@]}"
 need "$RUN/masked_encoder/checkpoints/best.pt"; mark "$RUN/masked_encoder"; rm -f "$RUN/masked_encoder/checkpoints/last.pt"
fi
step masked_teacher
if [ ! -s "$RUN/masked_teacher/checkpoints/partial_last2_best.pt" ]; then
 python -u "$STAGE4/runner/prepare_features.py" --variant partial_last2 --stage1-root "$STAGE1" --checkpoint "$RUN/masked_encoder/checkpoints/best.pt" --dataset-root "$DATA" --list-root "$META" --output "$RUN/masked_cache"
 python -u "$ROOT/runner/train_caption_fixed_refs.py" --variant partial_last2 --cache "$RUN/masked_cache" --token-root "$META/tokens" --vocab "$META/vocab.json" --change-agent-root "$CA" --output "$RUN/masked_teacher"
 need "$RUN/masked_teacher/checkpoints/partial_last2_best.pt"; rm -rf "$RUN/masked_cache"
fi
for MODEL in dino_aug p3; do
 step "$MODEL train+eval"
 if [ ! -f "$RUN/$MODEL/.complete" ]; then
  if [ "$MODEL" = dino_aug ]; then python -u "$STAGE66/runner/train_m2.py" --config "$CFG/dino_aug.json" --variant m1 --output "$RUN/$MODEL";
  else resume=(); test -s "$RUN/$MODEL/last.pt" && resume=(--resume "$RUN/$MODEL/last.pt"); python -u "$STAGE67/runner/train.py" --config "$CFG/p3.json" --output "$RUN/$MODEL" "${resume[@]}"; fi
  need "$RUN/$MODEL/best.pt"; mark "$RUN/$MODEL"; rm -f "$RUN/$MODEL/last.pt"
 fi
 if [ ! -f "$RES/$MODEL/.complete" ]; then
  if [ "$MODEL" = dino_aug ]; then EVALUATOR="$STAGE66/runner/evaluate.py"; else EVALUATOR="$STAGE67/runner/evaluate.py"; fi
  python -u "$EVALUATOR" --checkpoint "$RUN/$MODEL/best.pt" --conditions all --output "$RES/$MODEL"
  need "$RES/$MODEL/all_metrics.json"; mark "$RES/$MODEL"
 fi
done

step nomask_encoder
if [ ! -f "$RUN/nomask_encoder/.complete" ]; then
 resume=(); test -s "$RUN/nomask_encoder/checkpoints/last.pt" && resume=(--resume "$RUN/nomask_encoder/checkpoints/last.pt")
 python -u "$ROOT/runner/train_encoder_nomask_generic.py" --config "$CFG/nomask_encoder.json" --output "$RUN/nomask_encoder" "${resume[@]}"
 need "$RUN/nomask_encoder/checkpoints/best.pt"; mark "$RUN/nomask_encoder"; rm -f "$RUN/nomask_encoder/checkpoints/last.pt"
fi
step nomask_teacher
if [ ! -s "$RUN/nomask_teacher/checkpoints/partial_last2_best.pt" ]; then
 python -u "$STAGE4/runner/prepare_features.py" --variant partial_last2 --stage1-root "$PROJECT/stage10_3_p3_nomask" --checkpoint "$RUN/nomask_encoder/checkpoints/best.pt" --dataset-root "$DATA" --list-root "$META" --output "$RUN/nomask_cache"
 python -u "$ROOT/runner/train_caption_fixed_refs.py" --variant partial_last2 --cache "$RUN/nomask_cache" --token-root "$META/tokens" --vocab "$META/vocab.json" --change-agent-root "$CA" --output "$RUN/nomask_teacher"
 need "$RUN/nomask_teacher/checkpoints/partial_last2_best.pt"; rm -rf "$RUN/nomask_cache"
fi
step p3_nomask
if [ ! -f "$RUN/p3_nomask/.complete" ]; then
 resume=(); test -s "$RUN/p3_nomask/last.pt" && resume=(--resume "$RUN/p3_nomask/last.pt")
 python -u "$STAGE67/runner/train.py" --config "$CFG/p3_nomask.json" --output "$RUN/p3_nomask" "${resume[@]}"
 need "$RUN/p3_nomask/best.pt"; mark "$RUN/p3_nomask"; rm -f "$RUN/p3_nomask/last.pt"
fi
if [ ! -f "$RES/p3_nomask/.complete" ]; then python -u "$STAGE67/runner/evaluate.py" --checkpoint "$RUN/p3_nomask/best.pt" --conditions all --output "$RES/p3_nomask"; need "$RES/p3_nomask/all_metrics.json"; mark "$RES/p3_nomask"; fi
touch "$ROOT/${FAMILY}_COMPLETE"
