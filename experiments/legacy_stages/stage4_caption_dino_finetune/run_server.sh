#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-__WORK_ROOT__}"
STAGE1="$ROOT/project/dino_finetune_step1"
STAGE3="$ROOT/project/stage3_caption_retrain"
STAGE4="$ROOT/project/stage4_caption_dino_finetune"
DATASET="$ROOT/data/LEVIR-MCI-dataset"
CA="$STAGE3/vendor/Change-Agent/Multi_change"
LISTS="$CA/data/LEVIR_MCI"
TOKENS="$LISTS/tokens"
mkdir -p "$STAGE4/logs" "$STAGE4/features" "$STAGE4/results" "$STAGE4/evaluation"

for VARIANT in lora partial_last2; do
  if [[ "$VARIANT" == "lora" ]]; then
    ENCODER="$STAGE1/runs/dino_lora_qkv_all/checkpoints/best.pt"
  else
    ENCODER="$STAGE1/runs/dino_partial_last2/checkpoints/best.pt"
  fi
  CACHE="$STAGE4/features/$VARIANT"
  RESULT="$STAGE4/results/$VARIANT"
  if [[ ! -f "$CACHE/manifest.json" ]]; then
    python "$STAGE4/runner/prepare_features.py" --variant "$VARIANT" --stage1-root "$STAGE1" --checkpoint "$ENCODER" --dataset-root "$DATASET" --list-root "$LISTS" --output "$CACHE" 2>&1 | tee "$STAGE4/logs/01_features_${VARIANT}.log"
  else
    echo "$VARIANT feature cache exists; skip extraction"
  fi
  if [[ ! -f "$RESULT/checkpoints/${VARIANT}_best.pt" ]]; then
    python "$STAGE4/runner/train_caption.py" --variant "$VARIANT" --cache "$CACHE" --token-root "$TOKENS" --vocab "$LISTS/vocab.json" --change-agent-root "$CA" --output "$RESULT" 2>&1 | tee "$STAGE4/logs/02_train_${VARIANT}.log"
  else
    echo "$VARIANT caption checkpoint exists; skip training"
  fi
done

for VARIANT in lora partial_last2; do
  if [[ "$VARIANT" == "lora" ]]; then
    ENCODER="$STAGE1/runs/dino_lora_qkv_all/checkpoints/best.pt"
  else
    ENCODER="$STAGE1/runs/dino_partial_last2/checkpoints/best.pt"
  fi
  python "$STAGE4/runner/evaluate.py" --variant "$VARIANT" --stage1-root "$STAGE1" --encoder-checkpoint "$ENCODER" --caption-checkpoint "$STAGE4/results/$VARIANT/checkpoints/${VARIANT}_best.pt" --dataset-root "$DATASET" --list-root "$LISTS" --token-root "$TOKENS" --vocab "$LISTS/vocab.json" --change-agent-root "$CA" --degradation-script "$STAGE1/runner/degradations.py" --output "$STAGE4/evaluation" 2>&1 | tee "$STAGE4/logs/03_eval_${VARIANT}.log"
done
echo "[$(date '+%F %T')] stage4 all completed" | tee -a "$STAGE4/logs/pipeline.log"
