#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-__WORK_ROOT__}"
S65="$ROOT/project/stage6_5_calibrated_v3"
S1="$ROOT/project/dino_finetune_step1"
S3="$ROOT/project/stage3_caption_retrain"
S4="$ROOT/project/stage4_caption_dino_finetune"
DATA="$ROOT/data/LEVIR-MCI-dataset"
DEG="$S65/degradation/calibrated_v3.py"
CA="$S3/vendor/Change-Agent/Multi_change"
LISTS="$CA/data/LEVIR_MCI"
TOKENS="$LISTS/tokens"
LOG="$S65/logs/pipeline.log"
mkdir -p "$S65/logs" "$S65/runs" "$S65/caption_results" "$S65/evaluation"

phase() {
  python "$S65/runner/make_status.py" --root "$ROOT" --phase "$1" --state running
  echo "[$(date '+%F %T')] PHASE=$1" | tee -a "$LOG"
}

require() { [[ -e "$1" ]] || { echo "MISSING: $1" | tee -a "$LOG"; exit 2; }; }
for p in "$DATA/images/train/A" "$DATA/images/val/A" "$DATA/images/test/A" "$DEG" "$S65/runner/prepare_patch_features.py" "$S1/runner/train.py" "$S3/runner/train_caption.py" "$S4/runner/train_caption.py" "$S3/caption_features/manifest.json"; do require "$p"; done

phase preflight
python "$S65/runner/preflight.py" --root "$ROOT"

SHARED="$S65/adapter_shared"
if [[ ! -f "$SHARED/patch_features/manifest.json" ]]; then
  phase adapter_feature_extraction
  python "$S65/runner/prepare_patch_features.py" --config "$S65/configs/adapter_full.json" --output "$SHARED" 2>&1 | tee -a "$LOG"
fi

for variant in control full; do
  OUT="$S65/runs/${variant}_adapter"
  CFG="$S65/configs/adapter_${variant}.json"
  if [[ ! -f "$OUT/patch_results/summary.json" ]] || ! grep -q '"status": "complete"' "$OUT/patch_results/summary.json"; then
    phase "train_${variant}_adapter"
    python "$S65/runner/train_patch_adapter.py" --config "$CFG" --features "$SHARED/patch_features" --output "$OUT" 2>&1 | tee -a "$LOG"
  fi
done

# Shared adapter features are reproducible and can occupy about 11 GB.
if [[ -f "$S65/runs/control_adapter/patch_checkpoints/best.pt" && -f "$S65/runs/full_adapter/patch_checkpoints/best.pt" ]]; then
  case "$SHARED/patch_features" in "$S65"/*) rm -rf -- "$SHARED/patch_features";; *) echo "unsafe cleanup path"; exit 3;; esac
fi

for variant in lora partial_last2; do
  OUT="$S65/runs/${variant}_encoder"
  CFG="$S65/configs/${variant}.json"
  if [[ ! -f "$OUT/results/summary.json" ]] || ! grep -q '"status": "complete"' "$OUT/results/summary.json"; then
    phase "train_${variant}_encoder"
    RESUME=(); [[ -f "$OUT/checkpoints/last.pt" ]] && RESUME=(--resume "$OUT/checkpoints/last.pt")
    python "$S1/runner/train.py" --config "$CFG" --output "$OUT" "${RESUME[@]}" 2>&1 | tee -a "$LOG"
  fi
done

phase train_caption_raw_control_full
for variant in raw control full; do
  OUT="$S65/caption_results/$variant"
  CKPT="$OUT/checkpoints/${variant}_best.pt"
  if [[ ! -f "$CKPT" ]]; then
    EXTRA=()
    [[ "$variant" == control ]] && EXTRA=(--adapter-checkpoint "$S65/runs/control_adapter/patch_checkpoints/best.pt")
    [[ "$variant" == full ]] && EXTRA=(--adapter-checkpoint "$S65/runs/full_adapter/patch_checkpoints/best.pt")
    python "$S3/runner/train_caption.py" --variant "$variant" --cache "$S3/caption_features" --token-root "$TOKENS" --vocab "$LISTS/vocab.json" --change-agent-root "$CA" --output "$OUT" --seed 20261001 "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
  fi
done

for variant in lora partial_last2; do
  FEAT="$S65/features/$variant"
  OUT="$S65/caption_results/$variant"
  CKPT="$OUT/checkpoints/${variant}_best.pt"
  ENC="$S65/runs/${variant}_encoder/checkpoints/best.pt"
  if [[ ! -f "$CKPT" ]]; then
    phase "caption_features_${variant}"
    if [[ ! -f "$FEAT/manifest.json" ]]; then
      python "$S4/runner/prepare_features.py" --variant "$variant" --stage1-root "$S1" --checkpoint "$ENC" --dataset-root "$DATA" --list-root "$LISTS" --output "$FEAT" --splits train val 2>&1 | tee -a "$LOG"
    fi
    phase "train_caption_${variant}"
    python "$S4/runner/train_caption.py" --variant "$variant" --cache "$FEAT" --token-root "$TOKENS" --vocab "$LISTS/vocab.json" --change-agent-root "$CA" --output "$OUT" --seed 20261001 2>&1 | tee -a "$LOG"
  fi
  if [[ -f "$CKPT" ]]; then case "$FEAT" in "$S65"/*) rm -rf -- "$FEAT";; *) exit 3;; esac; fi
done

phase evaluate_raw_control_full
python "$S3/runner/evaluate_degraded.py" \
  --dataset-root "$DATA" --list-root "$LISTS" --token-root "$TOKENS" --vocab "$LISTS/vocab.json" \
  --change-agent-root "$CA" --degradation-script "$DEG" \
  --raw-checkpoint "$S65/caption_results/raw/checkpoints/raw_best.pt" \
  --control-checkpoint "$S65/caption_results/control/checkpoints/control_best.pt" \
  --full-checkpoint "$S65/caption_results/full/checkpoints/full_best.pt" \
  --control-adapter "$S65/runs/control_adapter/patch_checkpoints/best.pt" \
  --full-adapter "$S65/runs/full_adapter/patch_checkpoints/best.pt" \
  --output "$S65/evaluation" 2>&1 | tee -a "$LOG"

for variant in lora partial_last2; do
  phase "evaluate_${variant}"
  python "$S4/runner/evaluate.py" --variant "$variant" --stage1-root "$S1" \
    --encoder-checkpoint "$S65/runs/${variant}_encoder/checkpoints/best.pt" \
    --caption-checkpoint "$S65/caption_results/$variant/checkpoints/${variant}_best.pt" \
    --dataset-root "$DATA" --list-root "$LISTS" --token-root "$TOKENS" --vocab "$LISTS/vocab.json" \
    --change-agent-root "$CA" --degradation-script "$DEG" --output "$S65/evaluation" 2>&1 | tee -a "$LOG"
done

phase summarize
python "$S65/runner/summarize_six_models.py" --root "$S65"
python "$S65/runner/make_status.py" --root "$ROOT" --phase complete --state complete
echo "[$(date '+%F %T')] STAGE6_5_COMPLETE" | tee -a "$LOG"
