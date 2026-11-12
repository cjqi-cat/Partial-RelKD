#!/usr/bin/env bash
set -euo pipefail

ROOT=__PROJECT_ROOT__/stage10_25_lchg_ablation
PROJECT=__PROJECT_ROOT__
DATA=__DATA_ROOT__/LEVIR-MCI-dataset
META=$PROJECT/stage3_caption_retrain/vendor/Change-Agent/Multi_change/data/LEVIR_MCI
CA=$PROJECT/stage3_caption_retrain/vendor/Change-Agent/Multi_change
ENCODER_CFG=$ROOT/configs/encoder_wo_lchg.json
P3_CFG=$ROOT/configs/p3_wo_lchg.json
ENCODER_OUT=$ROOT/runs/encoder_wo_lchg
CACHE=$ROOT/cache/clean_wo_lchg
TEACHER_OUT=$ROOT/runs/teacher_wo_lchg
P3_OUT=$ROOT/runs/p3_wo_lchg
RESULTS=$ROOT/results/p3_wo_lchg
CONDITIONS=clean,blur__A,blur__B,blur__both,low_resolution__A,low_resolution__B,low_resolution__both,misregistration__A,misregistration__B,misregistration__both,mixed__both

cd "$ROOT"
mkdir -p logs runs cache results

echo "===== $(date '+%F %T') PREFLIGHT ====="
python runner/preflight.py

if [[ ! -f "$ENCODER_OUT/checkpoints/best.pt" ]]; then
  echo "===== $(date '+%F %T') STAGE 1: TRAIN WITHOUT L_chg ====="
  RESUME_ARGS=()
  [[ -f "$ENCODER_OUT/checkpoints/last.pt" ]] && RESUME_ARGS=(--resume "$ENCODER_OUT/checkpoints/last.pt")
  python -u "$PROJECT/stage6_6_m2_backbones/runner/run_dino_lofo.py" \
    --source-train "$PROJECT/dino_finetune_step1/runner/train.py" \
    --config "$ENCODER_CFG" \
    --output "$ENCODER_OUT" \
    --family geometry_quality \
    "${RESUME_ARGS[@]}"
fi

if [[ ! -f "$CACHE/manifest.json" ]]; then
  echo "===== $(date '+%F %T') CLEAN FEATURE CACHE ====="
  python -u "$PROJECT/stage4_caption_dino_finetune/runner/prepare_features.py" \
    --variant partial_last2 \
    --stage1-root "$PROJECT/dino_finetune_step1" \
    --checkpoint "$ENCODER_OUT/checkpoints/best.pt" \
    --dataset-root "$DATA" \
    --list-root "$META" \
    --output "$CACHE" \
    --splits train val \
    --batch-size 64 \
    --seed 20261008
fi

if [[ ! -f "$TEACHER_OUT/checkpoints/partial_last2_best.pt" ]]; then
  echo "===== $(date '+%F %T') CLEAN CAPTION TEACHER ====="
  python -u "$PROJECT/stage4_caption_dino_finetune/runner/train_caption.py" \
    --variant partial_last2 \
    --cache "$CACHE" \
    --token-root "$META/tokens" \
    --vocab "$META/vocab.json" \
    --change-agent-root "$CA" \
    --output "$TEACHER_OUT" \
    --epochs 30 \
    --patience 6 \
    --seed 20261008
fi

echo "===== $(date '+%F %T') P3 PREFLIGHT AND SMOKE ====="
python "$PROJECT/stage6_7_partial_hybrid/runner/preflight.py" --config "$P3_CFG"
python "$PROJECT/stage6_7_partial_hybrid/runner/smoke_test.py" --config "$P3_CFG"

if [[ ! -f "$P3_OUT/best.pt" ]]; then
  echo "===== $(date '+%F %T') STAGE 2: P3 WITHOUT L_chg ====="
  RESUME_ARGS=()
  [[ -f "$P3_OUT/last.pt" ]] && RESUME_ARGS=(--resume "$P3_OUT/last.pt")
  python -u "$PROJECT/stage6_7_partial_hybrid/runner/train.py" \
    --config "$P3_CFG" \
    --output "$P3_OUT" \
    "${RESUME_ARGS[@]}"
fi

echo "===== $(date '+%F %T') REDUCED EVALUATION ====="
python -u "$PROJECT/stage6_7_partial_hybrid/runner/evaluate.py" \
  --checkpoint "$P3_OUT/best.pt" \
  --conditions "$CONDITIONS" \
  --output "$RESULTS" \
  --batch-size 16

BASELINE=$PROJECT/stage6_7_partial_hybrid/results/p3_partial_relation/all_metrics.json
python runner/summarize.py \
  --new "$RESULTS/all_metrics.json" \
  --baseline "$BASELINE" \
  --output "$ROOT/results"

touch "$ROOT/STAGE1025_COMPLETE"
echo "===== LCHG ABLATION COMPLETED: $(date '+%F %T') ====="
