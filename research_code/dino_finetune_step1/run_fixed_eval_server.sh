#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="__PROJECT_ROOT__/dino_finetune_step1"
cd "$PROJECT_DIR"
mkdir -p fixed_eval/logs
python runner/evaluate_fixed.py \
  --dataset-root __DATA_ROOT__/LEVIR-MCI-dataset \
  --manifest reference/sample_manifest.json \
  --output fixed_eval \
  --checkpoint "lora=runs/dino_lora_qkv_all/checkpoints/best.pt" \
  --checkpoint "partial_last2=runs/dino_partial_last2/checkpoints/best.pt" \
  --device cuda --batch-size 32 \
  2>&1 | tee fixed_eval/logs/pipeline.log
