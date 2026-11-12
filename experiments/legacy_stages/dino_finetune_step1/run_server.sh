#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="__PROJECT_ROOT__/dino_finetune_step1"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$PROJECT_DIR"
mkdir -p logs runs

echo "[$(date '+%F %T')] smoke test" | tee -a logs/pipeline.log
"$PYTHON_BIN" runner/self_test.py 2>&1 | tee -a logs/pipeline.log

for CONFIG in configs/lora.json configs/partial_last2.json; do
  RUN_NAME=$("$PYTHON_BIN" -c "import json;print(json.load(open('$CONFIG'))['run_name'])")
  OUTPUT="$PROJECT_DIR/runs/$RUN_NAME"
  if [[ -f "$OUTPUT/results/summary.json" ]] && grep -q '"status": "complete"' "$OUTPUT/results/summary.json"; then
    echo "[$(date '+%F %T')] skip completed $RUN_NAME" | tee -a logs/pipeline.log
    continue
  fi
  RESUME_ARGS=()
  if [[ -f "$OUTPUT/checkpoints/last.pt" ]]; then
    RESUME_ARGS=(--resume "$OUTPUT/checkpoints/last.pt")
  fi
  echo "[$(date '+%F %T')] start $RUN_NAME" | tee -a logs/pipeline.log
  "$PYTHON_BIN" runner/train.py --config "$CONFIG" --output "$OUTPUT" "${RESUME_ARGS[@]}" 2>&1 | tee -a logs/pipeline.log
done

echo "[$(date '+%F %T')] all training runs completed" | tee -a logs/pipeline.log
