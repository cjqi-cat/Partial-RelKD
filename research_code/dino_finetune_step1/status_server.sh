#!/usr/bin/env bash
set -u
PROJECT_DIR="__PROJECT_ROOT__/dino_finetune_step1"
echo "time: $(date '+%F %T %Z')"
if pgrep -af "runner/train.py.*dino_finetune_step1" >/dev/null; then
  echo "process: RUNNING"
  pgrep -af "runner/train.py.*dino_finetune_step1"
else
  echo "process: NOT RUNNING"
fi
for NAME in dino_lora_qkv_all dino_partial_last2; do
  SUMMARY="$PROJECT_DIR/runs/$NAME/results/summary.json"
  if [[ -f "$SUMMARY" ]]; then
    echo "$NAME:"
    cat "$SUMMARY"
  else
    echo "$NAME: PENDING"
  fi
done
echo "--- latest log ---"
tail -n 12 "$PROJECT_DIR/logs/pipeline.log" 2>/dev/null || true
