#!/usr/bin/env bash
set -u
PROJECT_DIR="__PROJECT_ROOT__/dino_finetune_step1"
cd "$PROJECT_DIR"
echo "time: $(date '+%F %T %Z')"
if pgrep -af "runner/evaluate_fixed.py" >/dev/null; then
  echo "process: RUNNING"
else
  echo "process: NOT RUNNING"
fi
for NAME in lora partial_last2; do
  COUNT=0
  [[ -f "fixed_eval/$NAME/metrics.csv" ]] && COUNT=$(($(wc -l < "fixed_eval/$NAME/metrics.csv")-1))
  echo "$NAME: $COUNT/31"
  [[ -f "fixed_eval/$NAME/summary.json" ]] && cat "fixed_eval/$NAME/summary.json"
done
echo "--- latest ---"
tail -n 10 fixed_eval/logs/pipeline.log 2>/dev/null || true
