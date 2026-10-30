#!/usr/bin/env bash
ROOT="${1:-__WORK_ROOT__}"
STAGE4="$ROOT/project/stage4_caption_dino_finetune"
echo "time: $(date '+%F %T %Z')"
if [[ -f "$STAGE4/logs/pipeline.pid" ]] && kill -0 "$(cat "$STAGE4/logs/pipeline.pid")" 2>/dev/null; then echo "process: RUNNING pid=$(cat "$STAGE4/logs/pipeline.pid")"; else echo "process: NOT RUNNING"; fi
for VARIANT in lora partial_last2; do
  [[ -f "$STAGE4/features/$VARIANT/manifest.json" ]] && FEATURE=DONE || FEATURE=PENDING
  [[ -f "$STAGE4/results/$VARIANT/checkpoints/${VARIANT}_best.pt" ]] && TRAIN=DONE_OR_BEST_EXISTS || TRAIN=PENDING
  COUNT=$(find "$STAGE4/evaluation/results/$VARIANT" -name metrics.json 2>/dev/null | wc -l)
  echo "$VARIANT feature=$FEATURE caption=$TRAIN evaluation=$COUNT/31"
done
echo "--- latest ---"
tail -n 12 "$STAGE4/logs/pipeline.log" 2>/dev/null || true
