#!/usr/bin/env bash
ROOT=__PROJECT_ROOT__/stage10_20_rmnet_multidataset
pgrep -af 'stage10_20_rmnet_multidataset|stage10_18_rmnet_lofo/runner/(train|evaluate).py' || true
find "$ROOT" -maxdepth 4 -type f \( -name '*_COMPLETE' -o -name 'TRAINING_COMPLETE' -o -name 'EVALUATION_COMPLETE' \) -print 2>/dev/null | sort
for dataset in SECOND_CC WHU_CDC; do
  echo "===== $dataset ====="
  tail -n 12 "$ROOT/logs/$dataset/train.log" 2>/dev/null || true
  tail -n 12 "$ROOT/logs/$dataset/evaluate.log" 2>/dev/null || true
done
