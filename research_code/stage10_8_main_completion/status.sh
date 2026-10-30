#!/usr/bin/env bash
ROOT=__PROJECT_ROOT__/stage10_8_main_completion
pgrep -af 'stage10_8_main_completion|run_dino_lofo|train_m2.py' || true
echo '===== markers =====';find "$ROOT" -name .complete -o -name '*_COMPLETE' | sort | tail -40
echo '===== logs =====';for f in "$ROOT"/logs/*.log;do echo "--- $f";tail -3 "$f";done
