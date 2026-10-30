#!/usr/bin/env bash
ROOT=__PROJECT_ROOT__/stage10_3_p3_nomask
pgrep -af 'stage10_3_p3_nomask|train_encoder_nomask|stage6_7_partial_hybrid/runner/train.py' || true
echo '===== completion ====='
test -f "$ROOT/STAGE103_COMPLETED" && echo COMPLETE || echo RUNNING_OR_NOT_STARTED
echo '===== recent log ====='
tail -n 30 "$ROOT/run.log" 2>/dev/null || true
