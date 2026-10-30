#!/usr/bin/env bash
ROOT=__PROJECT_ROOT__/stage10_5_p0_p2_nomask
pgrep -af 'stage10_5_p0_p2_nomask|stage6_7_partial_hybrid/runner/(train|evaluate).py' || true
echo '===== completed ====='
find "$ROOT" -maxdepth 3 -name .complete -o -name STAGE105_COMPLETED 2>/dev/null | sort
echo '===== recent log ====='
tail -n 40 "$ROOT/run.log" 2>/dev/null || true
