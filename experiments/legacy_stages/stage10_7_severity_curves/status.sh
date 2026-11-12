#!/usr/bin/env bash
ROOT=__PROJECT_ROOT__/stage10_7_severity_curves
pgrep -af 'stage10_7_severity_curves|runner/evaluate.py' || true
test -f "$ROOT/STAGE107_COMPLETED" && echo '===== completed =====' || true
find "$ROOT/results" -name '*.metrics.json' 2>/dev/null | wc -l
tail -n 30 "$ROOT/run.log" 2>/dev/null || true
