#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=__PROJECT_ROOT__/stage10_8_main_completion;cd "$ROOT";mkdir -p logs
python -u runner/efficiency.py | tee logs/efficiency.log
python -u runner/qualitative.py | tee logs/qualitative.log
python -u runner/summarize_all.py --root "$ROOT" | tee logs/summary.log
echo POSTHOC_OK
