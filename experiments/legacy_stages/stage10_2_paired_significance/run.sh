#!/usr/bin/env bash
set -euo pipefail

ROOT=__PROJECT_ROOT__/stage10_2_paired_significance
cd "$ROOT"

python -m compileall -q runner
python runner/preflight.py
python -u runner/bootstrap_p3.py --reps 10000 --seed 20261016 --output results

echo "===== PAIRED SIGNIFICANCE COMPLETED: $(date '+%F %T') ====="
