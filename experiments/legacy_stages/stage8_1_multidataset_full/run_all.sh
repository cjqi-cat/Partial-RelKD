#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=__PROJECT_ROOT__/stage8_1_multidataset_full
cd "$ROOT"
python -u runner/prepare_views.py --root "$ROOT"
python -u runner/make_configs.py
python -u runner/preflight.py
python -u runner/smoke_test.py --config "$ROOT/configs/SECOND_CC/change_agent.json"
bash run_dataset.sh SECOND_CC
bash run_dataset.sh WHU_CDC
echo "===== MULTIDATASET FULL TRAINING COMPLETED: $(date '+%F %T') ====="
