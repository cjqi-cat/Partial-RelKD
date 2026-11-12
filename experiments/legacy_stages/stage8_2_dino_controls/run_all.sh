#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=__PROJECT_ROOT__/stage8_2_dino_controls
cd "$ROOT"
python -u runner/make_configs.py
python -u runner/preflight.py
python -u __PROJECT_ROOT__/stage6_6_m2_backbones/runner/smoke_test.py --config configs/SECOND_CC/dino_aug.json
bash run_dataset.sh LEVIR_MCI
bash run_dataset.sh SECOND_CC
bash run_dataset.sh WHU_CDC
du -sh "$ROOT"
echo "===== DINO CONTROLS ALL DATASETS COMPLETED: $(date '+%F %T') ====="
