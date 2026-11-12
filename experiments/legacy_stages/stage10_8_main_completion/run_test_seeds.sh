#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=__PROJECT_ROOT__/stage10_8_main_completion;cd "$ROOT"
for seed in 20261028 20261128;do for model in change_agent p3 p3_nomask;do out="results/test_seeds/seed${seed}/${model}";python -u runner/evaluate_seed.py --model "$model" --degradation-seed "$seed" --output "$out";done;done
python -u runner/summarize_all.py --root "$ROOT"
touch TEST_SEEDS_COMPLETE
