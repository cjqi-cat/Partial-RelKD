#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=__PROJECT_ROOT__/stage8_4_levir_multiseed_server
for seed in 20261018 20261028; do python "$ROOT/runner/make_configs.py" --seed "$seed"; done
python "$ROOT/runner/preflight.py"
for seed in 20261018 20261028; do bash "$ROOT/run_seed.sh" "$seed"; done
python "$ROOT/runner/summarize.py"
