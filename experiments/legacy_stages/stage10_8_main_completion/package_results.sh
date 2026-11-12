#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=__PROJECT_ROOT__/stage10_8_main_completion
cd "$(dirname "$ROOT")"
tar -czf stage10_8_main_completion_results.tar.gz \
  --exclude='*.pt' --exclude='*.npy' --exclude='__pycache__' \
  stage10_8_main_completion/results stage10_8_main_completion/logs \
  stage10_8_main_completion/*_COMPLETE stage10_8_main_completion/README.md
sha256sum stage10_8_main_completion_results.tar.gz
ls -lh stage10_8_main_completion_results.tar.gz
