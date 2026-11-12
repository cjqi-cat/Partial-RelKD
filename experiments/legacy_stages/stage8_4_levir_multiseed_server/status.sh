#!/usr/bin/env bash
set -euo pipefail
ROOT=__PROJECT_ROOT__/stage8_4_levir_multiseed_server
echo "===== process ====="
pgrep -af 'stage8_4_levir_multiseed_server|run_dino_lofo|train_m2.py|stage6_7_partial_hybrid/runner/train.py' || true
echo "===== completed stages ====="
find "$ROOT/state" -type f -name '*.complete' -printf '%P\n' 2>/dev/null | sort || true
echo "===== latest logs ====="
find "$ROOT/logs" -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 4 | cut -d' ' -f2- | while read -r f; do echo "--- $f"; tail -n 4 "$f"; done
echo "===== disk ====="
du -sh "$ROOT" 2>/dev/null || true
df -h __STORAGE_ROOT__
