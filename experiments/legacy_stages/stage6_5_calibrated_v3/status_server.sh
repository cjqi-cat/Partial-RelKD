#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-__WORK_ROOT__}"
S="$ROOT/project/stage6_5_calibrated_v3"
python "$S/runner/make_status.py" --root "$ROOT"
cat "$S/status.json"
echo '--- GPU ---'; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
echo '--- DISK ---'; df -h __STORAGE_ROOT__
echo '--- LOG ---'; tail -n 15 "$S/logs/pipeline.nohup.log" 2>/dev/null || true
