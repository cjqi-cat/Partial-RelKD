#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-__WORK_ROOT__}"
S="$ROOT/project/stage6_5_calibrated_v3"
mkdir -p "$S/logs"
PID=$(cat "$S/logs/pipeline.pid" 2>/dev/null || true)
if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then echo "already running PID=$PID"; exit 0; fi
nohup bash "$S/run_server.sh" "$ROOT" > "$S/logs/pipeline.nohup.log" 2>&1 &
echo $! > "$S/logs/pipeline.pid"
nohup python "$S/runner/watch_status.py" --root "$ROOT" > "$S/logs/status_watcher.log" 2>&1 &
echo $! > "$S/logs/status_watcher.pid"
echo "started PID=$(cat "$S/logs/pipeline.pid")"
