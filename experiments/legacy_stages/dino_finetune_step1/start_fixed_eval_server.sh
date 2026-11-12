#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="__PROJECT_ROOT__/dino_finetune_step1"
cd "$PROJECT_DIR"
mkdir -p fixed_eval/logs
if pgrep -af "runner/evaluate_fixed.py" >/dev/null; then
  echo "fixed evaluation already running"
  pgrep -af "runner/evaluate_fixed.py"
  exit 0
fi
nohup bash "$PROJECT_DIR/run_fixed_eval_server.sh" > "$PROJECT_DIR/fixed_eval/logs/nohup.log" 2>&1 &
echo $! > "$PROJECT_DIR/fixed_eval/logs/pipeline.pid"
echo "started pid=$(cat "$PROJECT_DIR/fixed_eval/logs/pipeline.pid")"
