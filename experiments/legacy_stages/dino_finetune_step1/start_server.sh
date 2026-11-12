#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="__PROJECT_ROOT__/dino_finetune_step1"
cd "$PROJECT_DIR"
mkdir -p logs
if pgrep -af "runner/train.py.*dino_finetune_step1" >/dev/null; then
  echo "training already running"
  pgrep -af "runner/train.py.*dino_finetune_step1"
  exit 0
fi
nohup bash "$PROJECT_DIR/run_server.sh" > "$PROJECT_DIR/logs/nohup.log" 2>&1 &
echo $! > "$PROJECT_DIR/logs/pipeline.pid"
echo "started pid=$(cat "$PROJECT_DIR/logs/pipeline.pid")"
echo "progress: bash $PROJECT_DIR/status_server.sh"
