#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-__WORK_ROOT__}"
STAGE4="$ROOT/project/stage4_caption_dino_finetune"
mkdir -p "$STAGE4/logs"
if [[ -f "$STAGE4/logs/pipeline.pid" ]] && kill -0 "$(cat "$STAGE4/logs/pipeline.pid")" 2>/dev/null; then
  echo "pipeline already running pid=$(cat "$STAGE4/logs/pipeline.pid")"; exit 0
fi
nohup bash "$STAGE4/run_server.sh" "$ROOT" > "$STAGE4/logs/pipeline.log" 2>&1 &
echo $! > "$STAGE4/logs/pipeline.pid"
echo "started pid=$!"
echo "progress: bash $STAGE4/status_server.sh"
