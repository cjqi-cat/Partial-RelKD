#!/usr/bin/env bash
ROOT=__PROJECT_ROOT__/stage10_25_lchg_ablation
echo "===== processes ====="
pgrep -af 'stage10_25_lchg_ablation|run_dino_lofo.py|stage6_7_partial_hybrid/runner/train.py' || true
echo "===== completion ====="
[[ -f "$ROOT/STAGE1025_COMPLETE" ]] && echo COMPLETE || echo RUNNING_OR_NOT_STARTED
echo "===== checkpoints ====="
find "$ROOT/runs" -type f \( -name 'best.pt' -o -name 'partial_last2_best.pt' \) -printf '%p %s bytes\n' 2>/dev/null || true
echo "===== recent log ====="
tail -n 30 "$ROOT/run.log" 2>/dev/null || true
