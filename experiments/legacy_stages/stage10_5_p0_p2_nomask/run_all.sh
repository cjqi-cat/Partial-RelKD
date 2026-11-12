#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=__PROJECT_ROOT__/stage10_5_p0_p2_nomask
S67=__PROJECT_ROOT__/stage6_7_partial_hybrid
S103=__PROJECT_ROOT__/stage10_3_p3_nomask
cd "$ROOT"

step(){ echo; echo "===== $(date '+%F %T') START: $1 ====="; }
need(){ test -s "$1" || { echo "ERROR missing $1"; exit 1; }; }
trap 'echo "===== FAILED line=$LINENO time=$(date "+%F %T") ====="' ERR

train(){
  local model=$1 cfg="configs/${1}_nomask.json" out="runs/${1}_nomask"
  if [ -f "$out/.complete" ]; then need "$out/best.pt"; echo "SKIP training $model"; return; fi
  mkdir -p "$out" "logs/${1}_nomask"
  local resume=()
  if [ -s "$out/last.pt" ]; then resume=(--resume "$out/last.pt"); fi
  python -u "$S67/runner/train.py" --config "$cfg" --output "$out" "${resume[@]}" 2>&1 | tee "logs/${1}_nomask/train.log"
  need "$out/best.pt"
  touch "$out/.complete"
}

evaluate(){
  local model=$1 out="results/${1}_nomask"
  if [ -f "$out/.complete" ]; then need "$out/all_metrics.json"; echo "SKIP evaluation $model"; return; fi
  mkdir -p "$out" "logs/${1}_nomask"
  python -u "$S67/runner/evaluate.py" --checkpoint "runs/${1}_nomask/best.pt" --conditions all --output "$out" 2>&1 | tee "logs/${1}_nomask/eval.log"
  need "$out/all_metrics.json"
  test "$(find "$out" -maxdepth 1 -name '*.metrics.json' | wc -l)" -eq 31
  test "$(find "$out" -maxdepth 1 -name '*.predictions.json' | wc -l)" -eq 31
  touch "$out/.complete"
}

step "strict mask-free preflight"
python runner/preflight.py --root "$ROOT"

step "P0/P1/P2 smoke tests"
for model in p0 p1 p2; do
  python "$S67/runner/preflight.py" --config "configs/${model}_nomask.json"
  python "$S67/runner/smoke_test.py" --config "configs/${model}_nomask.json"
done

for model in p0 p1 p2; do
  step "${model^^}-NoMask training"
  train "$model"
  step "${model^^}-NoMask 31-condition evaluation"
  evaluate "$model"
done

step "P0-P3 no-mask ablation summary"
python runner/summarize.py --root "$ROOT" --stage103 "$S103"
echo "===== P0-P2 NOMASK COMPLETED: $(date '+%F %T') ====="
touch STAGE105_COMPLETED
