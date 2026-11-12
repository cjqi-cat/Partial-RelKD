#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=__PROJECT_ROOT__/stage10_7_severity_curves
PROJECT=__PROJECT_ROOT__
DATA=__DATA_ROOT__/LEVIR-MCI-dataset
cd "$ROOT"
trap 'echo "===== FAILED line=$LINENO time=$(date "+%F %T") ====="' ERR
step(){ echo; echo "===== $(date '+%F %T') START: $1 ====="; }
step "preflight and severity audit"
python runner/preflight.py --root "$PROJECT"
python runner/audit_severity.py --project-root "$PROJECT" --dataset-root "$DATA" --output results/severity_audit.json
for model in change_agent p3 p3_nomask; do
  python -u runner/evaluate.py --model "$model" --root "$PROJECT" --severity 2 --conditions blur__both --limit 2 --workers 0 --batch-size 2 --output "results/_smoke/$model"
done
echo "SMOKE_OK"
for model in change_agent p3 p3_nomask; do
  step "$model clean"
  python -u runner/evaluate.py --model "$model" --root "$PROJECT" --severity 2 --conditions clean --output "results/$model/clean"
  for severity in 1 2 3 4 5; do
    step "$model severity=$severity"
    python -u runner/evaluate.py --model "$model" --root "$PROJECT" --severity "$severity" --conditions degraded --output "results/$model/severity_$severity"
  done
done
step "summary and curves"
python runner/summarize.py --root "$ROOT"
echo "===== SEVERITY CURVES COMPLETED: $(date '+%F %T') ====="
touch STAGE107_COMPLETED
