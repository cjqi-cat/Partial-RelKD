#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=__PROJECT_ROOT__/stage10_3_p3_nomask
S4=__PROJECT_ROOT__/stage4_caption_dino_finetune
S67=__PROJECT_ROOT__/stage6_7_partial_hybrid
DATA=__DATA_ROOT__/LEVIR-MCI-dataset
META=__CHANGE_AGENT_ROOT__/data/LEVIR_MCI
CA=__CHANGE_AGENT_ROOT__
cd "$ROOT"
need(){ test -s "$1" || { echo "ERROR missing $1"; exit 1; }; }
python runner/preflight.py --root "$ROOT"
python runner/smoke_test.py --config configs/partial_encoder_nomask.json
if [ ! -f runs/partial_encoder_nomask/.complete ]; then
  resume=(); test -s runs/partial_encoder_nomask/checkpoints/last.pt && resume=(--resume runs/partial_encoder_nomask/checkpoints/last.pt)
  python -u runner/train_encoder_nomask.py --config configs/partial_encoder_nomask.json --output runs/partial_encoder_nomask "${resume[@]}"
  need runs/partial_encoder_nomask/checkpoints/best.pt; touch runs/partial_encoder_nomask/.complete
fi
if [ ! -s caches/partial_clean/manifest.json ]; then
  python -u "$S4/runner/prepare_features.py" --variant partial_last2 --stage1-root "$ROOT" --checkpoint runs/partial_encoder_nomask/checkpoints/best.pt --dataset-root "$DATA" --list-root "$META" --output caches/partial_clean
fi
if [ ! -s runs/partial_teacher_nomask/checkpoints/partial_last2_best.pt ]; then
  python -u "$S4/runner/train_caption.py" --variant partial_last2 --cache caches/partial_clean --token-root "$META/tokens" --vocab "$META/vocab.json" --change-agent-root "$CA" --output runs/partial_teacher_nomask
fi
python "$S67/runner/preflight.py" --config configs/p3_nomask.json
python "$S67/runner/smoke_test.py" --config configs/p3_nomask.json
if [ ! -f runs/p3_nomask/.complete ]; then
  resume=(); test -s runs/p3_nomask/last.pt && resume=(--resume runs/p3_nomask/last.pt)
  python -u "$S67/runner/train.py" --config configs/p3_nomask.json --output runs/p3_nomask "${resume[@]}"
  need runs/p3_nomask/best.pt; touch runs/p3_nomask/.complete
fi
if [ ! -f results/p3_nomask/.complete ]; then
  python -u "$S67/runner/evaluate.py" --checkpoint runs/p3_nomask/best.pt --conditions all --output results/p3_nomask
  need results/p3_nomask/all_metrics.json; touch results/p3_nomask/.complete
fi
echo "MASK_FREE_PARTIAL_RELKD_COMPLETE"
