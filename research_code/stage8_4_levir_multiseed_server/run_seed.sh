#!/usr/bin/env bash
set -Eeuo pipefail
SEED=${1:?Usage: bash run_seed.sh SEED}
PROJECT=__PROJECT_ROOT__; ROOT=$PROJECT/stage8_4_levir_multiseed_server
TAG=seed_$SEED; CFG=$ROOT/configs/$TAG; RUN=$ROOT/runs/$TAG; RES=$ROOT/results/$TAG
DATA=__DATA_ROOT__/LEVIR-MCI-dataset; META=__CHANGE_AGENT_ROOT__/data/LEVIR_MCI
mkdir -p "$RUN" "$RES"; python "$ROOT/runner/make_configs.py" --seed "$SEED"

python "$PROJECT/stage6_6_m2_backbones/runner/run_dino_lofo.py" \
  --source-train "$PROJECT/dino_finetune_step1/runner/train.py" \
  --config "$CFG/partial_encoder.json" --output "$RUN/partial_encoder" --family geometry_quality
python "$PROJECT/stage4_caption_dino_finetune/runner/prepare_features.py" \
  --variant partial_last2 --stage1-root "$PROJECT/dino_finetune_step1" \
  --checkpoint "$RUN/partial_encoder/checkpoints/best.pt" --dataset-root "$DATA" \
  --list-root "$META" --output "$RUN/partial_cache" --seed "$SEED"
python "$PROJECT/stage4_caption_dino_finetune/runner/train_caption.py" \
  --variant partial_last2 --cache "$RUN/partial_cache" --token-root "$META/tokens" \
  --vocab "$META/vocab.json" --change-agent-root __CHANGE_AGENT_ROOT__ \
  --output "$RUN/partial_teacher" --seed "$SEED"
python "$PROJECT/stage6_6_m2_backbones/runner/train_m2.py" \
  --config "$CFG/dino_aug.json" --variant m1 --output "$RUN/dino_aug"
python "$PROJECT/stage6_6_m2_backbones/runner/evaluate.py" \
  --checkpoint "$RUN/dino_aug/best.pt" --conditions all --output "$RES/dino_aug"
python "$PROJECT/stage6_7_partial_hybrid/runner/train.py" \
  --config "$CFG/p3.json" --output "$RUN/p3"
python "$PROJECT/stage6_7_partial_hybrid/runner/evaluate.py" \
  --checkpoint "$RUN/p3/best.pt" --conditions all --output "$RES/p3"
echo "SEED_COMPLETE $SEED"
