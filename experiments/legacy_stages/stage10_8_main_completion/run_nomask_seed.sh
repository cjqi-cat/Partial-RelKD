#!/usr/bin/env bash
set -Eeuo pipefail
SEED=${1:?20261018|20261028}; TAG=geometry_seed$SEED
ROOT=__PROJECT_ROOT__/stage10_8_main_completion; PROJECT=__PROJECT_ROOT__
STAGE4=$PROJECT/stage4_caption_dino_finetune; STAGE67=$PROJECT/stage6_7_partial_hybrid; CA=$PROJECT/stage3_caption_retrain/vendor/Change-Agent/Multi_change
DATA=__DATA_ROOT__/LEVIR-MCI-dataset; META=$CA/data/LEVIR_MCI; CFG=$ROOT/configs/$TAG; RUN=$ROOT/runs/$TAG; RES=$ROOT/results/multiseed/$TAG
mkdir -p "$RUN" "$RES"; need(){ test -s "$1" || exit 1; }; mark(){ touch "$1/.complete"; }
if [ ! -f "$RUN/nomask_encoder/.complete" ]; then resume=(); test -s "$RUN/nomask_encoder/checkpoints/last.pt" && resume=(--resume "$RUN/nomask_encoder/checkpoints/last.pt"); python -u "$ROOT/runner/train_encoder_nomask_generic.py" --config "$CFG/nomask_encoder.json" --output "$RUN/nomask_encoder" "${resume[@]}"; need "$RUN/nomask_encoder/checkpoints/best.pt"; mark "$RUN/nomask_encoder"; rm -f "$RUN/nomask_encoder/checkpoints/last.pt"; fi
if [ ! -s "$RUN/nomask_teacher/checkpoints/partial_last2_best.pt" ]; then python -u "$STAGE4/runner/prepare_features.py" --variant partial_last2 --stage1-root "$PROJECT/stage10_3_p3_nomask" --checkpoint "$RUN/nomask_encoder/checkpoints/best.pt" --dataset-root "$DATA" --list-root "$META" --output "$RUN/cache"; python -u "$ROOT/runner/train_caption_fixed_refs.py" --variant partial_last2 --cache "$RUN/cache" --token-root "$META/tokens" --vocab "$META/vocab.json" --change-agent-root "$CA" --output "$RUN/nomask_teacher" --seed "$SEED"; need "$RUN/nomask_teacher/checkpoints/partial_last2_best.pt"; rm -rf "$RUN/cache"; fi
if [ ! -f "$RUN/p3_nomask/.complete" ]; then resume=(); test -s "$RUN/p3_nomask/last.pt" && resume=(--resume "$RUN/p3_nomask/last.pt"); python -u "$STAGE67/runner/train.py" --config "$CFG/p3_nomask.json" --output "$RUN/p3_nomask" "${resume[@]}"; need "$RUN/p3_nomask/best.pt"; mark "$RUN/p3_nomask"; rm -f "$RUN/p3_nomask/last.pt"; fi
if [ ! -f "$RES/.complete" ]; then python -u "$STAGE67/runner/evaluate.py" --checkpoint "$RUN/p3_nomask/best.pt" --conditions all --output "$RES"; need "$RES/all_metrics.json"; touch "$RES/.complete"; fi
echo "NOMASK_SEED_COMPLETE $SEED"
