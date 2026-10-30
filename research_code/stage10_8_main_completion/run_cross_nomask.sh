#!/usr/bin/env bash
set -Eeuo pipefail
DS=${1:?SECOND_CC|WHU_CDC}; case "$DS" in SECOND_CC|WHU_CDC);; *) exit 2;; esac
PROJECT=__PROJECT_ROOT__; ROOT=$PROJECT/stage10_8_main_completion; TAG=cross_${DS,,}
STAGE4=$PROJECT/stage4_caption_dino_finetune; STAGE67=$PROJECT/stage6_7_partial_hybrid
DATA=$PROJECT/stage8_1_multidataset_full/data_views/$DS
META=$PROJECT/stage8_1_multidataset_full/metadata/$DS
CFG=$ROOT/configs/$TAG; RUN=$ROOT/runs/$TAG; RES=$ROOT/results/cross/$DS/p3_nomask
mkdir -p "$RUN" "$RES"; need(){ test -s "$1" || { echo "missing $1"; exit 1; }; }

python -u "$ROOT/runner/train_encoder_nomask_generic.py" --config "$CFG/nomask_encoder.json" --output "$RUN/nomask_encoder"
need "$RUN/nomask_encoder/checkpoints/best.pt"
python -u "$STAGE4/runner/prepare_features.py" --variant partial_last2 \
  --stage1-root "$PROJECT/stage10_3_p3_nomask" --checkpoint "$RUN/nomask_encoder/checkpoints/best.pt" \
  --dataset-root "$DATA" --list-root "$META" --output "$RUN/cache"
python -u "$ROOT/runner/train_caption_fixed_refs.py" --variant partial_last2 --cache "$RUN/cache" \
  --token-root "$META/tokens" --vocab "$META/vocab.json" --change-agent-root __CHANGE_AGENT_ROOT__ \
  --output "$RUN/nomask_teacher"
python -u "$STAGE67/runner/train.py" --config "$CFG/p3_nomask.json" --output "$RUN/p3_nomask"
python -u "$STAGE67/runner/evaluate.py" --checkpoint "$RUN/p3_nomask/best.pt" --conditions all --output "$RES"
touch "$RES/.complete"; echo "CROSS_NOMASK_COMPLETE $DS"
