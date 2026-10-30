#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=__PROJECT_ROOT__/stage6_7_partial_hybrid
STAGE1=__PROJECT_ROOT__/dino_finetune_step1
STAGE4=__PROJECT_ROOT__/stage4_caption_dino_finetune
DATA=__DATA_ROOT__/LEVIR-MCI-dataset
META=__CHANGE_AGENT_ROOT__/data/LEVIR_MCI
CA=__CHANGE_AGENT_ROOT__

cd "$ROOT"

step() { echo; echo "===== $(date '+%F %T') START: $1 ====="; }
need() { test -s "$1" || { echo "ERROR missing: $1"; exit 1; }; }
trap 'echo "===== FAILED line=$LINENO time=$(date "+%F %T") ====="' ERR

train_stage2() {
    local config=$1
    local output=$2
    if [ -f "$output/.complete" ]; then
        echo "SKIP completed $output"
        need "$output/best.pt"
        return
    fi
    local resume=()
    if [ -s "$output/last.pt" ]; then
        resume=(--resume "$output/last.pt")
    fi
    python -u runner/train.py --config "$config" --output "$output" "${resume[@]}"
    need "$output/best.pt"
    touch "$output/.complete"
}

evaluate_stage2() {
    local checkpoint=$1
    local output=$2
    if [ -f "$output/.complete" ]; then
        echo "SKIP completed $output"
        need "$output/all_metrics.json"
        return
    fi
    python -u runner/evaluate.py --checkpoint "$checkpoint" --conditions all --output "$output"
    need "$output/all_metrics.json"
    touch "$output/.complete"
}

step "preflight P1-P3"
python runner/preflight.py \
    --config configs/p1_partial_e2e.json \
    --config configs/p2_partial_clean.json \
    --config configs/p3_partial_relation.json

step "smoke P3 full loss path"
python runner/smoke_test.py --config configs/p3_partial_relation.json

step "P1 Partial end-to-end"
train_stage2 configs/p1_partial_e2e.json runs/p1_partial_e2e
evaluate_stage2 runs/p1_partial_e2e/best.pt results/p1_partial_e2e

step "P2 Partial plus clean supervision"
train_stage2 configs/p2_partial_clean.json runs/p2_partial_clean
evaluate_stage2 runs/p2_partial_clean/best.pt results/p2_partial_clean

step "P3 Partial plus clean supervision and relation KD"
train_stage2 configs/p3_partial_relation.json runs/p3_partial_relation
evaluate_stage2 runs/p3_partial_relation/best.pt results/p3_partial_relation

echo "===== ALL P1-P3 COMPLETED: $(date '+%F %T') ====="
touch STAGE67_COMPLETED
