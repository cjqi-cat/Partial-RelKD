# Reproducing Partial-RelKD

This guide reproduces the main LEVIR-MCI geometry-quality LOFO experiment from
visual pre-adaptation through 31-condition caption evaluation. Commands assume
a Linux host with a CUDA-capable GPU and Bash.

## 1. Prerequisites

Prepare the following resources before starting:

- the LEVIR-MCI images and masks described in [`DATASETS.md`](../DATASETS.md);
- the pinned Change-Agent `Multi_change` checkout described in
  [`THIRD_PARTY.md`](../THIRD_PARTY.md), including `data/LEVIR_MCI`, the caption
  decoder, and the BLEU/ROUGE/CIDEr implementation;
- a Python environment compatible with the versions in `requirements.txt`.

Install the public package and its main dependencies:

```bash
REPO=/path/to/Partial-RelKD
python -m pip install -r "$REPO/requirements.txt"
python -m pip install -e "$REPO" --no-deps
```

## 2. Create the runnable workspace

Choose an empty experiment directory and provide the local data and
Change-Agent paths:

```bash
PROJECT=/path/to/robust_cc/project
DATA_ROOT=/path/to/robust_cc/data
CHANGE_AGENT_ROOT=/path/to/Change-Agent/Multi_change

python "$REPO/tools/materialize_workspace.py" \
  --output "$PROJECT" \
  --data-root "$DATA_ROOT" \
  --change-agent-root "$CHANGE_AGENT_ROOT"
```

The command resolves local path placeholders and creates the stage-oriented
workspace used by the validated experiment drivers. It refuses to overwrite a
non-empty output directory.

## 3. Stage 1: change-sensitive visual pre-adaptation

Train the Partial-last2 DINOv2 encoder using only the six degradations allowed
by the geometry-quality LOFO protocol:

```bash
python "$REPO/scripts/train_stage1.py" \
  --workspace "$PROJECT" \
  -- \
  --source-train "$PROJECT/dino_finetune_step1/runner/train.py" \
  --config "$PROJECT/configs/stage1/geometry_quality.json" \
  --output "$PROJECT/stage6_6_m2_backbones/runs/geometry_partial_encoder" \
  --family geometry_quality
```

Expected checkpoint:

```text
$PROJECT/stage6_6_m2_backbones/runs/geometry_partial_encoder/checkpoints/best.pt
```

`blur`, `low_resolution`, `misregistration`, and `mixed` are excluded from
training, validation, early stopping, and checkpoint selection.

## 4. Train the clean caption teacher

First cache clean visual features from the Stage-1 encoder:

```bash
python "$REPO/scripts/prepare_teacher_features.py" \
  --workspace "$PROJECT" \
  -- \
  --variant partial_last2 \
  --stage1-root "$PROJECT/dino_finetune_step1" \
  --checkpoint "$PROJECT/stage6_6_m2_backbones/runs/geometry_partial_encoder/checkpoints/best.pt" \
  --dataset-root "$DATA_ROOT/LEVIR-MCI-dataset" \
  --list-root "$CHANGE_AGENT_ROOT/data/LEVIR_MCI" \
  --output "$PROJECT/stage6_6_m2_backbones/caches/geometry_partial_clean"
```

Then train the clean caption model:

```bash
python "$REPO/scripts/train_teacher.py" \
  --workspace "$PROJECT" \
  -- \
  --variant partial_last2 \
  --cache "$PROJECT/stage6_6_m2_backbones/caches/geometry_partial_clean" \
  --token-root "$CHANGE_AGENT_ROOT/data/LEVIR_MCI/tokens" \
  --vocab "$CHANGE_AGENT_ROOT/data/LEVIR_MCI/vocab.json" \
  --change-agent-root "$CHANGE_AGENT_ROOT" \
  --output "$PROJECT/stage6_6_m2_backbones/runs/geometry_partial_teacher"
```

Expected checkpoint:

```text
$PROJECT/stage6_6_m2_backbones/runs/geometry_partial_teacher/checkpoints/partial_last2_best.pt
```

## 5. Stage 2: Partial-RelKD caption training

Train the degraded student with dirty-caption CE, token KD, clean-caption
preservation, relation KD, and Stage-1 anchoring:

```bash
python "$REPO/scripts/train_partial_relkd.py" \
  --workspace "$PROJECT" \
  -- \
  --config "$PROJECT/configs/stage2/partial_relkd.json" \
  --output "$PROJECT/stage6_7_partial_hybrid/runs/p3_partial_relation"
```

Expected checkpoint:

```text
$PROJECT/stage6_7_partial_hybrid/runs/p3_partial_relation/best.pt
```

To resume an interrupted Stage-2 run, append:

```bash
--resume "$PROJECT/stage6_7_partial_hybrid/runs/p3_partial_relation/last.pt"
```

## 6. Evaluate all 31 conditions

```bash
python "$REPO/scripts/evaluate_lofo.py" \
  --workspace "$PROJECT" \
  -- \
  --checkpoint "$PROJECT/stage6_7_partial_hybrid/runs/p3_partial_relation/best.pt" \
  --conditions all \
  --output "$PROJECT/stage6_7_partial_hybrid/results/p3_partial_relation"
```

The output contains one metrics file and one prediction file per condition,
plus the combined summary:

```text
$PROJECT/stage6_7_partial_hybrid/results/p3_partial_relation/all_metrics.json
```

The evaluation uses the severity and degradation seed stored in the training
checkpoint configuration. The public main configuration uses severity 2 and
degradation seed `20260928`.

## 7. Mask-free extension

The independently retrained mask-free lineage has a self-contained validated
pipeline after workspace setup:

```bash
cd "$PROJECT/stage10_3_p3_nomask"
python runner/preflight.py --root "$PWD"
bash run_all.sh
```

Its final checkpoint and 31-condition summary are written to:

```text
$PROJECT/stage10_3_p3_nomask/runs/p3_nomask/best.pt
$PROJECT/stage10_3_p3_nomask/results/p3_nomask/all_metrics.json
```

## 8. Expected result files

Canonical aggregate results are listed in
[`reported_results`](../reported_results/README.md). The compact public
31-condition metrics for the main and mask-free models are under
[`reported_results/per_condition`](../reported_results/per_condition/README.md).

The repository does not include datasets, pretrained weights, trained
checkpoints, feature caches, or per-sample generated captions.
