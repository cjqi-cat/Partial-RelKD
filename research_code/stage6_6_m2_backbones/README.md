# Stage 6.6: M2 across Change-Agent and DINOv2 backbones

This directory implements a strict geometry-quality LOFO comparison:

- CA-M1 / CA-M2;
- QKV-LoRA-DINO-M1 / M2;
- Partial-last2-DINO-M1 / M2.

The final task, vocabulary, captions and metrics are unchanged. M1 is online
degradation plus caption CE. M2 adds clean-teacher token-level KD.

## Strict LOFO rule

Training, validation, early stopping and checkpoint selection may use only:

`cloud, haze, gaussian_noise, jpeg, pixel_dropout, radiometric_shift`.

`blur, low_resolution, misregistration, mixed` are test-only. The DINO Stage-1
wrapper overrides the source trainer's hard-coded ten-corruption tuple and
checks the six allowed corruptions before training.

## Server destination

`__PROJECT_ROOT__/stage6_6_m2_backbones`

## Pipeline

### 1. CA-M1 and CA-M2

Verify that the original Change-Agent checkpoint exists at the path in
`configs/ca_m2_geometry.json`, then run preflight and smoke tests.

```bash
python runner/preflight.py --config configs/ca_m2_geometry.json
python runner/smoke_test.py --config configs/ca_m2_geometry.json
python runner/train_m2.py --config configs/ca_m2_geometry.json --variant m1 --output runs/ca_m1_geometry
python runner/train_m2.py --config configs/ca_m2_geometry.json --variant m2 --output runs/ca_m2_geometry
```

### 2. Strict-LOFO DINO encoders

```bash
python runner/run_dino_lofo.py \
  --source-train __PROJECT_ROOT__/dino_finetune_step1/runner/train.py \
  --config configs/dino_lora_encoder_geometry.json \
  --output runs/geometry_lora_encoder \
  --family geometry_quality

python runner/run_dino_lofo.py \
  --source-train __PROJECT_ROOT__/dino_finetune_step1/runner/train.py \
  --config configs/dino_partial_encoder_geometry.json \
  --output runs/geometry_partial_encoder \
  --family geometry_quality
```

### 3. Architecture-matched clean DINO caption teachers

Reuse the audited Stage-4 feature preparation and caption training scripts.

```bash
python __PROJECT_ROOT__/stage4_caption_dino_finetune/runner/prepare_features.py \
  --variant lora \
  --stage1-root __PROJECT_ROOT__/dino_finetune_step1 \
  --checkpoint runs/geometry_lora_encoder/checkpoints/best.pt \
  --dataset-root __DATA_ROOT__/LEVIR-MCI-dataset \
  --list-root __CHANGE_AGENT_ROOT__/data/LEVIR_MCI \
  --output caches/geometry_lora_clean

python __PROJECT_ROOT__/stage4_caption_dino_finetune/runner/train_caption.py \
  --variant lora \
  --cache caches/geometry_lora_clean \
  --token-root __CHANGE_AGENT_ROOT__/data/LEVIR_MCI/tokens \
  --vocab __CHANGE_AGENT_ROOT__/data/LEVIR_MCI/vocab.json \
  --change-agent-root __CHANGE_AGENT_ROOT__ \
  --output runs/geometry_lora_teacher
```

Repeat these two commands with `partial_last2`, the partial encoder checkpoint,
`caches/geometry_partial_clean`, and `runs/geometry_partial_teacher`.

### 4. DINO-M1 and DINO-M2

Run preflight after both clean caption teachers exist.

```bash
python runner/preflight.py \
  --config configs/dino_lora_m2_geometry.json \
  --config configs/dino_partial_m2_geometry.json

python runner/smoke_test.py --config configs/dino_lora_m2_geometry.json
python runner/smoke_test.py --config configs/dino_partial_m2_geometry.json

python runner/train_m2.py --config configs/dino_lora_m2_geometry.json --variant m1 --output runs/dino_lora_m1_geometry
python runner/train_m2.py --config configs/dino_lora_m2_geometry.json --variant m2 --output runs/dino_lora_m2_geometry
python runner/train_m2.py --config configs/dino_partial_m2_geometry.json --variant m1 --output runs/dino_partial_m1_geometry
python runner/train_m2.py --config configs/dino_partial_m2_geometry.json --variant m2 --output runs/dino_partial_m2_geometry
```

The first DINO experiment freezes the Stage-1 encoder during caption M2 training.
This protects the six-loss robust representation. End-to-end low-LR variants
should be attempted only after the frozen versions establish a stable baseline.

### 5. Evaluation

```bash
python runner/evaluate.py --checkpoint runs/ca_m2_geometry/best.pt --conditions all --output results/ca_m2_geometry
python runner/evaluate.py --checkpoint runs/dino_lora_m2_geometry/best.pt --conditions all --output results/dino_lora_m2_geometry
python runner/evaluate.py --checkpoint runs/dino_partial_m2_geometry/best.pt --conditions all --output results/dino_partial_m2_geometry
```

Use the corresponding M1 checkpoints for the required augmentation-only ablations.
