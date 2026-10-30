# Stage 6.5: calibrated-v3 five-model retraining and six-model comparison

This stage repeats the original non-LOFO five-model protocol and changes only
the degradation profile from the old generator to the audited calibrated-v3
profile.

## Fixed protocol

- Dataset: original LEVIR-MCI clean train/val/test images.
- Encoder train subset: 4,000 train pairs and 800 validation pairs.
- Caption training: full clean train (6,815) and validation (1,333) splits.
- Test: clean plus 30 calibrated-v3 conditions over all 1,929 test pairs.
- Training seed: 20261001.
- Test degradation seed: 20260928.
- No degraded PNGs are written to disk.
- LOFO is out of scope.

## Models

- Raw: frozen raw DINOv2, clean caption training.
- Control: calibrated-v3 inputs with change/patch/distillation objectives, but
  `lambda_invariance=0` and `lambda_adversarial=0`.
- Full: calibrated-v3 inputs with the complete adapter objective.
- LoRA: calibrated-v3 inputs with 12-block QKV LoRA.
- Partial-last2: calibrated-v3 inputs with the last two DINO blocks unfrozen.
- Original Change-Agent: locked Stage 6.4 calibrated-v3 baseline metrics.

## Server location

`__PROJECT_ROOT__/stage6_5_calibrated_v3`

## Run

```bash
cd __PROJECT_ROOT__/stage6_5_calibrated_v3
chmod +x *.sh
bash start_server.sh __WORK_ROOT__
bash status_server.sh __WORK_ROOT__
```

Open `status.html` from the JupyterLab file browser for a lightweight progress
page. The pipeline skips completed checkpoints and condition metrics. LoRA and
Partial-last2 resume from `last.pt`. Large feature caches created inside this
stage are deleted only after their dependent checkpoint has been written.
