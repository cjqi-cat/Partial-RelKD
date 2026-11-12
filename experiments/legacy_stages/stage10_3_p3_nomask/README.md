# Mask-free Partial-RelKD

This stage independently retrains the Partial-RelKD lineage without using
ground-truth change masks. The task, captions, vocabulary, calibrated-v3
degradations, geometry-quality LOFO split, and 31-condition evaluation remain
unchanged.

The visual encoder is selected only with mask-free validation objectives:
degradation-view consistency, raw-DINO feature distillation, synthetic
degradation adversarial learning, and backbone anchoring. Stage 2 retains
caption CE, token KD, clean-caption preservation, and relation KD.

```bash
cd __PROJECT_ROOT__/stage10_3_p3_nomask
python -m compileall -q runner
python runner/preflight.py --root "$PWD"
bash run_all.sh
```

The final model is written to `runs/p3_nomask/best.pt`; the 31-condition
metrics are written to `results/p3_nomask/all_metrics.json`.
