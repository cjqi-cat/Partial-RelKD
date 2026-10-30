# Stage 6.7: Partial-RelKD caption ablations

This stage contains the three caption-training ablations used in the paper.
It keeps the task, labels, vocabulary, test conditions, and metrics fixed.
For the geometry-quality LOFO protocol, blur, low resolution,
misregistration, and mixed degradation are excluded from training,
validation, early stopping, and checkpoint selection.

## Ablations

- **P1:** train the partial-last2 student with degraded-view caption CE and
  token distillation.
- **P2:** P1 plus clean-caption preservation through a clean forward pass of
  the same trainable student.
- **P3 / Partial-RelKD:** P2 plus relation-memory distillation from the frozen
  clean teacher.

The Stage-2 objective is

`L = L_dirty + 0.5 L_token + w_c L_clean + w_r L_relation + 0.02 L_anchor`.

P1 uses `w_c=w_r=0`, P2 uses `w_c=0.5`, and P3 additionally uses
`w_r=0.1`. The teacher always receives the corresponding clean pair and is
detached from gradient updates.

## Reused inputs

- Partial-last2 Stage-1 encoder and clean caption teacher from Stage 6.6.
- `calibrated_v3.py` online operators.
- LEVIR-MCI lists, tokens, vocabulary, and Change-Agent metric code.
- Test severity 2 and degradation seed 20260928.

## Run

```bash
cd __PROJECT_ROOT__/stage6_7_partial_hybrid
python -m compileall -q .
bash run_all.sh
```

`STAGE67_COMPLETED` is written after P1--P3 and their 31-condition evaluations
finish successfully. Training can resume from `last.pt`.
