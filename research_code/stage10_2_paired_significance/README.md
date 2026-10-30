# Paired clustered Bootstrap

This stage performs the paper's paired significance test between
**Partial-RelKD (P3)** and **DINO-Aug** on the nine held-out geometry-quality
conditions. The same original test image is treated as one cluster across all
nine conditions and is resampled synchronously.

The script requires aligned per-sample prediction JSON files from both models.
It reports 95% confidence intervals and two-sided Monte Carlo p-values for
BLEU-4, ROUGE-L, and CIDEr, followed by Holm correction across the three
metrics. This fixed-checkpoint test complements, but does not replace, the
independent training-seed experiment.

```bash
bash run.sh
```
