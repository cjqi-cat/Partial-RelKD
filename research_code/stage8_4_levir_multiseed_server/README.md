# LEVIR-MCI training-seed stability

This stage repeats **DINO-Aug** and **Partial-RelKD (P3)** with seeds
20261018 and 20261028. The primary run uses seed 20261008. Every run uses the
same split, strict geometry-quality LOFO exclusion, severity-2 test images, and
degradation seed 20260928. Held-out conditions are never used for checkpoint
selection.

Run `bash run_all.sh` after the primary-seed outputs are available. The summary
reports mean and sample standard deviation over the three independently
trained checkpoints.
