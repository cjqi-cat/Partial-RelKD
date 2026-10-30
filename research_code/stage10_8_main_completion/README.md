# Main-protocol completion experiments

This stage adds the controlled evidence reported around the main LEVIR-MCI
experiment without changing the caption task or metrics:

1. strict atmospheric and noise-loss family holdouts for DINO-Aug,
   Partial-RelKD, and Partial-RelKD (mask-free);
2. two additional mask-free training seeds;
3. two additional deterministic test-degradation seeds;
4. independently trained mask-free reproductions on SECOND-CC and WHU-CDC;
5. student-only inference efficiency and qualitative exports.

For every LOFO run, all members of the held-out family and `mixed` are removed
from training, validation, early stopping, and checkpoint selection. The
mask-free lineage does not read masks or semantic labels.

```bash
cd __PROJECT_ROOT__/stage10_8_main_completion
python runner/make_configs.py
python runner/preflight.py
bash run_all.sh
```

Paired clustered significance testing is maintained separately in
`stage10_2_paired_significance` because it requires aligned per-sample
predictions from Partial-RelKD and DINO-Aug.
