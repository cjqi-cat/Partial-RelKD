# Paired Clustered Bootstrap

The paired test compares Partial-RelKD with the architecture-matched DINO-Aug
control on the nine held-out geometry-quality conditions. Original test-image
IDs are the resampling clusters, and all nine degraded views of a sampled
image are resampled together. The test uses 10,000 repetitions and Holm
correction across BLEU-4, ROUGE-L, and CIDEr.

For unseen CIDEr, the paper reports:

- difference: `+0.0185`;
- 95% confidence interval: `[+0.0082, +0.0289]`;
- Holm-adjusted p-value: `0.0012`.

The significance claim is restricted to Partial-RelKD versus DINO-Aug.
