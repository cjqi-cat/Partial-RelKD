# Three-Dataset Unseen-Degradation Comparison

Paper-aligned results under the geometry-quality LOFO protocol. The seven
external baselines are Change-Agent, Chg2Cap, MCCFormer-D, RSICCFormer, SEN,
SCNet, and RMNet. Every valid model is independently trained using each
dataset's own training images, captions, vocabulary, and split. Best and
second-best values are marked per dataset and metric.

RMNet--SECOND-CC is not reported: the controlled extension collapsed to one
generic caption, so its corpus scores did not measure image-conditioned
generation and were excluded rather than presented as valid low scores.

| Dataset | Model | Unseen B4 | Unseen R-L | Unseen CIDEr |
|---|---|---:|---:|---:|
| LEVIR-MCI | Change-Agent | 0.5903 | 0.6942 | 1.2339 |
|  | Chg2Cap | 0.4987 | 0.6189 | 1.0496 |
|  | MCCFormer-D | 0.5856 | 0.7187 | 1.2799 |
|  | RSICCFormer | 0.5814 | 0.7102 | 1.2636 |
|  | SEN | 0.5554 | 0.6518 | 1.1199 |
|  | SCNet | 0.5287 | 0.6344 | 1.0731 |
|  | RMNet | 0.5802 | 0.6668 | 1.0905 |
|  | Partial-RelKD (mask-free) | **0.6362** | <u>0.7406</u> | <u>1.3481</u> |
|  | Partial-RelKD | <u>0.6361</u> | **0.7536** | **1.3787** |
| SECOND-CC | Change-Agent | 0.2770 | 0.4878 | 0.6336 |
|  | Chg2Cap | 0.2566 | 0.4513 | 0.5660 |
|  | MCCFormer-D | 0.2987 | 0.5019 | 0.6854 |
|  | RSICCFormer | 0.3082 | 0.5125 | 0.7237 |
|  | SEN | 0.2786 | 0.4792 | 0.6320 |
|  | SCNet | 0.2667 | 0.4816 | 0.6265 |
|  | RMNet | -- | -- | -- |
|  | Partial-RelKD (mask-free) | <u>0.3124</u> | <u>0.5190</u> | <u>0.7334</u> |
|  | Partial-RelKD | **0.3294** | **0.5360** | **0.7805** |
| WHU-CDC | Change-Agent | 0.6893 | 0.7749 | 1.4323 |
|  | Chg2Cap | 0.6686 | 0.7362 | 1.3183 |
|  | MCCFormer-D | 0.6523 | 0.7600 | 1.4083 |
|  | RSICCFormer | 0.6922 | 0.7909 | 1.4925 |
|  | SEN | 0.6915 | 0.7671 | 1.4160 |
|  | SCNet | 0.6616 | 0.7535 | 1.3715 |
|  | RMNet | 0.7286 | 0.7814 | 1.4417 |
|  | Partial-RelKD (mask-free) | **0.7536** | <u>0.8236</u> | **1.5962** |
|  | Partial-RelKD | <u>0.7478</u> | **0.8273** | <u>1.5955</u> |

SCNet and RMNet use their public architectures inside the common data,
degradation, checkpoint-selection, and metric shell; these are controlled
retraining results rather than copied paper numbers.
