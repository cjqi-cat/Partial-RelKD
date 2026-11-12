# LEVIR-MCI Full-Family LOFO Summary

The table reports Change-Agent, Partial-RelKD, and the independently retrained
mask-free variant under the three complete held-out degradation families.
Values are B4/R-L/CIDEr. Mixed composition is reported separately in the
condition archives and is not included in the 6/9/9 family means.

| Family | Model | Clean B4/R-L/CIDEr | Seen B4/R-L/CIDEr | Unseen B4/R-L/CIDEr |
|---|---|---:|---:|---:|
| Atmospheric | Change-Agent | 0.6619/0.7645/1.4167 | 0.5853/0.6947/1.2264 | 0.6243/0.7277/1.3084 |
|  | Partial-RelKD (mask-free) | 0.6382/0.7466/1.3621 | 0.6329/0.7415/1.3488 | 0.6256/0.7336/1.3287 |
|  | Partial-RelKD | 0.6474/0.7620/1.4035 | 0.6425/0.7570/1.3902 | 0.6363/0.7512/1.3749 |
| Geometry-quality | Change-Agent | 0.6619/0.7645/1.4167 | 0.5959/0.7059/1.2500 | 0.5903/0.6942/1.2339 |
|  | Partial-RelKD (mask-free) | 0.6442/0.7493/1.3712 | 0.6415/0.7472/1.3649 | 0.6362/0.7406/1.3481 |
|  | Partial-RelKD | 0.6436/0.7638/1.4049 | 0.6424/0.7585/1.3904 | 0.6361/0.7536/1.3787 |
| Noise-loss | Change-Agent | 0.6619/0.7645/1.4167 | 0.6046/0.7095/1.2672 | 0.5728/0.6871/1.1995 |
|  | Partial-RelKD (mask-free) | 0.6499/0.7526/1.3744 | 0.6453/0.7488/1.3659 | 0.6371/0.7410/1.3460 |
|  | Partial-RelKD | 0.6607/0.7644/1.4077 | 0.6525/0.7587/1.3937 | 0.6439/0.7506/1.3720 |

Change-Agent is a fixed pretrained baseline and is regrouped under each LOFO
definition rather than retrained. Each Partial-RelKD lineage is retrained with
the degradation family excluded from training, validation, early stopping,
and checkpoint selection.
