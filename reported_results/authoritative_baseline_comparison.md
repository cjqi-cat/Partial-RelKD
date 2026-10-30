# Authoritative RSCC Baseline Comparison

LEVIR-MCI results under the primary geometry-quality LOFO protocol. Seen is
the macro-average over the allowed training degradation conditions; Unseen is
the macro-average over blur, low resolution, and misregistration applied to
temporal image A, temporal image B, or both; Mixed is reported separately.

The five external baselines are Change-Agent, Chg2Cap, MCCFormer-D,
RSICCFormer, and SEN. DINO-Aug is an architecture-matched control rather than
an external method. All models use identical splits, vocabulary, metric code,
and degradation exposure. Best and second-best values are marked independently
for every condition and metric.

| Model | Clean B4 | Clean R-L | Clean CIDEr | Seen B4 | Seen R-L | Seen CIDEr | Unseen B4 | Unseen R-L | Unseen CIDEr | Mixed B4 | Mixed R-L | Mixed CIDEr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Change-Agent | **0.6619** | **0.7645** | **1.4167** | 0.5959 | 0.7060 | 1.2500 | 0.5903 | 0.6942 | 1.2339 | 0.5824 | 0.6791 | 1.1893 |
| Chg2Cap | 0.5565 | 0.6918 | 1.2212 | 0.5367 | 0.6702 | 1.1632 | 0.4987 | 0.6189 | 1.0496 | 0.5010 | 0.6166 | 1.0412 |
| MCCFormer-D | 0.6082 | 0.7372 | 1.3271 | 0.6024 | 0.7312 | 1.3088 | 0.5856 | 0.7187 | 1.2799 | 0.5867 | 0.7172 | 1.2715 |
| RSICCFormer | 0.6272 | 0.7513 | 1.3691 | 0.6156 | 0.7424 | 1.3404 | 0.5814 | 0.7102 | 1.2636 | 0.6081 | 0.7297 | 1.3129 |
| SEN | 0.6198 | 0.7152 | 1.2706 | 0.6080 | 0.7029 | 1.2392 | 0.5554 | 0.6518 | 1.1199 | 0.5629 | 0.6574 | 1.1337 |
| DINO-Aug | 0.6388 | 0.7600 | 1.3983 | <u>0.6313</u> | <u>0.7533</u> | <u>1.3802</u> | <u>0.6199</u> | <u>0.7455</u> | <u>1.3602</u> | <u>0.6148</u> | <u>0.7331</u> | <u>1.3225</u> |
| **Partial-RelKD** | <u>0.6436</u> | <u>0.7638</u> | <u>1.4049</u> | **0.6424** | **0.7585** | **1.3904** | **0.6361** | **0.7536** | **1.3787** | **0.6256** | **0.7377** | **1.3352** |

Partial-RelKD leads all nine degraded Seen/Unseen/Mixed condition--metric
cells. Change-Agent retains the highest Clean scores, while Partial-RelKD is
second on all three Clean metrics.
