# Incremental Stage-2 Ablation

LEVIR-MCI results under the geometry-quality LOFO protocol. The first four
rows use the same Stage-1 change-sensitive initialization. The final
`Full w/o L_chg` row independently retrains Stage 1 with only
`lambda_change: 1.0 -> 0.0`; `L_patch` and every other Stage-1/Stage-2 setting
remain enabled. Values are B4/R-L/CIDEr and match the current manuscript.

| Model | Clean B4/R-L/CIDEr | Unseen B4/R-L/CIDEr |
|---|---:|---:|
| Frozen visual + token KD | 0.6411/0.7547/1.3773 | 0.6295/0.7458/1.3579 |
| + Caption-stage E2E visual update | 0.6439/0.7595/1.3851 | 0.6366/0.7516/1.3692 |
| + Clean-caption preservation | 0.6434/0.7623/1.4004 | 0.6345/0.7526/1.3743 |
| Full w/o `L_chg` | 0.6479/0.7606/1.3976 | 0.6335/0.7484/1.3648 |
| + Relation KD (Partial-RelKD) | 0.6436/0.7638/1.4049 | 0.6361/0.7536/1.3787 |

The `L_chg` ablation is a single-run mechanism check rather than a statistical
significance claim. Its mixed-both score is 0.6195/0.7294/1.3149, compared
with 0.6300/0.7379/1.3382 for Partial-RelKD under the same evaluation.
