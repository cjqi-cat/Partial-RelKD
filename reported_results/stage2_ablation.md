# Incremental Stage-2 Ablation

LEVIR-MCI results under the geometry-quality LOFO protocol. Every row uses the
same Stage-1 change-sensitive initialization. Values are B4/R-L/CIDEr and
match Table III of the submitted manuscript.

| Model | Clean B4/R-L/CIDEr | Unseen B4/R-L/CIDEr |
|---|---:|---:|
| Frozen visual + token KD | 0.6411/0.7547/1.3773 | 0.6295/0.7458/1.3579 |
| + Caption-stage E2E visual update | 0.6439/0.7595/1.3851 | 0.6366/0.7516/1.3692 |
| + Clean-caption preservation | 0.6434/0.7623/1.4004 | 0.6345/0.7526/1.3743 |
| + Relation KD (Partial-RelKD) | 0.6436/0.7638/1.4049 | 0.6361/0.7536/1.3787 |
