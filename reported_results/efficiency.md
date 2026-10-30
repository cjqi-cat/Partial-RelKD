# 推理效率（单卡，batch=1，greedy）

| Model | Params(M) | Trainable(M) | Latency(ms) | Peak memory(MB) |
|---|---:|---:|---:|---:|
| Change-Agent | 130.50 | 44.15 | 47.50 | 530.6 |
| Partial-RelKD | 106.33 | 34.79 | 34.32 | 492.0 |
| Partial-RelKD (mask-free) | 106.33 | 34.79 | 33.87 | 493.0 |
