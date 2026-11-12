# External baseline wrappers

`wrappers/` contains the local integration, training, and evaluation interfaces
used to place seven external baselines inside the common LOFO protocol.
Third-party model source code and checkpoints are deliberately excluded.

| Baseline | Wrapper stage |
|---|---|
| Change-Agent | `change_agent` plus `stage8_1_multidataset_full` under `experiments/legacy_stages` |
| Chg2Cap, MCCFormer-D, RSICCFormer, SEN | `stage10_authoritative_baselines`, `stage10_9_three_dataset_baselines` |
| SCNet | `stage10_17_scnet_lofo`, `stage10_19_scnet_multidataset` |
| RMNet | `stage10_18_rmnet_lofo`, `stage10_20_rmnet_multidataset` |

Clone the pinned upstream revisions documented in `THIRD_PARTY.md` into the
vendor locations expected by the materialized configuration files.
