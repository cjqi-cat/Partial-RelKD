# Three-dataset authoritative baseline reproduction

This stage applies the same target-supervised geometry-quality LOFO protocol
to Chg2Cap, MCCFormer-D, RSICCFormer, and SEN on SECOND-CC and WHU-CDC. The
corresponding LEVIR-MCI runs are produced by `stage10_authoritative_baselines`.
Change-Agent is trained by `stage8_1_multidataset_full`.

Third-party repositories are intentionally not vendored. Clone the commits
listed in the repository-level `THIRD_PARTY.md` into the paths expected by
`stage10_authoritative_baselines`, then run:

```bash
python runner/make_configs.py
python runner/preflight.py
bash run_all.sh
```
