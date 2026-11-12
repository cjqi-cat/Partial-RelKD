# Change-Agent baseline interface

Change-Agent is treated as a peer external baseline in all reported comparison
tables. Its upstream source and checkpoints are not redistributed; obtain the
pinned revision documented in `THIRD_PARTY.md` and provide its `Multi_change`
directory through `--change-agent-root` during workspace setup.

The shared Change-Agent decoder, vocabulary interface, and metric code are used
throughout the repository. Dataset-specific supervised Change-Agent training
and 31-condition evaluation for SECOND-CC and WHU-CDC are implemented in
`experiments/legacy_stages/stage8_1_multidataset_full`. LEVIR-MCI integration
and controlled Change-Agent variants are implemented in
`experiments/legacy_stages/stage6_6_m2_backbones`.
