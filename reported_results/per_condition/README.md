# Per-condition LEVIR-MCI results

This directory provides the canonical 31-condition metrics for the main
Partial-RelKD model and its independently retrained mask-free counterpart under
the geometry-quality LOFO protocol.

Each JSON file contains one clean condition and 30 degraded conditions formed
by nine individual degradation operators plus one mixed composition, each
applied to temporal image A, temporal image B, or both images. Degraded
conditions use severity 2 and degradation seed `20260928`.

- `partial_relkd_31_conditions.json`: Partial-RelKD.
- `partial_relkd_mask_free_31_conditions.json`: Partial-RelKD (mask-free).

Every condition records BLEU-1 through BLEU-4, ROUGE-L, CIDEr, sample count,
model variant, LOFO family, severity, and degradation seed. The files contain
aggregate condition-level metrics only; per-sample captions are intentionally
excluded.

For the summary tables, seen is the mean over `cloud`, `haze`,
`gaussian_noise`, `jpeg`, `pixel_dropout`, and `radiometric_shift`; unseen is
the mean over `blur`, `low_resolution`, and `misregistration`. Mixed is
reported separately.
