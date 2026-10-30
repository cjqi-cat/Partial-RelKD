# Reported Results

This directory contains the compact, paper-aligned results retained in the
public release. All caption scores use the same BLEU-4 (B4), ROUGE-L (R-L),
and CIDEr implementation.

`PAPER_ALIGNMENT.md` records the final submitted-manuscript consistency audit.

## Primary tables

- [`authoritative_baseline_comparison.md`](authoritative_baseline_comparison.md):
  complete LEVIR-MCI comparison against five authoritative RSCC baselines,
  plus the architecture-matched DINO-Aug control.
- [`three_dataset_full_results.md`](three_dataset_full_results.md): complete
  Clean/Seen/Unseen/Mixed results on LEVIR-MCI, SECOND-CC, and WHU-CDC.

## Supporting results

- `stage2_ablation.md`: paper Table III incremental caption-stage ablation.
- `statistical_significance.md`: paper-aligned paired clustered-Bootstrap result.
- `main_completion_summary.md`: full-family LOFO and seed-stability summary.
- `severity_summary.md` / `severity_summary.json`: severity 1--5 curves and
  machine-readable values.
- `severity_audit.json`: deterministic severity-generation audit.
- `efficiency.md`: parameter, latency, and memory measurements.

SECOND-CC and WHU-CDC results are obtained after independent supervised
training with each dataset's own images, captions, vocabulary, and splits,
demonstrating method-level reproducibility across datasets.
