# Reported Results

This directory contains the compact, paper-aligned results retained in the
public release. All caption scores use the same BLEU-4 (B4), ROUGE-L (R-L),
and CIDEr implementation.

## Primary tables

- [`three_dataset_full_results.md`](three_dataset_full_results.md): complete
  Clean/Seen/Unseen/Mixed comparison against seven external RSCC baselines,
  Partial-RelKD (mask-free), and Partial-RelKD on LEVIR-MCI, SECOND-CC, and
  WHU-CDC.

## Supporting results

- `stage2_ablation.md`: paper-aligned incremental ablation, including the
  independently retrained model without Stage-1 change preservation.
- `main_completion_summary.md`: full-family LOFO summary for Change-Agent,
  Partial-RelKD, and its mask-free counterpart.
- `severity_summary.md` / `severity_summary.json`: severity 1--5 curves and
  machine-readable values.
- `severity_audit.json`: deterministic severity-generation audit.
- `efficiency.md`: parameter, latency, and memory measurements.
- `per_condition/`: canonical 31-condition metrics for Partial-RelKD and its
  independently retrained mask-free counterpart.

SECOND-CC and WHU-CDC results are obtained after independent supervised
training with each dataset's own images, captions, vocabulary, and splits,
demonstrating method-level reproducibility across datasets.
