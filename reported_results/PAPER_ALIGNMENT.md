# Current-Manuscript Result Alignment

Audit date: 2026-11-12. The release summaries were checked against the current
extended-baseline GRSL manuscript without rerunning training or evaluation.

| Manuscript content | Canonical release source | Status |
|---|---|---|
| Three-dataset unseen comparison: seven external baselines, mask-free model, and Partial-RelKD | `authoritative_baseline_comparison.md` | Exact at displayed precision |
| Incremental method ablation, including `Full w/o L_chg` | `stage2_ablation.md` | Exact at displayed precision |
| Three full-family LEVIR-MCI LOFO protocols | `main_completion_summary.md` | Exact at displayed precision |
| Severity 1--5 curves | `severity_summary.md` and `severity_summary.json` | Exact at displayed precision |
| Parameter, latency, and memory statement | `efficiency.md` | Exact at displayed precision |
| Complete three-dataset Clean/Seen/Unseen/Mixed archive | `three_dataset_full_results.md` | Repository extension; consistent with manuscript cells |

No numerical mismatch was found. RMNet--SECOND-CC is deliberately shown as
missing because the controlled extension failed the image-conditioned caption
audit; no collapsed corpus score is promoted as a valid comparison.
