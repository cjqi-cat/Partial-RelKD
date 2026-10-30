# Submitted-Paper Result Alignment

Audit date: 2026-10-30.

The aggregate files in this directory were checked against the final v7
submitted-manuscript source without rerunning training or evaluation. Values
are compared at the precision displayed in the paper.

| Paper content | Canonical release source | Status |
|---|---|---|
| Table I: three unseen degradation families | `main_completion_summary.md` | Exact at displayed precision |
| Table II: LEVIR-MCI four-condition comparison | `authoritative_baseline_comparison.md`, `main_completion_summary.md`, and `severity_summary.md` | Exact at displayed precision |
| Table III: incremental Stage-2 ablation | `stage2_ablation.md` | Exact at displayed precision |
| Table IV: three datasets, four conditions, three metrics | `three_dataset_full_results.md` | Exact at displayed precision |
| Three-training-seed and three-degradation-seed statements | `main_completion_summary.md` | Exact at displayed precision |
| Paired clustered-Bootstrap statement | `statistical_significance.md` | Exact at displayed precision |
| Severity 1--5 curves | `severity_summary.md` and `severity_summary.json` | Exact at displayed precision |
| Parameter, latency, and memory statement | `efficiency.md` | Exact at displayed precision |

No numerical mismatch was found. This file records an alignment audit; it is
not a new experiment and does not replace the canonical machine-readable
evaluation archives retained by the authors.
