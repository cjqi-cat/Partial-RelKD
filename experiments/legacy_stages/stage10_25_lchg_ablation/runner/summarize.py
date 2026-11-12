#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("BLEU-4", "ROUGE-L", "CIDEr")
UNSEEN = tuple(
    f"{kind}__{side}"
    for kind in ("blur", "low_resolution", "misregistration")
    for side in ("A", "B", "both")
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(rows, conditions):
    return {metric: sum(float(rows[c][metric]) for c in conditions) / len(conditions) for metric in METRICS}


def group(rows, condition):
    if condition == "unseen_mean":
        return aggregate(rows, UNSEEN)
    return {metric: float(rows[condition][metric]) for metric in METRICS}


def fmt(value):
    return f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    new_path = Path(args.new)
    baseline_path = Path(args.baseline) if args.baseline else None
    new = load(new_path)
    baseline = load(baseline_path) if baseline_path and baseline_path.exists() else None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    groups = ("clean", "unseen_mean", "mixed__both")
    payload = {
        "ablation": "disable L_chg only; retain L_patch and every other Stage-1/Stage-2 setting",
        "new_metrics": {name: group(new, name) for name in groups},
        "baseline_available": baseline is not None,
    }
    if baseline is not None:
        payload["baseline_metrics"] = {name: group(baseline, name) for name in groups}
        payload["delta_wo_lchg_minus_p3"] = {
            name: {
                metric: payload["new_metrics"][name][metric] - payload["baseline_metrics"][name][metric]
                for metric in METRICS
            }
            for name in groups
        }

    lines = [
        "# Independent L_chg ablation (exploratory)",
        "",
        "> `P3 w/o L_chg` changes only Stage-1 `lambda_change: 1.0 -> 0.0`. "
        "`L_patch` and all other Stage-1 and Stage-2 settings remain enabled.",
        "",
        "| Model | Group | B4 | R-L | CIDEr |",
        "|---|---|---:|---:|---:|",
    ]
    if baseline is not None:
        for name in groups:
            row = payload["baseline_metrics"][name]
            lines.append(f"| P3 | {name} | {fmt(row['BLEU-4'])} | {fmt(row['ROUGE-L'])} | {fmt(row['CIDEr'])} |")
    for name in groups:
        row = payload["new_metrics"][name]
        lines.append(f"| P3 w/o L_chg | {name} | {fmt(row['BLEU-4'])} | {fmt(row['ROUGE-L'])} | {fmt(row['CIDEr'])} |")

    if baseline is not None:
        lines += [
            "",
            "## Delta: P3 w/o L_chg - P3",
            "",
            "| Group | Delta B4 | Delta R-L | Delta CIDEr |",
            "|---|---:|---:|---:|",
        ]
        for name in groups:
            row = payload["delta_wo_lchg_minus_p3"][name]
            lines.append(f"| {name} | {row['BLEU-4']:+.6f} | {row['ROUGE-L']:+.6f} | {row['CIDEr']:+.6f} |")
        lines += [
            "",
            "> A negative delta means removing L_chg hurts performance, supporting its usefulness. "
            "This is a single-seed exploratory ablation, not a significance claim.",
        ]
    else:
        lines += ["", "> Baseline `all_metrics.json` was not found; only the ablated result is reported."]

    lines += [
        "",
        "## Per-condition unseen results",
        "",
        "| Condition | B4 | R-L | CIDEr |",
        "|---|---:|---:|---:|",
    ]
    for condition in UNSEEN:
        row = new[condition]
        lines.append(f"| {condition} | {fmt(row['BLEU-4'])} | {fmt(row['ROUGE-L'])} | {fmt(row['CIDEr'])} |")

    (output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("SUMMARY_OK")


if __name__ == "__main__":
    main()
