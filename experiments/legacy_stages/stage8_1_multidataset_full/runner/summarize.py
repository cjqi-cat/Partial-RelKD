#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = (
    ("Change-Agent-Target", "change_agent"),
    ("M2", "m2"),
    ("P3", "p3"),
)
METRICS = ("BLEU-4", "ROUGE-L", "CIDEr")
SEEN = {"cloud", "haze", "gaussian_noise", "jpeg", "pixel_dropout", "radiometric_shift"}
UNSEEN = {"blur", "low_resolution", "misregistration"}


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    tables = {}
    for label, folder in MODELS:
        path = root / "results" / args.dataset / folder / "all_metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        seen = [v for k, v in metrics.items() if k != "clean" and k.split("__")[0] in SEEN]
        unseen = [v for k, v in metrics.items() if k != "clean" and k.split("__")[0] in UNSEEN]
        tables[label] = {
            "clean": {m: float(metrics["clean"][m]) for m in METRICS},
            "seen": {m: mean(seen, m) for m in METRICS},
            "unseen": {m: mean(unseen, m) for m in METRICS},
        }
    baseline = tables["Change-Agent-Target"]
    lines = [f"# {args.dataset} 完整监督跨基准复现实验", "",
             "> Each model uses the dataset's own training images, captions, vocabulary, and splits. M2 and P3 use only geometry-quality-LOFO-allowed online degradations; blur, low_resolution, misregistration, and mixed are excluded from training and model selection.", "",
             "| Model | Clean B4 | Clean R-L | Clean CIDEr | Seen B4 | Seen R-L | Seen CIDEr | Unseen B4 | Unseen R-L | Unseen CIDEr |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, _ in MODELS:
        x = tables[label]
        values = [x[group][metric] for group in ("clean", "seen", "unseen") for metric in METRICS]
        lines.append("| " + label + " | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", "## 相对Change-Agent-Target的绝对差", "",
              "| Model | Clean ΔB4 | Clean ΔR-L | Clean ΔCIDEr | Seen ΔB4 | Seen ΔR-L | Seen ΔCIDEr | Unseen ΔB4 | Unseen ΔR-L | Unseen ΔCIDEr |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, _ in MODELS[1:]:
        x = tables[label]
        values = [x[group][metric] - baseline[group][metric] for group in ("clean", "seen", "unseen") for metric in METRICS]
        lines.append("| " + label + " | " + " | ".join(f"{v:+.6f}" for v in values) + " |")
    output = root / "results" / args.dataset / "summary.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output.parent / "summary.json").write_text(json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
