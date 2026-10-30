#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("BLEU-4", "ROUGE-L", "CIDEr")
SEEN = {"cloud", "haze", "gaussian_noise", "jpeg", "pixel_dropout", "radiometric_shift"}
UNSEEN = {"blur", "low_resolution", "misregistration"}


def means(payload, types):
    rows = [v for k, v in payload.items() if k != "clean" and k.split("__")[0] in types]
    return {m: sum(float(x[m]) for x in rows) / len(rows) for m in METRICS}


def aggregate(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "clean": {m: float(payload["clean"][m]) for m in METRICS},
        "seen": means(payload, SEEN),
        "unseen": means(payload, UNSEEN),
    }


def row(label, values):
    flat = [values[g][m] for g in ("clean", "seen", "unseen") for m in METRICS]
    return "| " + label + " | " + " | ".join(f"{x:.6f}" for x in flat) + " |"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    result_root = root / "results" / args.dataset
    models = {
        "DINO-Clean": aggregate(result_root / "dino_clean/all_metrics.json"),
        "DINO-Aug": aggregate(result_root / "dino_aug/all_metrics.json"),
    }
    # SECOND/WHU already have the complete Stage8.1 table.  Link those
    # values into this control report without copying predictions/checkpoints.
    old_root = Path("__PROJECT_ROOT__/stage8_1_multidataset_full/results") / args.dataset
    if old_root.is_dir():
        for label, folder in (
            ("Change-Agent-Target", "change_agent"), ("M2", "m2"),
            ("P3", "p3"),
        ):
            path = old_root / folder / "all_metrics.json"
            if path.is_file():
                models[label] = aggregate(path)
    header = [
        f"# {args.dataset} DINO同架构控制实验", "",
        "> DINO-Clean与DINO-Aug使用和该数据集M2相同的encoder/decoder初始化。DINO-Clean直接评测clean teacher；DINO-Aug仅使用允许退化上的caption CE，不使用token KD、relation KD或额外clean约束。", "",
        "| Model | Clean B4 | Clean R-L | Clean CIDEr | Seen B4 | Seen R-L | Seen CIDEr | Unseen B4 | Unseen R-L | Unseen CIDEr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ("Change-Agent-Target", "DINO-Clean", "DINO-Aug", "M2", "P3")
    lines = header + [row(label, models[label]) for label in order if label in models]
    clean, aug = models["DINO-Clean"], models["DINO-Aug"]
    lines += ["", "## DINO-Aug相对DINO-Clean的绝对差", "",
              "| Condition | ΔB4 | ΔR-L | ΔCIDEr |", "|---|---:|---:|---:|"]
    for group in ("clean", "seen", "unseen"):
        delta = [aug[group][m] - clean[group][m] for m in METRICS]
        lines.append(f"| {group} | {delta[0]:+.6f} | {delta[1]:+.6f} | {delta[2]:+.6f} |")
    output = result_root / "summary.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (result_root / "summary.json").write_text(json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
