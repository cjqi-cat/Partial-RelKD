#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

SEEN = ("cloud", "haze", "gaussian_noise", "jpeg", "pixel_dropout", "radiometric_shift")
UNSEEN = ("blur", "low_resolution", "misregistration")
METRICS = ("BLEU-4", "ROUGE-L", "CIDEr")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def grouped(rows):
    def conditions(kinds):
        return [f"{kind}__{side}" for kind in kinds for side in ("A", "B", "both")]

    result = {"clean": {key: float(rows["clean"][key]) for key in METRICS}}
    for group, kinds in (("seen", SEEN), ("unseen", UNSEEN)):
        names = conditions(kinds)
        result[group] = {
            key: sum(float(rows[name][key]) for name in names) / len(names) for key in METRICS
        }
    names = [f"mixed__{side}" for side in ("A", "B", "both")]
    result["mixed"] = {
        key: sum(float(rows[name][key]) for name in names) / len(names) for key in METRICS
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--stage103", default="__PROJECT_ROOT__/stage10_3_p3_nomask")
    args = parser.parse_args()
    root, s103 = Path(args.root), Path(args.stage103)
    paths = {
        "P0-NoMask": root / "results" / "p0_nomask" / "all_metrics.json",
        "P1-NoMask": root / "results" / "p1_nomask" / "all_metrics.json",
        "P2-NoMask": root / "results" / "p2_nomask" / "all_metrics.json",
        "P3-NoMask": s103 / "results" / "p3_nomask" / "all_metrics.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing result files:\n" + "\n".join(missing))
    data = {name: grouped(load(path)) for name, path in paths.items()}
    lines = [
        "# P0–P2完整无Mask消融",
        "",
        "> 所有模型从相同的无-mask partial encoder与clean caption teacher独立初始化；P0冻结encoder，P1开启端到端更新，P2增加clean-caption约束，P3进一步增加relation KD。",
        "",
        "| Model | Clean B4 | Clean R-L | Clean CIDEr | Seen B4 | Seen R-L | Seen CIDEr | Unseen B4 | Unseen R-L | Unseen CIDEr | Mixed B4 | Mixed R-L | Mixed CIDEr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, groups in data.items():
        values = []
        for group in ("clean", "seen", "unseen", "mixed"):
            values.extend(groups[group][key] for key in METRICS)
        lines.append("| " + model + " | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    lines += ["", "## 相邻消融的绝对差", "", "| Comparison | Group | ΔB4 | ΔR-L | ΔCIDEr |", "|---|---|---:|---:|---:|"]
    comparisons = (("P1-P0", "P1-NoMask", "P0-NoMask"), ("P2-P1", "P2-NoMask", "P1-NoMask"), ("P3-P2", "P3-NoMask", "P2-NoMask"))
    for label, newer, older in comparisons:
        for group in ("clean", "seen", "unseen", "mixed"):
            delta = {key: data[newer][group][key] - data[older][group][key] for key in METRICS}
            lines.append(f"| {label} | {group} | {delta['BLEU-4']:+.6f} | {delta['ROUGE-L']:+.6f} | {delta['CIDEr']:+.6f} |")
    output = root / "results" / "summary.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "results" / "summary.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(output.read_text(encoding="utf-8"))
    print("SUMMARY_OK")


if __name__ == "__main__":
    main()
