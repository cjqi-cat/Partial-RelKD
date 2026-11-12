from __future__ import annotations

import argparse
import json
from pathlib import Path

SEEN = ("cloud", "haze", "gaussian_noise", "jpeg", "pixel_dropout", "radiometric_shift")
UNSEEN = ("blur", "low_resolution", "misregistration")
METRICS = ("BLEU-4", "ROUGE-L", "CIDEr")


def group_mean(rows, operators, metric):
    values = [payload[metric] for name, payload in rows.items() if "__" in name and name.split("__")[0] in operators]
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = json.loads((Path(args.input) / "all_metrics.json").read_text(encoding="utf-8"))
    result = {"model": "RMNet", "dataset": args.dataset, "protocol": "geometry-quality LOFO"}
    for metric in METRICS:
        result[f"Clean {metric}"] = rows["clean"][metric]
        result[f"Seen {metric}"] = group_mean(rows, SEEN, metric)
        result[f"Unseen {metric}"] = group_mean(rows, UNSEEN, metric)
        result[f"Mixed {metric}"] = group_mean(rows, ("mixed",), metric)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    values = [result[f"{group} {metric}"] for group in ("Clean", "Seen", "Unseen", "Mixed") for metric in METRICS]
    text = "\n".join([
        f"# RMNet — {args.dataset} geometry-quality LOFO",
        "",
        "| Model | Clean B4 | Clean R-L | Clean CIDEr | Seen B4 | Seen R-L | Seen CIDEr | Unseen B4 | Unseen R-L | Unseen CIDEr | Mixed B4 | Mixed R-L | Mixed CIDEr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| RMNet | " + " | ".join(f"{value:.6f}" for value in values) + " |",
        "",
        "> Independently target-supervised retraining; checkpoint selected using clean validation BLEU-4 only.",
        "",
    ])
    output.write_text(text, encoding="utf-8")
    print(text, end="")
    print("SUMMARY_OK")


if __name__ == "__main__":
    main()
