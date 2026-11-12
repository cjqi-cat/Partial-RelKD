from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("SECOND_CC", "WHU_CDC")
GROUPS = ("Clean", "Seen", "Unseen", "Mixed")
METRICS = ("BLEU-4", "ROUGE-L", "CIDEr")


def main():
    rows = []
    for dataset in DATASETS:
        result = json.loads((ROOT / "results" / dataset / "summary.json").read_text(encoding="utf-8"))
        rows.append((dataset, [result[f"{group} {metric}"] for group in GROUPS for metric in METRICS]))
    lines = [
        "# RMNet multi-dataset geometry-quality LOFO",
        "",
        "| Dataset | Clean B4 | Clean R-L | Clean CIDEr | Seen B4 | Seen R-L | Seen CIDEr | Unseen B4 | Unseen R-L | Unseen CIDEr | Mixed B4 | Mixed R-L | Mixed CIDEr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, values in rows:
        lines.append("| " + dataset + " | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    lines.extend(["", "> Each row is independently target-supervised. Checkpoints are selected by clean validation BLEU-4; unseen and mixed conditions are evaluation only.", ""])
    text = "\n".join(lines)
    (ROOT / "results" / "rmnet_multidataset_summary.md").write_text(text, encoding="utf-8")
    print(text, end="")
    print("MULTIDATASET_SUMMARY_OK")


if __name__ == "__main__":
    main()
