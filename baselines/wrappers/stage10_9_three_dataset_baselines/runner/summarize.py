from __future__ import annotations

import json
from pathlib import Path

from common import SEEN, UNSEEN

ROOT = Path("__PROJECT_ROOT__/stage10_9_three_dataset_baselines")
PROJECT = ROOT.parent
METRICS = ("BLEU-4", "ROUGE-L", "CIDEr")


def load(path): return json.loads(Path(path).read_text())


def means(rows):
    def avg(kinds, metric):
        vals = [v[metric] for name, v in rows.items() if "__" in name and name.split("__")[0] in kinds]
        return sum(vals) / len(vals)
    mixed = [v for name, v in rows.items() if name.startswith("mixed__")]
    out = {}
    for metric in METRICS:
        out[f"Clean {metric}"] = rows["clean"][metric]
        out[f"Seen {metric}"] = avg(SEEN, metric); out[f"Unseen {metric}"] = avg(UNSEEN, metric)
        out[f"Mixed {metric}"] = sum(x[metric] for x in mixed) / len(mixed)
    return out


def source(dataset, model):
    new = ROOT / "results" / dataset / model / "all_metrics.json"
    if new.exists(): return new
    if dataset == "LEVIR_MCI":
        paths = {
            "Change-Agent": PROJECT / "stage10_7_severity_curves/results/change_agent/severity_2/all_metrics.json",
            "chg2cap": PROJECT / "stage10_authoritative_baselines/results/chg2cap/all_metrics.json",
            "mccformer_d": PROJECT / "stage10_authoritative_baselines/results/mccformer_d/all_metrics.json",
            "rsiccformer": PROJECT / "stage10_authoritative_baselines/results/rsiccformer/all_metrics.json",
            "sen": PROJECT / "stage10_authoritative_baselines/results/sen/all_metrics.json",
            "DINO-Clean": PROJECT / "stage8_2_dino_controls/results/LEVIR_MCI/dino_clean/all_metrics.json",
            "DINO-Aug": PROJECT / "stage8_2_dino_controls/results/LEVIR_MCI/dino_aug/all_metrics.json",
            "P3": PROJECT / "stage6_7_partial_hybrid/results/p3_partial_relation/all_metrics.json",
            "P3-NoMask": PROJECT / "stage10_7_severity_curves/results/p3_nomask/severity_2/all_metrics.json",
        }
    else:
        paths = {
            "Change-Agent": PROJECT / f"stage8_1_multidataset_full/results/{dataset}/change_agent/all_metrics.json",
            "DINO-Clean": PROJECT / f"stage8_2_dino_controls/results/{dataset}/dino_clean/all_metrics.json",
            "DINO-Aug": PROJECT / f"stage8_2_dino_controls/results/{dataset}/dino_aug/all_metrics.json",
            "P3": PROJECT / f"stage8_1_multidataset_full/results/{dataset}/p3/all_metrics.json",
            "P3-NoMask": PROJECT / f"stage10_8_main_completion/results/cross/{dataset}/p3_nomask/all_metrics.json",
        }
    return paths.get(model)


def table(dataset):
    models = ("Change-Agent", "chg2cap", "mccformer_d", "rsiccformer", "sen",
              "DINO-Clean", "DINO-Aug", "P3", "P3-NoMask")
    display = {"chg2cap":"Chg2Cap", "mccformer_d":"MCCFormer-D", "rsiccformer":"RSICCFormer",
               "sen":"SEN-ImageNet"}
    rows = []
    for model in models:
        path = source(dataset, model)
        if path and path.exists():
            values = load(path)
            if dataset == "LEVIR_MCI" and "clean" not in values and model in ("Change-Agent", "P3-NoMask"):
                leaf = "change_agent" if model == "Change-Agent" else "p3_nomask"
                clean_path = PROJECT / f"stage10_7_severity_curves/results/{leaf}/clean/clean.metrics.json"
                if clean_path.exists(): values["clean"] = load(clean_path)
            rows.append({"Model": display.get(model, model), **means(values)})
    cols = ["Model"] + [f"{g} {m}" for g in ("Clean", "Seen", "Unseen", "Mixed") for m in METRICS]
    lines = [f"# {dataset} target-supervised comparison", "",
             "> All models use this dataset's own supervised training split and the common geometry-quality LOFO, calibrated-v3, severity-2, and degradation-seed protocol.", "",
             "|" + "|".join(cols) + "|", "|" + "|".join(["---"] + ["---:"] * (len(cols)-1)) + "|"]
    for row in rows: lines.append("|" + "|".join([row["Model"]] + [f"{row[x]:.6f}" for x in cols[1:]]) + "|")
    out = ROOT / "results" / dataset; out.mkdir(parents=True, exist_ok=True)
    (out / "unified_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "unified_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n".join(lines))


def main():
    for dataset in ("LEVIR_MCI", "SECOND_CC", "WHU_CDC"): table(dataset)
    print("SUMMARY_OK")


if __name__ == "__main__": main()
