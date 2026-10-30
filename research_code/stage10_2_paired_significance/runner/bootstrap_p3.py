#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT = Path("__PROJECT_ROOT__")
METRIC_ROOT = PROJECT / "stage3_caption_retrain/vendor/Change-Agent/Multi_change"
MODELS = {
    "Partial-RelKD": PROJECT / "stage6_7_partial_hybrid/results/p3_partial_relation",
    "DINO-Aug": PROJECT / "stage8_2_dino_controls/results/LEVIR_MCI/dino_aug",
}
COMPARISONS = (("Partial-RelKD", "DINO-Aug"),)
SIDES = ("A", "B", "both")
UNSEEN_TYPES = ("blur", "low_resolution", "misregistration")
CONDITIONS = tuple(f"{kind}__{side}" for kind in UNSEEN_TYPES for side in SIDES)
METRICS = ("BLEU-4", "ROUGE-L", "CIDEr")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_metrics(row):
    source = row.get("metrics", row)
    return {
        "BLEU-4": float(source.get("BLEU-4", source.get("BLEU_4"))),
        "ROUGE-L": float(source.get("ROUGE-L", source.get("ROUGE_L"))),
        "CIDEr": float(source["CIDEr"]),
    }


def load_metrics(root: Path):
    return {name: normalize_metrics(row) for name, row in load_json(root / "all_metrics.json").items()}


def load_prediction(path: Path):
    predictions, references = {}, {}
    for row in load_json(path):
        name = str(row["name"])
        if name in predictions:
            raise RuntimeError(f"duplicate sample {name} in {path}")
        predictions[name] = row["prediction_ids"]
        references[name] = row["reference_ids"]
    return predictions, references


def metric_arrays(refs, hyps):
    root = str(METRIC_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from eval_func.bleu.bleu import Bleu
    from eval_func.rouge.rouge import Rouge
    from eval_func.cider.cider import Cider

    gts = {i: [" ".join(map(str, row)) for row in group] for i, group in enumerate(refs)}
    res = {i: [" ".join(map(str, row))] for i, row in enumerate(hyps)}
    with contextlib.redirect_stdout(io.StringIO()):
        _, bleu = Bleu(4).compute_score(gts, res)
        _, rouge = Rouge().compute_score(gts, res)
        _, cider = Cider().compute_score(gts, res)
    return {
        "BLEU-4": np.asarray(bleu[3], dtype=np.float64),
        "ROUGE-L": np.asarray(rouge, dtype=np.float64),
        "CIDEr": np.asarray(cider, dtype=np.float64),
    }


def build_cache(cache_path: Path):
    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=False)
        print(f"loaded cache {cache_path}", flush=True)
        return {key: loaded[key] for key in loaded.files}

    cache = {}
    expected_names = None
    for index, condition in enumerate(CONDITIONS, 1):
        _, anchor_refs = load_prediction(MODELS["Partial-RelKD"] / f"{condition}.predictions.json")
        names = sorted(anchor_refs)
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise RuntimeError(f"{condition}: sample IDs differ across conditions")
        refs = [anchor_refs[name] for name in names]
        for model, root in MODELS.items():
            predictions, model_refs = load_prediction(root / f"{condition}.predictions.json")
            if sorted(predictions) != names:
                raise RuntimeError(f"{model}/{condition}: sample IDs do not align")
            if any(model_refs[name] != anchor_refs[name] for name in names):
                raise RuntimeError(f"{model}/{condition}: references do not align")
            for metric, values in metric_arrays(refs, [predictions[name] for name in names]).items():
                cache[f"{model}::{condition}::{metric}"] = values
        print(f"scored {index}/{len(CONDITIONS)} {condition}", flush=True)
    np.savez_compressed(cache_path, **cache)
    return cache


def aggregate(metrics, metric):
    return float(np.mean([metrics[condition][metric] for condition in CONDITIONS]))


def bootstrap(deltas, reps: int, seed: int):
    # Rows are conditions and columns are the same original test images.
    # Average conditions first, then resample image clusters synchronously.
    per_image = np.stack(deltas, axis=0).mean(axis=0)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, 250):
        count = min(250, reps - start)
        indices = rng.integers(0, per_image.size, size=(count, per_image.size))
        draws[start:start + count] = per_image[indices].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    # Plus-one correction avoids reporting an exact zero Monte Carlo p-value.
    non_positive = int(np.count_nonzero(draws <= 0))
    non_negative = int(np.count_nonzero(draws >= 0))
    p = min(1.0, 2.0 * min((non_positive + 1) / (reps + 1), (non_negative + 1) / (reps + 1)))
    return {
        "sentence_delta": float(per_image.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_two_sided": float(p),
        "probability_positive": float(np.mean(draws > 0)),
        "samples": int(per_image.size),
        "conditions": len(CONDITIONS),
    }


def holm_all(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_two_sided"])
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * rows[index]["p_two_sided"])
        running = max(running, adjusted)
        rows[index]["p_holm"] = running


def format_p(value, reps):
    floor = 2.0 / (reps + 1)
    return f"<{floor:.4g}" if value <= floor else f"{value:.4g}"


def write_outputs(rows, output: Path, reps: int, seed: int, elapsed: float):
    fields = (
        "comparison", "metric", "corpus_metric_delta", "sentence_delta",
        "ci95_low", "ci95_high", "p_two_sided", "p_holm",
        "probability_positive", "samples", "conditions",
    )
    with (output / "paired_bootstrap.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "paired_bootstrap.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# LEVIR-MCI Partial-RelKD vs DINO-Aug paired clustered Bootstrap", "",
        f"- 重复次数：{reps}",
        f"- 随机种子：{seed}",
        "- 范围：9 个 geometry-quality 完整留出退化条件。",
        "- 聚类单位：原始测试图像 ID；同一图像的 9 个条件同步重采样。",
        "- Multiple testing: Holm correction across three metrics.", "",
        "| 比较 | 指标 | corpus Δ | 逐样本 Δ | 95% CI | 原始 p | Holm p | 判定 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        positive = row["ci95_low"] > 0 and row["p_holm"] < 0.05
        negative = row["ci95_high"] < 0 and row["p_holm"] < 0.05
        verdict = "显著正向" if positive else ("显著负向" if negative else "未显著")
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['corpus_metric_delta']:+.6f} | "
            f"{row['sentence_delta']:+.6f} | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | "
            f"{format_p(row['p_two_sided'], reps)} | {format_p(row['p_holm'], reps)} | {verdict} |"
        )
    lines += ["", "This fixed-checkpoint test does not replace independent training-seed runs.", ""]
    (output / "paired_bootstrap_summary.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "bootstrap_repetitions": reps,
        "bootstrap_seed": seed,
        "conditions": list(CONDITIONS),
        "comparisons": [f"{a} - {b}" for a, b in COMPARISONS],
        "holm_family_size": len(rows),
        "severity": 2,
        "degradation_seed": 20260928,
        "metric_root": str(METRIC_ROOT),
        "elapsed_seconds": elapsed,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    metrics = {model: load_metrics(root) for model, root in MODELS.items()}
    cache = build_cache(args.output / "unseen_sentence_scores.npz")
    rows = []
    for comparison_index, (left, right) in enumerate(COMPARISONS):
        for metric_index, metric in enumerate(METRICS):
            deltas = [
                cache[f"{left}::{condition}::{metric}"] - cache[f"{right}::{condition}::{metric}"]
                for condition in CONDITIONS
            ]
            result = bootstrap(deltas, args.reps, args.seed + comparison_index * 10 + metric_index)
            rows.append({
                "comparison": f"{left} - {right}",
                "metric": metric,
                "corpus_metric_delta": aggregate(metrics[left], metric) - aggregate(metrics[right], metric),
                **result,
            })
    holm_all(rows)
    write_outputs(rows, args.output, args.reps, args.seed, time.time() - started)
    print(f"PAIRED_SIGNIFICANCE_OK elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
