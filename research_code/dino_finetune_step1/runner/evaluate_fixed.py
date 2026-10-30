#!/usr/bin/env python3
"""Evaluate fine-tuned DINO variants on the fixed 400-pair x 31-condition benchmark.

Degradations are generated in memory with the same seed protocol as the original
LEVIR-MCI degraded benchmark, so the server does not need 115,740 degraded PNGs.
Results are written after every condition and are safe to resume.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

from common import load_config
from degradations import OPERATORS
from model import DinoRobustModel


MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
CORRUPTIONS = tuple(OPERATORS)
TARGETS = ("A", "B", "both")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[], help="NAME=PATH; repeatable")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--severity", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-conditions", type=int)
    return parser.parse_args()


def stable_seed(base_seed: int, *parts: str):
    payload = "|".join((str(base_seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def to_tensor(image: Image.Image):
    image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1) - MEAN) / STD


def patch_mask(path: Path):
    with Image.open(path) as opened:
        array = (np.asarray(opened.convert("L"), dtype=np.uint8) > 0).astype(np.float32)
    return (F.adaptive_max_pool2d(torch.from_numpy(array)[None, None], (16, 16)).flatten() > 0).numpy().astype(np.uint8)


def load_model(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DinoRobustModel(checkpoint["config"])
    missing, unexpected = model.load_state_dict(checkpoint["trainable_state"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    logging.info("loaded checkpoint=%s epoch=%s best_score=%s frozen_missing=%d", checkpoint_path, checkpoint.get("epoch"), checkpoint.get("best_score"), len(missing))
    return model.to(device).eval(), checkpoint


def batch_images(dataset_root: Path, names: list[str], condition: str, severity: int, seed: int):
    corruption = target = None
    if condition != "clean":
        corruption, target = condition.split("__", 1)
    images_a, images_b = [], []
    for name in names:
        for side, output in (("A", images_a), ("B", images_b)):
            with Image.open(dataset_root / "images" / "test" / side / name) as opened:
                image = opened.convert("RGB").copy()
            if condition != "clean" and (target == side or target == "both"):
                rng = np.random.default_rng(stable_seed(seed, corruption, target, name, side))
                image, _ = OPERATORS[corruption](image, severity, rng)
            output.append(to_tensor(image))
    return torch.stack(images_a), torch.stack(images_b)


def extract_condition(model, dataset_root, names, condition, severity, seed, batch_size, device):
    features_a, features_b = [], []
    with torch.inference_mode():
        for start in range(0, len(names), batch_size):
            subset = names[start:start + batch_size]
            images_a, images_b = batch_images(dataset_root, subset, condition, severity, seed)
            images = torch.cat((images_a, images_b)).to(device, non_blocking=True)
            _, adapted = model.encode_images(images)
            adapted = adapted.cpu().numpy().astype(np.float16)
            size = len(subset)
            features_a.append(adapted[:size]); features_b.append(adapted[size:])
    return np.concatenate(features_a), np.concatenate(features_b)


def unit(array):
    array = array.astype(np.float32, copy=False)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8)


def cosine(a, b):
    return np.sum(a * b, axis=-1)


def metrics(clean_a, clean_b, deg_a, deg_b, masks, labels, condition):
    clean_a, clean_b, deg_a, deg_b = [unit(x) for x in (clean_a, clean_b, deg_a, deg_b)]
    labels = np.asarray(labels, dtype=int)
    changed = labels == 1
    row = {"condition": condition, "n": len(labels)}
    if condition == "clean":
        row.update(patch_invariance=float("nan"), pooled_cosine_invariance=float("nan"), delta_direction_cosine=float("nan"), delta_valid_patches=0, retention_ratio_of_means=float("nan"))
    else:
        row["patch_invariance"] = float(np.r_[cosine(clean_a, deg_a).mean(1), cosine(clean_b, deg_b).mean(1)].mean())
        pooled = np.r_[cosine(unit(clean_a.mean(1)), unit(deg_a.mean(1))), cosine(unit(clean_b.mean(1)), unit(deg_b.mean(1)))]
        row["pooled_cosine_invariance"] = float(pooled.mean())
        clean_delta, deg_delta = clean_b - clean_a, deg_b - deg_a
        clean_norm, deg_norm = np.linalg.norm(clean_delta, axis=-1), np.linalg.norm(deg_delta, axis=-1)
        valid = masks.astype(bool) & changed[:, None] & (clean_norm > 1e-6) & (deg_norm > 1e-6)
        direction = np.sum(clean_delta * deg_delta, axis=-1) / np.maximum(clean_norm * deg_norm, 1e-8)
        row["delta_direction_cosine"] = float(direction[valid].mean()) if valid.any() else float("nan")
        row["delta_valid_patches"] = int(valid.sum())
    patch_distance = 1 - cosine(deg_a, deg_b)
    clean_distance = 1 - cosine(clean_a, clean_b)
    top10 = np.sort(patch_distance, axis=1)[:, -max(1, patch_distance.shape[1] // 10):].mean(1)
    row["distance_auc_patch_mean"] = float(roc_auc_score(labels, patch_distance.mean(1)))
    row["distance_auc_patch_top10"] = float(roc_auc_score(labels, top10))
    row["patch_pixel_auc"] = float(roc_auc_score(masks.ravel().astype(int), patch_distance.ravel()))
    row["change_dist"] = float(patch_distance.mean(1)[changed].mean())
    row["nochange_dist"] = float(patch_distance.mean(1)[~changed].mean())
    if condition != "clean":
        row["retention_ratio_of_means"] = float(patch_distance.mean(1)[changed].mean() / max(clean_distance.mean(1)[changed].mean(), 1e-6))
    return row


def write_csv(path: Path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(args.output / "logs" / "fixed_eval.log", encoding="utf-8"), logging.StreamHandler()])
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.limit:
        manifest = manifest[:args.limit]
    names = [row["name"] for row in manifest]
    labels = np.asarray([int(row["is_change"]) for row in manifest], dtype=int)
    masks = np.stack([patch_mask(args.dataset_root / "images" / "test" / "label" / name) for name in names])
    conditions = ["clean"] + [f"{corruption}__{target}" for corruption in CORRUPTIONS for target in TARGETS]
    if args.max_conditions:
        conditions = conditions[:args.max_conditions]
    logging.info("samples=%d change=%d nochange=%d conditions=%d", len(names), int(labels.sum()), int((1-labels).sum()), len(conditions))
    summaries = []
    for entry in args.checkpoint:
        variant, checkpoint_text = entry.split("=", 1)
        variant_dir = args.output / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        model, checkpoint = load_model(Path(checkpoint_text), args.device)
        clean_a, clean_b = extract_condition(model, args.dataset_root, names, "clean", args.severity, args.seed, args.batch_size, args.device)
        metrics_path = variant_dir / "metrics.csv"
        rows = []
        if metrics_path.exists():
            rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8-sig")))
        done = {row["condition"] for row in rows}
        started = time.time()
        for index, condition in enumerate(conditions, 1):
            if condition in done:
                logging.info("variant=%s skip condition=%s %d/%d", variant, condition, index, len(conditions))
                continue
            if condition == "clean":
                deg_a, deg_b = clean_a, clean_b
            else:
                deg_a, deg_b = extract_condition(model, args.dataset_root, names, condition, args.severity, args.seed, args.batch_size, args.device)
            rows.append(metrics(clean_a, clean_b, deg_a, deg_b, masks, labels, condition))
            write_csv(metrics_path, rows)
            logging.info("variant=%s condition=%s %d/%d elapsed_min=%.1f", variant, condition, index, len(conditions), (time.time()-started)/60)
        degraded = [row for row in rows if row["condition"] != "clean"]
        clean = [row for row in rows if row["condition"] == "clean"][0]
        keys = ("patch_invariance", "pooled_cosine_invariance", "delta_direction_cosine", "distance_auc_patch_mean", "distance_auc_patch_top10", "patch_pixel_auc", "retention_ratio_of_means")
        summary = {"variant": variant, "checkpoint_epoch": checkpoint.get("epoch"), "checkpoint_score": checkpoint.get("best_score"), "clean_top10_auc": float(clean["distance_auc_patch_top10"])}
        summary.update({f"mean_{key}": float(np.nanmean([float(row[key]) for row in degraded])) for key in keys})
        summaries.append(summary)
        (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(args.output / "summary.csv", summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
