#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from common import atomic_torch_save, balanced_names, load_config, load_degradations, seed_everything, trainable_state_dict
from model import DinoRobustModel, GradientReverse


MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
CORRUPTIONS = ("cloud", "haze", "gaussian_noise", "blur", "jpeg", "low_resolution", "pixel_dropout", "misregistration", "radiometric_shift", "mixed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-pairs", type=int)
    parser.add_argument("--val-pairs", type=int)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def to_tensor(image: Image.Image):
    image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1) - MEAN) / STD


class OnlinePairDataset(Dataset):
    def __init__(self, dataset_root: Path, split: str, limit: int, seed: int, degradation_script: Path):
        self.root = dataset_root
        self.split = split
        self.seed = seed + (0 if split == "train" else 10_000_000)
        self.names, self.labels = balanced_names(dataset_root, split, limit, seed)
        self.operators = load_degradations(degradation_script)

    def __len__(self):
        return len(self.names)

    def _degrade(self, clean: Image.Image, rng: np.random.Generator):
        kind = int(rng.integers(0, len(CORRUPTIONS)))
        severity = int(rng.integers(1, 4))
        output, _ = self.operators[CORRUPTIONS[kind]](clean, severity, rng)
        return output, kind

    def __getitem__(self, index: int):
        name = self.names[index]
        views, kinds = [], []
        for side_index, side in enumerate(("A", "B")):
            with Image.open(self.root / "images" / self.split / side / name) as opened:
                clean = opened.convert("RGB").copy()
            rng1 = np.random.default_rng(self.seed + index * 1009 + side_index * 37)
            rng2 = np.random.default_rng(self.seed + index * 1009 + 1_000_003 + side_index * 37)
            deg1, kind1 = self._degrade(clean, rng1)
            deg2, kind2 = self._degrade(clean, rng2)
            views.extend((to_tensor(clean), to_tensor(deg1), to_tensor(deg2)))
            kinds.extend((-1, kind1, kind2))
        with Image.open(self.root / "images" / self.split / "label" / name) as opened:
            mask = (np.asarray(opened.convert("L"), dtype=np.uint8) > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask)[None, None]
        patch_mask = F.adaptive_max_pool2d(mask_tensor, (16, 16)).flatten()
        return torch.stack(views), patch_mask, torch.tensor(self.labels[index], dtype=torch.float32), torch.tensor(kinds, dtype=torch.long)


def nce(a: torch.Tensor, b: torch.Tensor, temperature: float):
    logits = a @ b.T / temperature
    target = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def compute(model: DinoRobustModel, batch, cfg):
    images, mask, pair_label, kinds = batch
    batch_size, view_count = images.shape[:2]
    raw_all, adapted_all = model.encode_images(images.flatten(0, 1))
    raw_views = raw_all.view(batch_size, view_count, *raw_all.shape[1:]).unbind(1)
    z = adapted_all.view(batch_size, view_count, *adapted_all.shape[1:]).unbind(1)
    ca, d1a, d2a, cb, d1b, d2b = z

    patch_count = ca.shape[1]
    selected_count = min(int(cfg["patches_per_image"]), patch_count)
    indices = torch.randperm(patch_count, device=ca.device)[:selected_count]
    positive_pairs = ((ca, d1a), (ca, d2a), (d1a, d2a), (cb, d1b), (cb, d2b), (d1b, d2b))
    patch_inv = sum((1 - (x[:, indices] * y[:, indices]).sum(-1)).mean() for x, y in positive_pairs) / len(positive_pairs)
    global_inv = sum(
        nce(F.normalize(x.mean(1), dim=-1), F.normalize(y.mean(1), dim=-1), float(cfg["temperature"]))
        for x, y in positive_pairs
    ) / len(positive_pairs)
    invariance = patch_inv + global_inv

    distance = 1 - (ca * cb).sum(-1)
    changed = mask > 0.5
    unchanged = ~changed
    margin = float(cfg["change_margin"])
    same_loss = distance[unchanged].pow(2).mean() if unchanged.any() else distance.mean() * 0
    changed_loss = F.relu(margin - distance[changed]).pow(2).mean() if changed.any() else distance.mean() * 0
    change = 0.5 * (same_loss + changed_loss)

    patch_logits = model.change_logits(ca, cb)
    positive = mask.sum().clamp_min(1)
    negative = (1 - mask).sum().clamp_min(1)
    pos_weight = (negative / positive).clamp(max=20)
    patch_cls = F.binary_cross_entropy_with_logits(patch_logits, mask, pos_weight=pos_weight)

    distill = sum((1 - (adapted * raw.detach()).sum(-1)).mean() for adapted, raw in zip(z, raw_views)) / len(z)
    adversarial_parts = []
    for feature, label in ((d1a, kinds[:, 1]), (d2a, kinds[:, 2]), (d1b, kinds[:, 4]), (d2b, kinds[:, 5])):
        reversed_feature = GradientReverse.apply(feature.mean(1), float(cfg.get("grl_scale", 1.0)))
        adversarial_parts.append(F.cross_entropy(model.degradation_head(reversed_feature), label))
    adversarial = sum(adversarial_parts) / len(adversarial_parts)
    anchor = model.backbone_anchor_loss()

    total = (
        float(cfg.get("lambda_invariance", 1.0)) * invariance
        + float(cfg["lambda_change"]) * change
        + float(cfg["lambda_patch_cls"]) * patch_cls
        + float(cfg["lambda_distill"]) * distill
        + float(cfg["lambda_adversarial"]) * adversarial
        + float(cfg.get("lambda_backbone_anchor", 0.0)) * anchor
    )
    parts = {"total": total, "inv": invariance, "change": change, "patch_cls": patch_cls, "distill": distill, "adversarial": adversarial, "anchor": anchor}
    return total, parts, patch_logits, distance, mask, pair_label


def safe_auc(target, score):
    return float(roc_auc_score(target, score)) if len(np.unique(target)) > 1 else float("nan")


def run_epoch(model, loader, cfg, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    sums = {key: 0.0 for key in ("total", "inv", "change", "patch_cls", "distill", "adversarial", "anchor")}
    count, patch_targets, patch_scores, pair_targets, pair_scores = 0, [], [], [], []
    amp = bool(cfg.get("amp", True)) and device.startswith("cuda")
    accumulation = int(cfg.get("gradient_accumulation", 1)) if training else 1
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, 1):
        batch = [value.to(device, non_blocking=True) for value in batch]
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                total, parts, logits, distance, mask, pairs = compute(model, batch, cfg)
                scaled_loss = total / accumulation
        if training:
            scaler.scale(scaled_loss).backward()
            if step % accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(cfg.get("grad_clip", 5.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        size = len(pairs)
        count += size
        for key in sums:
            sums[key] += float(parts[key].detach()) * size
        patch_targets.append(mask.detach().cpu().numpy().ravel())
        patch_scores.append(logits.detach().sigmoid().cpu().numpy().ravel())
        pair_targets.append(pairs.detach().cpu().numpy())
        top_count = max(1, distance.shape[1] // 10)
        pair_scores.append(distance.detach().topk(top_count, dim=1).values.mean(1).cpu().numpy())
    patch_target = np.concatenate(patch_targets)
    patch_score = np.concatenate(patch_scores)
    pair_target = np.concatenate(pair_targets)
    pair_score = np.concatenate(pair_scores)
    output = {key: value / count for key, value in sums.items()}
    output["patch_auc"] = safe_auc(patch_target, patch_score)
    output["pair_top10_auc"] = safe_auc(pair_target, pair_score)
    return output


def build_optimizer(model, cfg):
    backbone, task = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if name.startswith("backbone.") else task).append(parameter)
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": float(cfg["backbone_learning_rate"])})
    if task:
        groups.append({"params": task, "lr": float(cfg["task_learning_rate"])})
    return torch.optim.AdamW(groups, weight_decay=float(cfg["weight_decay"]))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output = args.output or args.config.parent.parent / "runs" / cfg["run_name"]
    device = args.device or cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for path in (output / "logs", output / "checkpoints", output / "results"):
        path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(output / "logs" / "train.log", encoding="utf-8"), logging.StreamHandler()])
    seed_everything(int(cfg["seed"]))
    model = DinoRobustModel(cfg).to(device)
    report = model.parameter_report()
    logging.info("mode=%s parameters=%s", cfg["finetune_mode"], report)
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and bool(cfg.get("amp", True)))
    start_epoch, best, stale = 1, -math.inf, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["trainable_state"], strict=False)
        allowed_missing = [name for name in missing if name not in checkpoint["trainable_state"]]
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_score"])
        logging.info("resumed epoch=%d best=%.6f ignored_frozen_missing=%d", start_epoch, best, len(allowed_missing))

    dataset_root = Path(cfg["dataset_root"])
    degradation_script = Path(cfg["degradation_script"])
    train_pairs = int(args.train_pairs or cfg["train_pairs"])
    val_pairs = int(args.val_pairs or cfg["val_pairs"])
    train_dataset = OnlinePairDataset(dataset_root, "train", train_pairs, int(cfg["seed"]), degradation_script)
    val_dataset = OnlinePairDataset(dataset_root, "val", val_pairs, int(cfg["seed"]), degradation_script)
    loader_kwargs = {"batch_size": int(cfg["batch_size_pairs"]), "num_workers": int(cfg.get("num_workers", 4)), "pin_memory": device.startswith("cuda")}
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    epochs = int(args.epochs or cfg["epochs"])
    rows, started = [], time.time()
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = run_epoch(model, train_loader, cfg, device, optimizer, scaler)
        val_metrics = run_epoch(model, val_loader, cfg, device)
        score = 0.55 * val_metrics["patch_auc"] + 0.45 * val_metrics["pair_top10_auc"] - 0.02 * val_metrics["inv"]
        row = {"epoch": epoch, "score": score, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        rows.append(row)
        logging.info("epoch=%d train=%.5f val=%.5f patch_auc=%.5f top10_auc=%.5f inv=%.5f score=%.5f elapsed_min=%.1f", epoch, train_metrics["total"], val_metrics["total"], val_metrics["patch_auc"], val_metrics["pair_top10_auc"], val_metrics["inv"], score, (time.time() - started) / 60)
        improved = score > best + float(cfg.get("min_delta", 1e-5))
        if improved:
            best, stale = score, 0
        else:
            stale += 1
        payload = {"trainable_state": trainable_state_dict(model), "optimizer": optimizer.state_dict(), "config": cfg, "epoch": epoch, "best_score": best, "parameter_report": report}
        atomic_torch_save(payload, output / "checkpoints" / "last.pt")
        if improved:
            atomic_torch_save(payload, output / "checkpoints" / "best.pt")
        with (output / "results" / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output / "results" / "summary.json").write_text(json.dumps({"status": "running", "best_score": best, "latest_epoch": epoch, "stale": stale, "parameter_report": report}, indent=2), encoding="utf-8")
        if stale >= int(cfg["patience"]):
            logging.info("early_stop epoch=%d best=%.5f", epoch, best)
            break
    (output / "results" / "summary.json").write_text(json.dumps({"status": "complete", "best_score": best, "epochs": rows[-1]["epoch"], "parameter_report": report}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
