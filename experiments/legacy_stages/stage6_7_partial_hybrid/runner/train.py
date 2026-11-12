#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import atomic_save, load_json, save_json, seed_everything
from data import CaptionTokens, OnlineCaptionPairs
from losses import caption_loss, relation_kd, token_kd
from models import build_model, frozen_teacher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def move(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def overlay_state(model):
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable or name.startswith("captioner.")
    }


def optimizer_groups(model, cfg):
    lora, backbone, main = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder."):
            (lora if ".lora_a" in name or ".lora_b" in name else backbone).append(parameter)
        else:
            main.append(parameter)
    groups = []
    if lora:
        groups.append({"params": lora, "lr": float(cfg.get("lora_lr", cfg.get("backbone_lr", 1e-6)))})
    if backbone:
        groups.append({"params": backbone, "lr": float(cfg.get("backbone_lr", 1e-6))})
    if main:
        groups.append({"params": main, "lr": float(cfg.get("main_lr", 5e-5))})
    return groups


@torch.no_grad()
def validate(model, loader, cfg, tokens, device):
    model.eval()
    dirty_total = clean_total = 0.0
    count = 0
    clean_weight = float(cfg.get("selection_clean_weight", 0.0))
    for batch in loader:
        batch = move(batch, device)
        decoder_in, targets = batch["tokens"][:, :-1], batch["tokens"][:, 1:]
        dirty = caption_loss(
            model(batch["a"], batch["b"], decoder_in), targets, tokens.pad, 0.0
        )
        clean = dirty.new_zeros(())
        if clean_weight > 0:
            clean = caption_loss(
                model(batch["clean_a"], batch["clean_b"], decoder_in), targets, tokens.pad, 0.0
            )
        size = batch["a"].size(0)
        count += size
        dirty_total += float(dirty) * size
        clean_total += float(clean) * size
    dirty_mean = dirty_total / max(count, 1)
    clean_mean = clean_total / max(count, 1) if clean_weight > 0 else 0.0
    return {
        "dirty": dirty_mean,
        "clean": clean_mean,
        "score": dirty_mean + clean_weight * clean_mean,
    }


def main():
    args = parse_args()
    cfg = load_json(args.config)
    seed_everything(int(cfg["seed"]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(cfg, output / "config.json")
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    tokens = CaptionTokens(cfg["metadata_root"], int(cfg.get("max_length", 41)))
    train_set = OnlineCaptionPairs(cfg, "train", tokens, training=True)
    val_set = OnlineCaptionPairs(cfg, "val", tokens, training=True)
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg.get("val_batch_size", cfg["batch_size"])),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )

    student = build_model(cfg, tokens.vocab).to(device)
    teacher = frozen_teacher(student)
    optimizer = torch.optim.AdamW(
        optimizer_groups(student, cfg), weight_decay=float(cfg.get("weight_decay", 1e-4))
    )
    epochs = int(cfg.get("epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, best, stale, history = 0, float("inf"), 0, []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected = student.load_state_dict(checkpoint["trainable_state"], strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected resume keys: {unexpected}")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_val_loss"])
        stale = int(checkpoint.get("stale", 0))
        history = checkpoint.get("history", [])
        print(f"resumed epoch={start_epoch} best={best:.6f} ignored_missing={len(missing)}", flush=True)

    lambda_clean = float(cfg.get("lambda_clean", 0.0))
    lambda_relation = float(cfg.get("lambda_relation", 0.0))
    lambda_token = float(cfg.get("lambda_token_kd", 0.5))
    lambda_anchor = float(cfg.get("lambda_backbone_anchor", 0.001))

    for epoch in range(start_epoch, epochs):
        train_set.set_epoch(epoch)
        student.train()
        teacher.eval()
        sums = {"dirty_caption": 0.0, "clean_caption": 0.0, "token_kd": 0.0, "relation": 0.0, "anchor": 0.0, "total": 0.0}
        seen = 0
        started = time.time()
        for step, batch in enumerate(train_loader, 1):
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            decoder_in, targets = batch["tokens"][:, :-1], batch["tokens"][:, 1:]
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                dirty_memory = student.memory(batch["a"], batch["b"])
                dirty_logits = student.decode_memory(dirty_memory, decoder_in)
                dirty_cap = caption_loss(
                    dirty_logits,
                    targets,
                    tokens.pad,
                    float(cfg.get("label_smoothing", 0.05)),
                )
                with torch.no_grad():
                    teacher_memory = teacher.memory(batch["clean_a"], batch["clean_b"])
                    teacher_logits = teacher.decode_memory(teacher_memory, decoder_in)
                kd = token_kd(
                    dirty_logits,
                    teacher_logits,
                    targets,
                    tokens.pad,
                    float(cfg.get("kd_temperature", 2.0)),
                )
                clean_cap = dirty_cap.new_zeros(())
                if lambda_clean > 0:
                    clean_memory = student.memory(batch["clean_a"], batch["clean_b"])
                    clean_logits = student.decode_memory(clean_memory, decoder_in)
                    clean_cap = caption_loss(
                        clean_logits,
                        targets,
                        tokens.pad,
                        float(cfg.get("label_smoothing", 0.05)),
                    )
                relation = relation_kd(dirty_memory, teacher_memory) if lambda_relation > 0 else dirty_cap.new_zeros(())
                anchor = student.encoder.backbone_anchor_loss() if lambda_anchor > 0 else dirty_cap.new_zeros(())
                total = (
                    dirty_cap
                    + lambda_token * kd
                    + lambda_clean * clean_cap
                    + lambda_relation * relation
                    + lambda_anchor * anchor
                )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in student.parameters() if parameter.requires_grad],
                float(cfg.get("grad_clip", 5.0)),
            )
            scaler.step(optimizer)
            scaler.update()
            size = batch["a"].size(0)
            seen += size
            values = {
                "dirty_caption": dirty_cap,
                "clean_caption": clean_cap,
                "token_kd": kd,
                "relation": relation,
                "anchor": anchor,
                "total": total,
            }
            for key, value in values.items():
                sums[key] += float(value.detach()) * size
            if args.smoke_batches and step >= args.smoke_batches:
                break

        scheduler.step()
        val = validate(student, val_loader, cfg, tokens, device)
        row = {
            "epoch": epoch,
            **{key: value / max(seen, 1) for key, value in sums.items()},
            "val_dirty": val["dirty"],
            "val_clean": val["clean"],
            "val_score": val["score"],
            "seconds": time.time() - started,
        }
        history.append(row)
        print(row, flush=True)
        save_json(history, output / "history.json")
        improved = val["score"] < best - float(cfg.get("min_delta", 1e-5))
        if improved:
            best, stale = val["score"], 0
        else:
            stale += 1
        payload = {
            "architecture": cfg["architecture"],
            "variant": cfg["run_name"],
            "epoch": epoch,
            "best_val_loss": best,
            "stale": stale,
            "config": cfg,
            "trainable_state": overlay_state(student),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
        }
        atomic_save(payload, output / "last.pt")
        if improved:
            atomic_save(payload, output / "best.pt")
        if args.smoke_batches:
            break
        if stale >= int(cfg.get("patience", 8)):
            break


if __name__ == "__main__":
    main()
