#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import atomic_save, load_json, save_json, seed_everything
from data import CaptionTokens, OnlineCaptionPairs
from losses import caption_loss, token_kd
from models import build_model, frozen_teacher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=("m1", "m2"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def move(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def overlay_state(model, architecture):
    if architecture == "change_agent":
        return {name: value.detach().cpu() for name, value in model.state_dict().items()}
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = model.state_dict()
    return {
        name: value.detach().cpu()
        for name, value in state.items()
        if name in names or name.startswith("captioner.")
    }


@torch.no_grad()
def validate(model, loader, cfg, tokens, device):
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = move(batch, device)
        logits = model(batch["a"], batch["b"], batch["tokens"][:, :-1])
        loss = caption_loss(logits, batch["tokens"][:, 1:], tokens.pad, 0.0)
        total += float(loss) * batch["a"].size(0)
        count += batch["a"].size(0)
    return total / max(count, 1)


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
        train_set, batch_size=int(cfg["batch_size"]), shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)), pin_memory=device.type == "cuda",
        drop_last=True, generator=generator,
    )
    val_loader = DataLoader(
        val_set, batch_size=int(cfg.get("val_batch_size", cfg["batch_size"])), shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)), pin_memory=device.type == "cuda",
    )
    student = build_model(cfg, tokens.vocab).to(device)
    teacher = frozen_teacher(student) if args.variant == "m2" else None
    backbone, main = [], []
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if name.startswith("encoder.") else main).append(parameter)
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": float(cfg.get("backbone_lr", 1e-6))})
    if main:
        groups.append({"params": main, "lr": float(cfg.get("main_lr", 5e-5))})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(cfg.get("weight_decay", 1e-4)))
    epochs = int(cfg.get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, stale, history = float("inf"), 0, []

    for epoch in range(epochs):
        train_set.set_epoch(epoch)
        student.train()
        if not bool(cfg.get("ca_train_encoder", False)) and cfg["architecture"] == "change_agent":
            student.encoder.eval()
        sums = {"caption": 0.0, "kd": 0.0, "total": 0.0}
        seen = 0
        started = time.time()
        for step, batch in enumerate(train_loader, 1):
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            decoder_in, targets = batch["tokens"][:, :-1], batch["tokens"][:, 1:]
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                student_logits = student(batch["a"], batch["b"], decoder_in)
                cap = caption_loss(
                    student_logits, targets, tokens.pad,
                    float(cfg.get("label_smoothing", 0.05)),
                )
                kd = cap.new_zeros(())
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(
                            batch["clean_a"], batch["clean_b"], decoder_in
                        )
                    kd = token_kd(
                        student_logits, teacher_logits, targets, tokens.pad,
                        float(cfg.get("kd_temperature", 2.0)),
                    )
                total = cap + float(cfg.get("lambda_token_kd", 0.5)) * kd
                if hasattr(student, "encoder") and hasattr(student.encoder, "backbone_anchor_loss"):
                    total = total + float(cfg.get("lambda_backbone_anchor", 0.001)) * student.encoder.backbone_anchor_loss()
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad],
                float(cfg.get("grad_clip", 5.0)),
            )
            scaler.step(optimizer)
            scaler.update()
            size = batch["a"].size(0)
            seen += size
            sums["caption"] += float(cap.detach()) * size
            sums["kd"] += float(kd.detach()) * size
            sums["total"] += float(total.detach()) * size
            if args.smoke_batches and step >= args.smoke_batches:
                break
        scheduler.step()
        val = validate(student, val_loader, cfg, tokens, device)
        row = {
            "epoch": epoch,
            **{key: value / max(seen, 1) for key, value in sums.items()},
            "val_loss": val,
            "seconds": time.time() - started,
        }
        history.append(row)
        print(row, flush=True)
        save_json(history, output / "history.json")
        improved = val < best - float(cfg.get("min_delta", 1e-5))
        if improved:
            best, stale = val, 0
        else:
            stale += 1
        payload = {
            "architecture": cfg["architecture"], "variant": args.variant,
            "epoch": epoch, "best_val_loss": best, "config": cfg,
            "trainable_state": overlay_state(student, cfg["architecture"]),
        }
        atomic_save(payload, output / "last.pt")
        if improved:
            atomic_save(payload, output / "best.pt")
        if args.smoke_batches:
            break
        if stale >= int(cfg.get("patience", 6)):
            break


if __name__ == "__main__":
    main()
